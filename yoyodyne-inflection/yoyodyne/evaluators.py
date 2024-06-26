"""Evaluators."""

from __future__ import annotations
import dataclasses
from typing import Iterator, List

import torch
from torch.nn import functional
from .data import indexes
import sacrebleu


class Error(Exception):
    pass


@dataclasses.dataclass
class EvalItem:
    num_correct: int
    num_predicted: int

    @property
    def accuracy(self) -> float:
        """Computes the accuracy."""
        return self.num_correct / self.num_predicted

    def __add__(self, other_eval: EvalItem) -> EvalItem:
        """Adds two EvalItems by summing along both attributes.

        Args:
            other_eval (EvalItem): The other eval item to add to self.

        Returns:
            EvalItem.
        """
        return EvalItem(
            self.num_correct + other_eval.num_correct,
            self.num_predicted + other_eval.num_predicted,
        )

    def __radd__(self, start_val: int) -> EvalItem:
        """Reverse add. Expects a zero-valued integer.

        Args:
            start_val (int): An initial value for calling the first add in an
                iterable. Expected to be 0.

        Returns:
            EvalItem.
        """
        return EvalItem(
            self.num_correct + start_val, self.num_predicted + start_val
        )


class Evaluator:
    """Evaluates predictions."""

    def evaluate(
        self,
        predictions: torch.Tensor,
        golds: torch.Tensor,
        source_mapping,
        target_mapping,
        end_idx: int,
        pad_idx: int,
    ) -> EvalItem:
        """Computes the exact word match accuracy.

        Args:
            predictions (torch.Tensor): B x vocab_size x seq_len.
            golds (torch.Tensor): B x seq_len x 1.
            end_idx (int): end of sequence index.
            pad_idx (int): padding index.

        Returns:
            float: exact accuracy.
        """
        if predictions.size(0) != golds.size(0):
            raise Error(
                f"Preds batch size ({predictions.size(0)}) and "
                f"golds batch size ({golds.size(0)} do not match"
            )
        # Gets the max value at each dim2 in predictions.
        _, predictions = torch.max(predictions, dim=2)
        # Finalizes the predictions.
        predictions = self.finalize_predictions(predictions, source_mapping,
                                                target_mapping,
                                                end_idx, pad_idx)
        predictions_string = self.decode_preds(predictions, target_mapping)
        predictions_string = [pred.capitalize() for pred in predictions_string]
        return predictions_string

    @staticmethod
    def get_acc(preds, target):
        accuracy = (
            sum([int(r == p) for r, p in zip(target, preds)]) / len(target) * 100
        )
        return accuracy

    @staticmethod
    def get_bleu_chrf(
        predictions_string: torch.Tensor,
        golds_string: torch.Tensor,
    ):

        bleu = sacrebleu.corpus_bleu(
            predictions_string, [golds_string]).format(score_only=True)
        chrf = sacrebleu.corpus_chrf(
            predictions_string, [golds_string]).format(score_only=True)

        return float(bleu), float(chrf)

    @staticmethod
    def get_eval_item(
        predictions: torch.Tensor,
        golds: torch.Tensor,
        pad_idx: int,
    ) -> EvalItem:
        if predictions.size(1) > golds.size(1):
            predictions = predictions[:, : golds.size(1)]
        elif predictions.size(1) < golds.size(1):
            num_pads = (0, golds.size(1) - predictions.size(1))
            predictions = functional.pad(
                predictions, num_pads, "constant", pad_idx
            )
        # Gets the count of exactly matching tensors in the batch.
        # -> B x max_seq_len.
        corr_count = torch.where(
            (predictions.to(golds.device) == golds).all(dim=1)
        )[0].size()[0]
        return EvalItem(
            num_correct=corr_count, num_predicted=predictions.size(0)
        )

    @staticmethod
    def finalize_predictions(
        predictions: torch.Tensor,
        source_mapping,
        target_mapping,
        end_idx: int,
        pad_idx: int,
    ) -> torch.Tensor:
        """Finalizes predictions.

        Cuts off tensors at the first end_idx, and replaces the rest of the
        predictions with pad_idx, as these are erroneously decoded while the
        rest of the batch is finishing decoding.

        Args:
            predictions (torch.Tensor): prediction tensor.
            end_idx (int).
            pad_idx (int).

        Returns:
            torch.Tensor: finalized predictions.
        """
        # Not necessary if batch size is 1.
        if predictions.size(0) == 1:
            return predictions
        for i, prediction in enumerate(predictions):
            # Gets first instance of EOS.
            eos = (prediction == end_idx).nonzero(as_tuple=False)
            if len(eos) > 0 and eos[0].item() < len(prediction):
                # If an EOS was decoded and it is not the last one in the
                # sequence.
                eos = eos[0]
            else:
                # Leaves predictions[i] alone.
                continue
            # Hack in case the first prediction is EOS. In this case
            # torch.split will result in an error, so we change these 0's to
            # 1's, which will make the entire sequence EOS as intended.
            eos[eos == 0] = 1
            symbols, *_ = torch.split(prediction, eos)
            # Replaces everything after with PAD, to replace erroneous decoding
            # While waiting on the entire batch to finish.
            pads = (
                torch.ones(
                    len(prediction) - len(symbols), device=symbols.device
                )
                * pad_idx
            )
            pads[0] = end_idx
            # Makes an in-place update to an inference tensor.
            with torch.inference_mode():
                predictions[i] = torch.cat((symbols, pads))

        return predictions

    @staticmethod
    def decode_preds(
        indices: torch.Tensor,
        symbol_map: indexes.SymbolMap,
    ) -> List[str]:
        symbols_list = _decode(indices, symbol_map)
        predictions_string = []
        for symbols in symbols_list:
            predictions_string.append("".join(symbols))
        return predictions_string


def _decode(
    indices: torch.Tensor,
    symbol_map: indexes.SymbolMap,
) -> Iterator[List[str]]:
    """Decodes the tensor of indices into lists of symbols.

    Args:
        indices (torch.Tensor): 2d tensor of indices.
        symbol_map (indexes.SymbolMap).

    Yields:
        List[str]: Decoded symbols.
    """
    for idx in indices.cpu().numpy():
        yield [
            symbol_map.symbol(c)
            for c in idx
            if c not in {1, 2, 3}  # HARDCODED: Special Tokens
        ]

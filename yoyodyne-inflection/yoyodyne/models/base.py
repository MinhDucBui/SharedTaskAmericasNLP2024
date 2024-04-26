"""Base model class, with PL integration.

This also includes init_embeddings, which has to go somewhere.
"""

import argparse
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn, optim

from .. import data, defaults, evaluators, schedulers, util
from . import modules
from ..data.indexes import Index
import os
import csv


class Error(Exception):
    pass


class BaseEncoderDecoder(pl.LightningModule):
    #  TODO: clean up type checking here.
    model_dir: str
    # Indices.
    end_idx: int
    pad_idx: int
    start_idx: int
    source_mapping: Index
    target_mapping: Index
    # Sizes.
    source_vocab_size: int
    features_vocab_size: int
    target_vocab_size: int
    # Optimizer arguments.
    beta1: float
    beta2: float
    optimizer: str
    scheduler: Optional[str]
    scheduler_kwargs: Optional[Dict]
    # Regularization arguments.
    dropout: float
    label_smoothing: float
    teacher_forcing: bool
    # Decoding arguments.
    beam_width: int
    max_source_length: int
    max_target_length: int
    # Model arguments.
    embedding_size: int
    encoder_layers: int
    decoder_layers: int
    features_encoder_cls: Optional[modules.base.BaseModule]
    hidden_size: int
    source_encoder_cls: modules.base.BaseModule
    # Constructed inside __init__.
    dropout_layer: nn.Dropout
    evaluator: evaluators.Evaluator
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        *,
        model_dir,
        val_targets,
        pad_idx,
        start_idx,
        end_idx,
        source_mapping,
        target_mapping,
        source_vocab_size,
        target_vocab_size,
        source_encoder_cls,
        features_encoder_cls=None,
        features_vocab_size=0,
        tie_embedding=False,
        beta1=defaults.BETA1,
        beta2=defaults.BETA2,
        learning_rate=defaults.LEARNING_RATE,
        optimizer=defaults.OPTIMIZER,
        scheduler=None,
        scheduler_kwargs=None,
        dropout=defaults.DROPOUT,
        label_smoothing=defaults.LABEL_SMOOTHING,
        teacher_forcing=defaults.TEACHER_FORCING,
        beam_width=defaults.BEAM_WIDTH,
        max_source_length=defaults.MAX_SOURCE_LENGTH,
        max_target_length=defaults.MAX_TARGET_LENGTH,
        encoder_layers=defaults.ENCODER_LAYERS,
        decoder_layers=defaults.DECODER_LAYERS,
        embedding_size=defaults.EMBEDDING_SIZE,
        hidden_size=defaults.HIDDEN_SIZE,
        **kwargs,  # Ignored.
    ):
        super().__init__()
        self.val_targets = val_targets
        self.model_dir = self.get_next_versioned_filename(model_dir)
        # Symbol processing.
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.source_mapping = source_mapping
        self.target_mapping = target_mapping
        self.source_vocab_size = source_vocab_size
        self.features_vocab_size = features_vocab_size
        self.target_vocab_size = target_vocab_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.loss_func = self._get_loss_func()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.teacher_forcing = teacher_forcing
        self.beam_width = beam_width
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.decoder_layers = decoder_layers
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.hidden_size = hidden_size
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
        self.evaluator = evaluators.Evaluator()
        # Checks compatibility with feature encoder and dataloader.
        modules.check_encoder_compatibility(
            source_encoder_cls, features_encoder_cls
        )
        # Instantiates encoders class.
        self.source_encoder = source_encoder_cls(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            num_embeddings=self.source_vocab_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            layers=self.encoder_layers,
            hidden_size=self.hidden_size,
            features_vocab_size=features_vocab_size,
            max_source_length=max_source_length,
            **kwargs,
        )
        self.features_encoder = (
            features_encoder_cls(
                pad_idx=self.pad_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                num_embeddings=features_vocab_size,
                dropout=self.dropout,
                embedding_size=self.embedding_size,
                layers=self.encoder_layers,
                hidden_size=self.hidden_size,
                max_source_length=max_source_length,
                **kwargs,
            )
            if features_encoder_cls is not None
            else None
        )
        self.decoder = self.get_decoder()

        if tie_embedding:
            self.decoder.embeddings = self.source_encoder.embeddings

        # Saves hyperparameters for PL checkpointing.
        self.save_hyperparameters(
            ignore=["source_encoder", "decoder", "features_encoder"]
        )
        # Logs the module names.
        util.log_info(f"Model: {self.name}")
        if self.features_encoder is not None:
            util.log_info(f"Source encoder: {self.source_encoder.name}")
            util.log_info(f"Features encoder: {self.features_encoder.name}")
        else:
            util.log_info(f"Encoder: {self.source_encoder.name}")
        util.log_info(f"Decoder: {self.decoder.name}")

    @staticmethod
    def _xavier_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer using Xavier initialization.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Xavier initialization.
        nn.init.normal_(
            embedding_layer.weight, mean=0, std=embedding_size**-0.5
        )
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def _normal_embedding_initialization(
        num_embeddings: int, embedding_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Initializes the embeddings layer from a normal distribution.

        The pad embeddings are also zeroed out.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Returns:
            nn.Embedding: embedding layer.
        """
        embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Zeroes out pad embeddings.
        if pad_idx is not None:
            nn.init.constant_(embedding_layer.weight[pad_idx], 0.0)
        return embedding_layer

    @staticmethod
    def init_embeddings(
        num_embed: int, embed_size: int, pad_idx: int
    ) -> nn.Embedding:
        """Method interface for initializing the embedding layer.

        Args:
            num_embeddings (int): number of embeddings.
            embedding_size (int): dimension of embeddings.
            pad_idx (int): index of pad symbol.

        Raises:
            NotImplementedError: This method needs to be overridden.

        Returns:
            nn.Embedding: embedding layer.
        """
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    @property
    def has_features_encoder(self):
        return self.features_encoder is not None

    def training_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one step of training.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch)
            batch_idx (int).

        Returns:
            torch.Tensor: loss.
        """
        # -> B x seq_len x target_vocab_size.
        predictions = self(batch)
        target_padded = batch.target.padded
        # -> B x target_vocab_size x seq_len. For loss.
        predictions = predictions.transpose(1, 2)
        loss = self.loss_func(predictions, target_padded)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> Dict:
        """Runs one validation step.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            Dict[str, float]: validation metrics.
        """
        # Greedy decoding.
        # -> B x seq_len x target_vocab_size.
        target_padded = batch.target.padded
        greedy_predictions = self(batch)
        predictions_string = self.evaluator.evaluate(
            greedy_predictions, target_padded, self.source_mapping,
            self.target_mapping, self.end_idx, self.pad_idx
        )
        # -> B x target_vocab_size x seq_len. For loss.
        greedy_predictions = greedy_predictions.transpose(1, 2)
        # Truncates predictions to the size of the target.
        greedy_predictions = torch.narrow(
            greedy_predictions, 2, 0, target_padded.size(1)
        )
        loss = self.loss_func(greedy_predictions, target_padded)
        return {"val_loss": loss,
                "predictions_string": predictions_string}

    def validation_epoch_end(self, validation_step_outputs: Dict) -> Dict:
        """Computes average loss and average accuracy.

        Args:
            validation_step_outputs (Dict).

        Returns:
            Dict: averaged metrics over all validation steps.
        """
        num_steps = len(validation_step_outputs)
        avg_val_loss = (
            sum([v["val_loss"] for v in validation_step_outputs]) / num_steps
        )

        predictions_string = [
            item for sublist in validation_step_outputs for item in sublist["predictions_string"]]
        golds_string = self.val_targets

        bleu, chrf = self.evaluator.get_bleu_chrf(predictions_string, golds_string)
        acc = self.evaluator.get_acc(predictions_string, golds_string)
        save_preds_dict = {"Predicted Target": predictions_string,
                           "Target": golds_string}

        self.dict_to_csv(save_preds_dict, self.model_dir)

        metrics = {
            "val_loss": avg_val_loss,
            "acc": acc,
            "bleu": bleu,
            "chrf": chrf
        }

        for metric, value in metrics.items():
            self.log(metric, value, prog_bar=True)
        return metrics

    def predict_step(
        self,
        batch: data.PaddedBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        """Runs one predict step.

        This is called by the PL Trainer.

        Args:
            batch (data.PaddedBatch).
            batch_idx (int).

        Returns:
            torch.Tensor: indices of the argmax at each timestep.
        """

        # QUICK FIX: JUST WORKS FOR TESTING; NO TIME FOR DEBUGGING
        predictions, all_attention_weights = self(batch, return_attention=True)
        # -> B x seq_len x 1.
        greedy_predictions = self._get_predicted(predictions)
        return greedy_predictions, all_attention_weights

    def _get_predicted(self, predictions: torch.Tensor) -> torch.Tensor:
        """Picks the best index from the vocabulary.

        Args:
            predictions (torch.Tensor): B x seq_len x target_vocab_size.

        Returns:
            torch.Tensor: indices of the argmax at each timestep.
        """
        assert len(predictions.size()) == 3
        _, indices = torch.max(predictions, dim=2)
        return indices

    def configure_optimizers(self) -> optim.Optimizer:
        """Gets the configured torch optimizer.

        This is called by the PL Trainer.

        Returns:
            optim.Optimizer: optimizer for training.
        """
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer[0])
        return optimizer, scheduler

    def _get_optimizer(self) -> optim.Optimizer:
        """Factory for selecting the optimizer.

        Returns:
            optim.Optimizer: optimizer for training.
        """
        optim_fac = {
            "adadelta": optim.Adadelta,
            "adam": optim.Adam,
            "sgd": optim.SGD,
        }
        optimizer = optim_fac[self.optimizer]
        kwargs = {"lr": self.learning_rate}
        if self.optimizer == "adam":
            kwargs["betas"] = self.beta1, self.beta2
        return [optimizer(self.parameters(), **kwargs)]

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler:
        """Factory for selecting the scheduler.

        Args:
            optimizer (optim.Optimizer): optimizer.

        Returns:
            optim.lr_scheduler: LR scheduler for training.
        """
        if self.scheduler is None:
            return []
        scheduler_fac = {
            "warmupinvsqrt": schedulers.WarmupInverseSquareRootSchedule,
            "lineardecay": schedulers.LinearDecay,
            "reduceonplateau": schedulers.ReduceOnPlateau,
            "None": None
        }
        scheduler_cls = scheduler_fac[self.scheduler]
        if scheduler_cls is None:
            return []
        scheduler = scheduler_cls(
            **dict(self.scheduler_kwargs, optimizer=optimizer)
        )
        scheduler_cfg = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        # When using `reduceonplateau_mode loss`, we reduce when validation
        # loss stops decreasing. When using `reduceonplateau_mode accuracy`,
        # we reduce when validation accuracy stops increasing.
        if self.scheduler == "reduceonplateau":
            scheduler_cfg["interval"] = "epoch"
            scheduler_cfg["frequency"] = self.scheduler_kwargs[
                "check_val_every_n_epoch"
            ]
            mode = self.scheduler_kwargs.get("reduceonplateau_mode")
            if not mode:
                raise Error("No reduceonplateaumode specified")
            elif mode == "loss":
                scheduler_cfg["monitor"] = "val_loss"
            elif mode == "accuracy":
                scheduler_cfg["monitor"] = "val_chrf"
            else:
                raise Error(f"reduceonplateau mode not found: {mode}")
        return [scheduler_cfg]

    def _get_loss_func(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the actual function used to compute loss.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: configured
                loss function.
        """
        return nn.CrossEntropyLoss(
            ignore_index=self.pad_idx,
            label_smoothing=self.label_smoothing,
        )

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds shared configuration options to the argument parser.

        These are only needed at training time.

        Args:
            parser (argparse.ArgumentParser).
        """
        # Optimizer arguments.
        parser.add_argument(
            "--beta1",
            type=float,
            default=defaults.BETA1,
            help="beta_1 (Adam optimizer only). Default: %(default)s.",
        )
        parser.add_argument(
            "--beta2",
            type=float,
            default=defaults.BETA2,
            help="beta_2 (Adam optimizer only). Default: %(default)s.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=defaults.LEARNING_RATE,
            help="Learning rate. Default: %(default)s.",
        )
        parser.add_argument(
            "--optimizer",
            choices=["adadelta", "adam", "sgd"],
            default=defaults.OPTIMIZER,
            help="Optimizer. Default: %(default)s.",
        )
        # Regularization arguments.
        parser.add_argument(
            "--dropout",
            type=float,
            default=defaults.DROPOUT,
            help="Dropout probability. Default: %(default)s.",
        )
        parser.add_argument(
            "--label_smoothing",
            type=float,
            default=defaults.LABEL_SMOOTHING,
            help="Coefficient for label smoothing. Default: %(default)s.",
        )
        # TODO: add --beam_width.
        # Model arguments.
        parser.add_argument(
            "--decoder_layers",
            type=int,
            default=defaults.DECODER_LAYERS,
            help="Number of decoder layers. Default: %(default)s.",
        )
        parser.add_argument(
            "--encoder_layers",
            type=int,
            default=defaults.ENCODER_LAYERS,
            help="Number of encoder layers. Default: %(default)s.",
        )
        parser.add_argument(
            "--embedding_size",
            type=int,
            default=defaults.EMBEDDING_SIZE,
            help="Dimensionality of embeddings. Default: %(default)s.",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=defaults.HIDDEN_SIZE,
            help="Dimensionality of the hidden layer(s). "
            "Default: %(default)s.",
        )

    @staticmethod
    # define the function
    def dict_to_csv(dictionary, filename):
        keys = list(dictionary.keys())
        values = list(dictionary.values())

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            writer.writerows(zip(*values))

    @staticmethod
    def get_next_versioned_filename(base_filename):
        version = 0
        while True:
            versioned_filename = f"version_{version}"
            folder_base = os.path.join(base_filename, versioned_filename)
            if not os.path.exists(folder_base):
                return os.path.join(folder_base, "predictions.csv")
            version += 1

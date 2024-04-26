"""Prediction."""

import argparse
import os
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import torch
from . import data, models, util


def get_trainer_from_argparse_args(
    args: argparse.Namespace,
) -> pl.Trainer:
    """Creates the trainer from CLI arguments.

    Args:
        args (argparse.Namespace).

    Return:
        pl.Trainer.
    """
    return pl.Trainer.from_argparse_args(args, max_epochs=0)


def get_datamodule_from_argparse_args(
    args: argparse.Namespace,
) -> data.DataModule:
    """Creates the dataset from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        data.DataModule.
    """
    separate_features = args.features_col != 0 and args.arch in [
        "pointer_generator_lstm",
        "transducer",
    ]
    index = data.Index.read(args.model_dir, args.experiment)
    return data.DataModule(
        predict=args.predict,
        batch_size=args.batch_size,
        source_col=args.source_col,
        features_col=args.features_col,
        target_col=args.target_col,
        source_sep=args.source_sep,
        features_sep=args.features_sep,
        target_sep=args.target_sep,
        separate_features=separate_features,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        index=index,
    )


def get_model_from_argparse_args(
    args: argparse.Namespace,
) -> models.BaseEncoderDecoder:
    """Creates the model from CLI arguments.

    Args:
        args (argparse.Namespace).

    Returns:
        models.BaseEncoderDecoder.
    """
    model_cls = models.get_model_cls_from_argparse_args(args)
    return model_cls.load_from_checkpoint(args.checkpoint)


def _mkdir(output: str) -> None:
    """Creates directory for output file if necessary.

    Args:
        output (str): output to output file.
    """
    dirname = os.path.dirname(output)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def get_element_at_index(batch_list, global_index):
    cumulative_batch_sizes = torch.cumsum(torch.tensor([batch.size(0) for batch in batch_list]), dim=0)
    batch_index = torch.sum(global_index >= cumulative_batch_sizes).item()
    within_batch_index = global_index - cumulative_batch_sizes[batch_index - 1] if batch_index > 0 else global_index
    return batch_list[batch_index][within_batch_index]


def predict(
    trainer: pl.Trainer,
    model: models.BaseEncoderDecoder,
    datamodule: data.DataModule,
    output: str,
    target_col: int
) -> None:
    """Predicts from the model.

    Args:
         trainer (pl.Trainer).
         model (pl.LightningModule).
         datamdule (data.DataModule).
         output (str).
    """
    util.log_info(f"Writing to {output}")
    output = output + "/predictions.csv"
    _mkdir(output)
    loader = datamodule.predict_dataloader()

    predictions = []
    all_attention_weights = []
    for batch in loader:
        preds = decode_batch(batch.source.padded, model, datamodule)
        for index, i in enumerate(preds):
            print(i)
            print(batch.source.padded[index])
        preds = decode_batch(batch.features.padded, model, datamodule)
        print(model.evaluator.decode_preds(batch.features.padded, datamodule.index.features_map))

    for batch, attention_weights in trainer.predict(model, loader):
        all_attention_weights.append(attention_weights)
        preds = decode_batch(batch, model, datamodule)
        predictions += preds

    print("skjdnaskdnaskn")
    # Decode UNK Tokens
    inputs = ["".join(example[0]) for example in datamodule.predict_dataloader().dataset.samples]
    for index, preds in enumerate(predictions):
        # BREAKS IF MULTIPLE <UNK>!!
        if "<UNK>" in preds:
            unk_index = preds.find("<UNK>")
            weights = get_element_at_index(all_attention_weights, index)
            source_att_weight = weights[unk_index]

            _, source_index = torch.max(source_att_weight, dim=0)
            source_index = source_index.item() - 1 # Because of start token index
            input_split = datamodule.parser.source_symbols(inputs[index])
            source_character = input_split[source_index]
            predictions[index] = preds.replace("<UNK>", source_character)
            print(predictions[index] )

    if target_col:
        targets = ["".join(example[2]) for example in datamodule.predict_dataloader().dataset.samples]

    if targets:
        preds_dict = {"Predicted Target": predictions, "Target": targets}
    else:
        preds_dict = {"Predicted Target": predictions}

    model.dict_to_csv(preds_dict, output)


def decode_batch(batch, model, datamodule):
    decoded_batch = []
    batch = model.evaluator.finalize_predictions(
            batch, datamodule.index.source_map, 
            datamodule.index.target_map, 
            datamodule.index.end_idx, 
            datamodule.index.pad_idx
    )
    for prediction in model.evaluator.decode_preds(batch, datamodule.index.target_map):
        decoded_batch.append(prediction)
    return decoded_batch


def add_argparse_args(parser: argparse.ArgumentParser) -> None:
    """Adds prediction arguments to parser.

    Args:
        parser (argparse.ArgumentParser).
    """
    # Path arguments.
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to output model directory.",
    )
    parser.add_argument(
        "--experiment", required=True, help="Name of experiment."
    )
    parser.add_argument(
        "--predict",
        required=True,
        help="Path to prediction input data TSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to prediction output data TSV.",
    )
    # Prediction arguments.
    # TODO: add --beam_width.
    # Data arguments.
    data.add_argparse_args(parser)
    # Architecture arguments; the architecture-specific ones are not needed.
    models.add_argparse_args(parser)
    # Among the things this adds, the following are likely to be useful:
    # --accelerator ("gpu" for GPU)
    # --devices (for multiple device support)
    pl.Trainer.add_argparse_args(parser)


def main() -> None:
    """Predictor."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_argparse_args(parser)
    args = parser.parse_args()
    util.log_arguments(args)
    trainer = get_trainer_from_argparse_args(args)
    datamodule = get_datamodule_from_argparse_args(args)
    model = get_model_from_argparse_args(args)
    predict(trainer, model, datamodule, args.output, args.target_col)


if __name__ == "__main__":
    main()

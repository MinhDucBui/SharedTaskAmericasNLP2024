import optuna
from subprocess import run
import argparse
import os
import pandas as pd
import json
import copy

# Construct the command for your training script
COMMAND = [
    'yoyodyne-train',
    '--model_dir', 'models',
    '--experiment', 'baseline',
    '--train', '../americasnlp2024/ST2_EducationalMaterials/data/LANGUAGE-train.tsv',
    '--val', '../americasnlp2024/ST2_EducationalMaterials/data/LANGUAGE-dev.tsv',
    '--log_wandb',
    '--patience', '10',
    '--source_col', '2',
    '--features_col', '3',
    '--target_col', '4',
    '--features_sep', ', ',
    '--max_epochs', '100',
    '--check_val_every_n_epoch', '2',
    '--log_every_n_step', '100',
    '--optimizer', 'adam',
    '--seed', '49',
    '--accelerator', 'gpu',
    '--enable_checkpointing', "True"
]

METRIC_PATH = None
ARGS = None


def define_search_space_transformer_pointer(trial):
    space = {
        'source_attention_heads': trial.suggest_categorical('source_attention_heads', [2, 4, 8]),
        # 'features_attention_heads': trial.suggest_categorical('features_attention_heads', [1, 2]),
        'batch_size': trial.suggest_int('batch_size', 2, 128, log=True),
        # 'beta1': trial.suggest_float('beta1', 0.8, 0.999),
        # 'beta2': trial.suggest_float('beta2', 0.98, 0.999),
        'decoder_layers': trial.suggest_categorical('decoder_layers', [2, 4, 6]),
        'dropout': trial.suggest_float('dropout', 0, 0.5),
        'embedding_size': trial.suggest_int('embedding_size', 16, 512, step=16),
        'encoder_layers': trial.suggest_categorical('encoder_layers', [2, 4, 6]),
        'reduceonplateau_factor': trial.suggest_float('reduceonplateau_factor', 0.1, 0.9),
        'hidden_size': trial.suggest_int('hidden_size', 64, 2048, step=64),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-04, 0.001, log=True),
        'min_learning_rate': trial.suggest_float('min_learning_rate', 1e-06, 0.001),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 400, step=10),
        'reduceonplateau_patience': trial.suggest_int('reduceonplateau_patience', 1, 5),
        'reduceonplateau_mode': trial.suggest_categorical('reduceonplateau_mode', ['loss']),
        'scheduler': trial.suggest_categorical('scheduler', ['reduceonplateau', 'warmupinvsqrt', None])
    }
    return space


def define_search_space_transformer(trial):
    space = {
        'source_attention_heads': trial.suggest_categorical('attention_heads', [2, 4, 8]),
        'batch_size': trial.suggest_int('batch_size', 2, 128, log=True),
        'beta1': trial.suggest_float('beta1', 0.8, 0.999),
        'beta2': trial.suggest_float('beta2', 0.98, 0.999),
        'decoder_layers': trial.suggest_categorical('decoder_layers', [2, 4, 6, 8]),
        'dropout': trial.suggest_float('dropout', 0, 0.5),
        'embedding_size': trial.suggest_int('embedding_size', 16, 512, step=16),
        'encoder_layers': trial.suggest_categorical('encoder_layers', [2, 4, 6, 8]),
        'reduceonplateau_factor': trial.suggest_float('reduceonplateau_factor', 0.1, 0.9),
        'hidden_size': trial.suggest_int('hidden_size', 64, 2048, step=64),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-05, 0.01, log=True),
        'min_learning_rate': trial.suggest_float('min_learning_rate', 1e-06, 0.001),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 400, step=10),
        'reduceonplateau_patience': trial.suggest_int('reduceonplateau_patience', 1, 5),
        'reduceonplateau_mode': trial.suggest_categorical('reduceonplateau_mode', ['loss']),
        'scheduler': trial.suggest_categorical('scheduler', ['reduceonplateau', 'warmupinvsqrt', None])
    }
    return space


def define_search_space_lstm(trial):
    space = {
        'source_attention_heads': trial.suggest_categorical('attention_heads', [1]),
        'batch_size': trial.suggest_int('batch_size', 2, 128, log=True),
        'beta1': trial.suggest_float('beta1', 0.8, 0.999),
        'beta2': trial.suggest_float('beta2', 0.98, 0.999),
        'encoder_layers': trial.suggest_categorical('encoder_layers', [1, 2]),
        'decoder_layers': trial.suggest_categorical('decoder_layers', [1]),
        'dropout': trial.suggest_float('dropout', 0, 0.5),
        'embedding_size': trial.suggest_int('embedding_size', 16, 512, step=16),
        'reduceonplateau_factor': trial.suggest_float('reduceonplateau_factor', 0.1, 0.9),
        'hidden_size': trial.suggest_int('hidden_size', 64, 2048, step=64),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-05, 0.01),
        'min_learning_rate': trial.suggest_float('min_learning_rate', 1e-06, 0.001),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000, step=10),
        'reduceonplateau_patience': trial.suggest_int('reduceonplateau_patience', 1, 5),
        'reduceonplateau_mode': trial.suggest_categorical('reduceonplateau_mode', ['loss']),
        'scheduler': trial.suggest_categorical('scheduler', ['reduceonplateau', 'warmupinvsqrt', None])
    }
    return space


def define_search_space_lstm_pointer(trial):
    space = {
        'source_attention_heads': trial.suggest_categorical('source_attention_heads', [1]),
        'features_attention_heads': trial.suggest_categorical('features_attention_heads', [1, 2]),
        'batch_size': trial.suggest_int('batch_size', 2, 128, log=True),
        'beta1': trial.suggest_float('beta1', 0.8, 0.999),
        'beta2': trial.suggest_float('beta2', 0.98, 0.999),
        'dropout': trial.suggest_float('dropout', 0, 0.5),
        'embedding_size': trial.suggest_int('embedding_size', 16, 512, step=16),
        'encoder_layers': trial.suggest_categorical('encoder_layers', [1, 2]),
        'reduceonplateau_factor': trial.suggest_float('reduceonplateau_factor', 0.1, 0.9),
        'hidden_size': trial.suggest_int('hidden_size', 64, 2048, step=64),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-05, 0.01),
        'min_learning_rate': trial.suggest_float('min_learning_rate', 1e-06, 0.001),
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000, step=10),
        'reduceonplateau_patience': trial.suggest_int('reduceonplateau_patience', 1, 5),
        'reduceonplateau_mode': trial.suggest_categorical('reduceonplateau_mode', ['loss']),
        'scheduler': trial.suggest_categorical('scheduler', ['reduceonplateau', 'warmupinvsqrt', None])
    }

    return space


def pretraining_space(trial):
    space = {
        'warmup_steps': trial.suggest_int('warmup_steps', 0, 10000, step=100),
        'early_every_n_epochs': 0,
        'early_every_n_train_steps': 0,
        # 'val_check_interval': 600,
        'val_check_interval': 200,
        'check_val_every_n_epoch': 0,
        'max_epochs': '1',
        'patience': '3'
    }
    return space


def delete_lower_chrf_file(model_dir, pretraining):
    files = [f for f in os.listdir(model_dir) if f.startswith(
        'model-') and f.endswith('.ckpt')]

    if len(files) < 2:
        print("Not enough files to compare.")
        return

    chrf_values = [float(f.split('=')[1].split('.ckpt')[
                         0].split("-")[0]) for f in files]
    if pretraining:
        min_chrf_value = max(chrf_values)
    else:
        min_chrf_value = min(chrf_values)
    min_chrf_index = chrf_values.index(min_chrf_value)
    file_to_delete = files[min_chrf_index]
    remove_ckpt = os.path.join(model_dir, file_to_delete)
    # Check if there are more than 10 files before attempting deletion
    if len(chrf_values) > 3:
        if os.path.isfile(remove_ckpt):
            os.remove(remove_ckpt)
        print(f"File '{remove_ckpt}' deleted.")
    else:
        print("Not enough files to delete. Skipping deletion.")


def get_next_versioned_filename(base_filename):
    version = 0
    while True:
        versioned_filename = f"version_{version}"
        folder_base = os.path.join(base_filename, versioned_filename)
        if not os.path.exists(folder_base):
            return os.path.join(folder_base, "metrics.csv")
        version += 1

# General objective function


def general_objective(trial, define_search_space, model_type):
    metric_file = get_next_versioned_filename(METRIC_PATH)
    # Define the hyperparameter search space
    hyperparameters = define_search_space(trial)

    # Ensure that decoder_layers match encoder_layers for certain models
    if model_type == 'lstm_pointer':
        hyperparameters['decoder_layers'] = hyperparameters['encoder_layers']

    command_run = copy.deepcopy(COMMAND)
    for key, value in hyperparameters.items():
        command_run += ["--" + key, str(value)]

    command_run = replace_with_arg(command_run)
    if ARGS.pretraining:
        hyperparameters = pretraining_space(trial)
        for key, value in hyperparameters.items():
            command_run += ["--" + key, str(value)]

    # Run the training script
    result = run(command_run, check=True)

    # Save Metric
    hp_file = metric_file.split("metrics.csv")[0] + "hyperparameter.json"
    with open(hp_file, 'w') as json_file:
        json.dump(hyperparameters, json_file)

    # Extract the validation accuracy or any other metric from the training result
    df = pd.read_csv(metric_file)
    latest_bleu = df['chrf'].dropna().max()

    # Delete Checkpoints
    delete_lower_chrf_file(METRIC_PATH, ARGS.pretraining)

    return latest_bleu

# Specific objective functions


def objective_lstm(trial):
    return general_objective(trial, define_search_space_lstm, 'lstm')


def objective_lstm_pointer(trial):
    return general_objective(trial, define_search_space_lstm_pointer, 'lstm_pointer')


def objective_transformer(trial):
    return general_objective(trial, define_search_space_transformer, 'transformer')


def objective_transformer_pointer(trial):
    return general_objective(trial, define_search_space_transformer_pointer, 'transformer_pointer')


def replace_with_arg(command):

    # Resume Checkpoint
    if "--checkpoint" in command or "--fix_arch" in command:
        new_args = ["--source_attention_heads",
                    "--features_attention_heads",
                    "--embedding_size",
                    "--encoder_layers",
                    "--hidden_size",
                    "--decoder_layers"]
        # Replace values in arch_commands with new args values
        for i in range(0, len(command)):
            if command[i] in new_args:
                command[i + 1] = getattr(ARGS, command[i][2:])
    if "--fix_arch" in command:
        command.remove("--fix_arch")
    return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Your script description here.')
    parser.add_argument('--arch', choices=['transformer', 'lstm', 'transformer_pointer',
                        'lstm_pointer'], help='Specify the architecture (transformer or lstm)', required=True)
    parser.add_argument('--tie_embedding',
                        action='store_true',
                        help='Tying Embedding')
    parser.add_argument('--experiment', type=str,
                        help='Name of experiment', required=True)
    parser.add_argument('--language', help='Language', required=True)
    parser.add_argument(
        "--trials",
        type=int,
        help="Number of trials",
    )
    parser.add_argument('--checkpoint', type=str,
                        help='Path to the checkpoint file', default=None)
    parser.add_argument('--source_attention_heads', type=str,
                        default=1, help='Number of attention heads for source')
    parser.add_argument('--features_attention_heads', type=str,
                        default=1, help='Number of attention heads for features')
    parser.add_argument('--embedding_size', type=str,
                        default=176, help='Size of the embedding')
    parser.add_argument('--encoder_layers', type=str,
                        default=1, help='Number of encoder layers')
    parser.add_argument('--hidden_size', type=str,
                        default=1280, help='Size of the hidden layer')
    parser.add_argument('--decoder_layers', type=str,
                        default=1, help='Number of decoder layers')
    parser.add_argument('--monitor_metric', type=str,
                        default="acc", help='Monitor which metric')
    parser.add_argument('--fix_arch',
                        action='store_true',
                        help='Tying Embedding')
    parser.add_argument('--all_features',
                        action='store_true',
                        help='HACK: Use all features.')
    parser.add_argument('--pretraining',
                        action='store_true',
                        help='Use pretraining HP space.')

    args = parser.parse_args()
    ARGS = args  # HACK

    if args.tie_embedding:
        COMMAND += ['--tie_embedding']
    if args.all_features:
        COMMAND += ['--all_features']

    for index, argument in enumerate(COMMAND):
        if "LANGUAGE" in argument:
            COMMAND[index] = COMMAND[index].replace("LANGUAGE", args.language)

    if args.arch == "transformer":
        objective = objective_transformer
        COMMAND += ['--arch', 'transformer']
    elif args.arch == "lstm":
        objective = objective_lstm
        COMMAND += ['--arch', 'lstm']
    elif args.arch == "lstm_pointer":
        objective = objective_lstm_pointer
        COMMAND += ['--features_encoder_arch', 'lstm']
        COMMAND += ['--arch', 'pointer_generator_lstm']
    elif args.arch == "transformer_pointer":
        objective = objective_transformer_pointer
        COMMAND += ['--features_encoder_arch', 'linear']
        COMMAND += ['--arch', 'pointer_generator_transformer']

    # Resume Checkpoint
    if args.checkpoint:
        COMMAND += ["--checkpoint", args.checkpoint]
    if args.fix_arch:
        COMMAND += ["--fix_arch"]
    if args.monitor_metric:
        COMMAND += ["--monitor_metric", args.monitor_metric]

    COMMAND += ['--experiment', args.experiment]
    METRIC_PATH = COMMAND[2] + "/" + COMMAND[-1] + "/"

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)

    # Print the best hyperparameters and their corresponding value
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    print(f"Best value: {best_trial}")

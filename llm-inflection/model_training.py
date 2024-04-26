# Load Libraries
# pip install transformers
# pip install peft
# pip install pandas
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import pandas as pd
from tqdm import tqdm
import pprint
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset
import pickle
import argparse
import sacrebleu


# Replace with your model folder
CACHE_DIR = ""
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Replace Paths
OUTPUT_PATH = ""
DATAFOLDER = ""


# Dataset
SEP_TOKEN = "[/INST]"
INPUT_LENGTH = 512
LABEL_LENGTH = 50

# LoRA
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1


def prompt_prefix():
    message = [
        # {"role": "system", "content": "You are a helpful assistant who follows the following pattern."},
        {"role": "user",
            "content": f"This is {LANGUAGE.capitalize()}. I will give you a source sentence and a grammar change and you have to output the correct change!"},
        {"role": "assistant", "content": "Ok!"},
    ]
    return message


def give_example(source, change, target=None):
    user_prompt = {
        "role": "user", "content": "Source Sentence: {} \ Grammar Change: {}".format(source, change)}
    if target:
        assistant_prompt = {"role": "assistant", "content": " " + target}
        return [user_prompt, assistant_prompt]
    return [user_prompt]

# Function to count overlapping elements between two columns


def count_overlapping_elements(row):
    # Assuming the values are space-separated strings
    set_change = set(row["Change"].split(","))
    set_current_change = set(row["Current_Change"].split(","))
    return len(set_change.intersection(set_current_change))


def get_examples(df, current_change, max_examples=2, random_state=None):
    df["Current_Change"] = current_change
    # Create a new column with the count of overlapping elements
    df['overlap_count'] = df.apply(count_overlapping_elements, axis=1)
    df = df[df['overlap_count'] > 0].sort_values(
        by='overlap_count', ascending=False)
    df = df[:max_examples]
    # df = df.sample(frac=1, random_state=random_state)
    examples = []
    for _, row_train in df[:max_examples].iterrows():
        example = give_example(
            row_train["Source"], row_train["Change"], row_train["Target"])
        examples += example
    return examples


def generate_examples(tokenized_prompts, prefix=""):
    answers = []

    for index, prompt in tqdm(enumerate(tokenized_prompts)):
        # print("\n-----Example {}----".format(index+1))

        # Replace Padding Token
        # print("Input Text: {}".format(input_text.replace("!", "")))
        # "!" = 28808
        output = model.generate(
            **prompt, max_new_tokens=50, pad_token_id=28808)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        answers.append(generated_text)

    # Save answers list into a pickle file
    with open(os.path.join(OUTPUT_PATH, prefix + '_answers.pickle'), 'wb') as f:
        pickle.dump(answers, f)

    return answers


def tokenize(prompt, tokenizer, return_tensors=None,
             cutoff_len=1500, padding=True, add_special_tokens=True):
    if padding:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
    else:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )


def get_first_folder(directory):
    # Get the list of contents in the directory
    contents = os.listdir(directory)

    # Iterate through the contents to find the first folder
    for item in contents:
        # Check if the item is a directory
        if os.path.isdir(os.path.join(directory, item)):
            # Return the first folder found
            return os.path.join(directory, item)
    return None  # Return None if no folders found


class CustomDataset(Dataset):
    def __init__(self, prompts, outputs):
        self.prompts = prompts
        self.outputs = outputs

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        # We ignore input text, except where the labels begin
        ignore_input_text = torch.tensor(
            [-100] * (INPUT_LENGTH + LABEL_LENGTH - len(self.outputs[index]['input_ids'][0])))
        label_ids = torch.cat(
            (ignore_input_text, self.outputs[index]["input_ids"][0]), dim=0)
        attention_mask = self.prompts[index]['attention_mask']
        input_ids = self.prompts[index]['input_ids']

        return {'input_ids': input_ids[0], 'attention_mask': attention_mask[0], 'labels': label_ids}


def get_acc(preds, target):
    accuracy = (
        sum([int(r == p) for r, p in zip(target, preds)]) / len(target) * 100
    )
    return accuracy


def get_bleu_chrf(
    predictions_string: torch.Tensor,
    golds_string: torch.Tensor,
):

    bleu = sacrebleu.corpus_bleu(
        predictions_string, [golds_string]).format(score_only=True)
    chrf = sacrebleu.corpus_chrf(
        predictions_string, [golds_string]).format(score_only=True)

    return float(bleu), float(chrf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Your script description here.')
    parser.add_argument("-m", '--multiplier', type=int,
                        help='Folder', required=True)
    parser.add_argument("-k", '--k_shots', type=int,
                        help='Folder', required=True)
    parser.add_argument("-p", '--prefix', type=str,
                        help='Folder', required=True)
    parser.add_argument("-n", '--name', type=str, help='Folder', required=True)
    parser.add_argument("-q", '--quantized', type=bool,
                        help='Folder', required=True)
    parser.add_argument("-l", '--language', type=str,
                        help='Folder', required=True)
    parser.add_argument("-r", '--learning_rate', type=float,
                        help='Folder', default=1e-4)
    args = parser.parse_args()
    N_MULTIPLIER = args.multiplier
    N_EXAMPLES = args.k_shots
    PREFIX = args.prefix
    MODEL_NAME = args.name
    load_in_4bit = args.quantized
    LANGUAGE = args.language
    LR = args.learning_rate
    print("N_MULTIPLIER:", N_MULTIPLIER)
    print("N_EXAMPLES:", N_EXAMPLES)
    print("PREFIX:", PREFIX)
    print("MODEL_NAME:", MODEL_NAME)
    print("load_in_4bit:", load_in_4bit)
    print("LANGUAGE:", LANGUAGE)
    print("LR:", LR)

    INPUT_LENGTH = int((INPUT_LENGTH/5) * N_EXAMPLES)
    print("INPUT_LENGTH:", INPUT_LENGTH)

    # Data Folder
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{LANGUAGE}")
    FILE_PATH_TRAIN = os.path.join(DATAFOLDER, f"{LANGUAGE}-train.tsv")
    FILE_PATH_DEV = os.path.join(DATAFOLDER, f"{LANGUAGE}-dev.tsv")

    # Load Tokenizer & Model from local
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, padding_side='left')
    tokenizer.pad_token = "!"  # Not EOS, will explain another time.\
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                 cache_dir=CACHE_DIR, load_in_4bit=load_in_4bit,
                                                 device_map="auto")

    # Load Data
    df_train = pd.read_csv(FILE_PATH_TRAIN, sep='\t')
    df_dev = pd.read_csv(FILE_PATH_DEV, sep='\t')

    # Create prompts for training:
    all_prompts = []
    for i in range(N_MULTIPLIER):
        for index_dev, (_, row) in enumerate(df_train.iterrows()):
            message = prompt_prefix()
            current_change = row["Change"]
            df_without_test = df_train.drop(index_dev)

            examples = get_examples(
                df_without_test, current_change, max_examples=N_EXAMPLES)
            test_example = give_example(
                row["Source"], row["Change"], row["Target"])
            message += examples
            message += test_example
            all_prompts.append(message)

    print("----EXAMPLE PROMPT FROM TRAINING----")
    pprint.pp(all_prompts[0])
    pprint.pp(all_prompts[1])
    pprint.pp(all_prompts[2])
    pprint.pp(all_prompts[3])

    tokenized_prompts = []
    all_labels = []
    for prompt in tqdm(all_prompts):
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True)
        prompt_splitted = prompt.split(SEP_TOKEN)
        # input_str = SEP_TOKEN.join(prompt_splitted[:-1]) + SEP_TOKEN
        # 1: because we have added a white space
        output_str = prompt_splitted[-1][1:]
        input_tokenized = tokenize(
            prompt, tokenizer, return_tensors="pt", cutoff_len=INPUT_LENGTH+LABEL_LENGTH)
        output_tokenized = tokenize(
            output_str, tokenizer, return_tensors="pt",
            padding=False, add_special_tokens=False)
        tokenized_prompts.append(input_tokenized)
        all_labels.append(output_tokenized)

    print("----EXAMPLE PROMPT DECODED FROM TRAINING----")
    print(tokenizer.decode(
        tokenized_prompts[0]["input_ids"][0], skip_special_tokens=False))
    print(tokenizer.decode(
        all_labels[0]["input_ids"][0], skip_special_tokens=False))

    # Create dataset
    dataset_prompts = CustomDataset(tokenized_prompts, all_labels)

    sample = dataset_prompts[0]
    print(sample)
    print(len(sample["input_ids"]))
    print(len(sample["labels"]))

    # Prepare model with LoRA
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    tokenizer.pad_token = "!"  # Not EOS, will explain another time.\

    if "8x7b" in MODEL_NAME:
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )

    model = get_peft_model(model, config)

    naming = PREFIX + str(N_MULTIPLIER) + "_" + str(N_EXAMPLES) + "_" + str(LR)
    trainer = Trainer(
        model=model,
        train_dataset=dataset_prompts,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            logging_steps=2,
            weight_decay=0.1,
            optim="adamw_torch",
            save_strategy="epoch",
            output_dir=OUTPUT_PATH + "/" + naming
        )
    )
    model.config.use_cache = False

    trainer.train()

    # Create Inference Dataset
    all_prompts = []
    all_targets = []
    for index_dev, (_, row) in enumerate(df_dev.iterrows()):
        message = prompt_prefix()
        current_change = row["Change"]
        examples = get_examples(df_train, current_change,
                                max_examples=N_EXAMPLES,
                                random_state=42)
        test_example = give_example(row["Source"], row["Change"])

        message += examples
        message += test_example
        all_prompts.append(message)
        all_targets.append(row["Target"])

    print("----Input Text----")
    pprint.pp(all_prompts[0])
    print("----Target-----")
    pprint.pp(all_targets[0])

    tokenized_prompts = []
    for prompt in tqdm(all_prompts):
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = "!"  # Not EOS, will explain another time.\
        tokenized = tokenize(
            prompt, tokenizer, return_tensors="pt", padding=False)
        tokenized = {key: value[:, :-1].to('cuda')
                     for key, value in tokenized.items()}
        tokenized_prompts.append(tokenized)

    print("----Inference Input Text----")
    pprint.pp(tokenized_prompts[0])
    input_text = tokenizer.decode(
        tokenized_prompts[0]["input_ids"][0], skip_special_tokens=False)
    pprint.pp(input_text)

    answers = generate_examples(tokenized_prompts, prefix=naming)

    preds = []
    for sample in answers:
        pred = sample.split("[/INST] ")[-1]
        pred = pred.split(" [INST]")[0]
        pred = pred.split("\n")[0].replace(".", "").strip()
        preds.append(pred)
    # Eval
    targets = list(df_dev["Target"])
    bleu, chrf = get_bleu_chrf(preds, targets)
    acc = get_acc(preds, targets)
    print(f"BLEU: {bleu}, CHR-F: {chrf}")
    print(f"Accuracy: {acc}")

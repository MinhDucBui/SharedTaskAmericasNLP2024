# Load Libraries
# pip install transformers
# pip install peft
# pip install pandas
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
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


## Load Mixtral & Utility
LANGUAGE = "bribri"

# Replace Paths
OUTPUT_PATH = ""
DATAFOLDER = ""

# Model Params
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# Give trained adapter Path
ADAPTER_PATH = "bribri/trainexpert_5_5_0.0001/checkpoint-96"
QUANTIZED = True

CACHE_DIR = "/pfs/work7/workspace/scratch/ma_mbui-minbui/models"

# Prompt
ADD_DESC = False

# Dataset
N_EXAMPLES = 5
SEP_TOKEN = "[/INST]"
INPUT_LENGTH = 512
LABEL_LENGTH = 50


def prompt_prefix():
    message = [
        #{"role": "system", "content": "You are a helpful assistant who follows the following pattern."},
        {"role": "user",
            "content": f"This is {LANGUAGE.capitalize()}. I will give you a source sentence and a grammar change and you have to output the correct change!"},
        {"role": "assistant", "content": "Ok!"},
    ]
    return message


def give_example(source, change, target=None):
    if ADD_DESC:
        new_change = []
        changes = change.split(", ")
        for change in changes:
            category = change.split(":")[0]
            form = change.split(":")[1]
            category = CATEGORY[category]
            form = FORM[form]
            new_change += [category + ":" + form]
        new_change = ", ".join(new_change)
        change = new_change

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


def get_examples(df, current_change, max_examples=2, random_state=42, reverse=False):
    df["Current_Change"] = current_change
    # Create a new column with the count of overlapping elements
    df['overlap_count'] = df.apply(count_overlapping_elements, axis=1)
    df = df[df['overlap_count'] > 0].sort_values(
        by='overlap_count', ascending=False)
    df = df[:max_examples]
    #df = df.sample(frac=1, random_state=random_state)
    if reverse:
        df = df[::-1]

    examples = []
    for _, row_train in df[:max_examples].iterrows():
        example = give_example(
            row_train["Source"], row_train["Change"], row_train["Target"])
        examples += example
    return examples


def generate_examples(tokenized_prompts, prefix="", print_n_examples=10000000):
    answers = []

    for index, prompt in enumerate(tokenized_prompts):

        if index >= print_n_examples:
            break

        #print("\n-----Example {}----".format(index+1))
        input_text = tokenizer.decode(prompt["input_ids"][0], skip_special_tokens=True)
        
        # Replace Padding Token
        #print("Input Text: {}".format(input_text.replace("!", "")))
        
        output = model.generate(**prompt, max_new_tokens=50, pad_token_id=28808)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_prediction = generated_text.split("[/INST]")[-1]
        #print("\nGenerated Text: {}".format(generated_prediction))
        #print("\nTrue Target: {}".format(df_dev["Target"][index]))

        answers.append(generated_text)

    # Save answers list into a pickle file
    with open(os.path.join(OUTPUT_PATH, 'loadpredict' + prefix + '_answers.pickle'), 'wb') as f:
        pickle.dump(answers, f)

    filtered = []
    for index, answer in enumerate(answers):
        answer = answer.split("\n")[0].replace(".", "").strip()
        filtered.append(answer)

    # Save answers list into a pickle file
    with open(os.path.join(OUTPUT_PATH, 'loadpredict' + prefix + '_answers.pickle'), 'wb') as f:
        pickle.dump(answers, f)

    return answers


def tokenize(prompt, tokenizer, return_tensors=None, 
             cutoff_len=1500, padding=True, add_special_tokens=True):
    if padding:
        return tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len ,
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
            return os.path.join(directory, item)  # Return the first folder found
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
    parser = argparse.ArgumentParser(description='Your script description here.')
    parser.add_argument("-c", '--checkpoint', type=str, help='Folder', required=True)
    parser.add_argument("-k", '--k_shots', type=int, help='Folder', required=True)
    parser.add_argument("-p", '--prefix', type=str, help='Folder', required=True)
    parser.add_argument("-n", '--name', type=str, help='Folder', required=True)
    parser.add_argument("-q", '--quantized', type=bool, help='Folder', required=True)
    parser.add_argument("-l", '--language', type=str, help='Folder', required=True)
    parser.add_argument("-m", '--multiplier', type=str, help='Folder', required=True)
    parser.add_argument("-r", '--reverse', type=int, help='Folder', required=True)
    parser.add_argument("-t", '--test', type=int, help='Folder', required=True)
    args = parser.parse_args()
    N_EXAMPLES = args.k_shots
    PREFIX = args.prefix
    MODEL_NAME = args.name
    load_in_4bit = args.quantized
    LANGUAGE = args.language
    N_MULTIPLIER = args.multiplier
    REVERSE = args.reverse
    REVERSE = bool(REVERSE)
    TEST = args.test
    TEST = bool(TEST)

    ADAPTER_PATH = args.checkpoint
    ADAPTER_PATH = get_first_folder(ADAPTER_PATH)
    # Data Folder
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, f"{LANGUAGE}")
    FILE_PATH_TRAIN = os.path.join(DATAFOLDER, f"{LANGUAGE}-train.tsv")
    FILE_PATH_DEV = os.path.join(DATAFOLDER, f"{LANGUAGE}-dev.tsv")

    print("ADAPTER_PATH:", ADAPTER_PATH)
    print("N_EXAMPLES:", N_EXAMPLES)
    print("N_MULTIPLIER:", N_MULTIPLIER)
    print("PREFIX:", PREFIX)
    print("MODEL_NAME:", MODEL_NAME)
    print("load_in_4bit:", load_in_4bit)
    print("LANGUAGE:", LANGUAGE)
    print("REVERSE:", REVERSE)
    print("TEST:", TEST)

    # Load Tokenizer & Model from local
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                              cache_dir=CACHE_DIR, 
                                              padding_side='left',
                                              device_map="auto")
    tokenizer.pad_token = "!" #Not EOS, will explain another time.\

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                 load_in_4bit=QUANTIZED, 
                                                 cache_dir=CACHE_DIR,
                                                 device_map="auto")
    # Load Data
    model.load_adapter(ADAPTER_PATH)
    print(model)

    df_train = pd.read_csv(FILE_PATH_TRAIN, sep='\t')
    if TEST:
        FILE_PATH_DEV = os.path.join(DATAFOLDER, f"{LANGUAGE}-test.tsv")
        df_dev = pd.read_csv(FILE_PATH_DEV, sep='\t')
    else:
        df_dev = pd.read_csv(FILE_PATH_DEV, sep='\t')

    # Prepare model with LoRA
    tokenizer.pad_token = "!" #Not EOS, will explain another time.\

    # Create Inference Dataset
    all_prompts = []
    all_targets = []
    for index_dev, (_, row) in enumerate(df_dev.iterrows()):
        message = prompt_prefix()
        current_change = row["Change"]
        examples = get_examples(df_train, current_change,
                                max_examples=N_EXAMPLES, reverse=REVERSE)
        test_example = give_example(row["Source"], row["Change"])

        message += examples
        message += test_example
        all_prompts.append(message)
        all_targets.append(row["Target"])

    print("----Input Text----")
    pprint.pp(all_prompts[0])
    print("----Target-----")
    pprint.pp(all_targets[0])

    all_lengths = []
    tokenized_prompts = []
    for prompt in tqdm(all_prompts):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = "!" #Not EOS, will explain another time.\
        tokenized = tokenize(prompt, tokenizer, return_tensors="pt", padding=False)
        all_lengths.append(len(tokenized["input_ids"][0]))
        tokenized = {key: value[:, :-1].to('cuda') for key, value in tokenized.items()}
        tokenized_prompts.append(tokenized)

    if REVERSE:
        naming = str(N_EXAMPLES) + "_" + str(N_MULTIPLIER) + "_reverse"
    else:
        naming = str(N_EXAMPLES) + "_" + str(N_MULTIPLIER)

    if TEST:
        naming = "test_" + naming

    answers = generate_examples(tokenized_prompts, prefix=naming)

    preds = []
    for sample in answers:
        print(sample)
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
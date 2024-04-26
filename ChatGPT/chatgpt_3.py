import pandas as pd
import sacrebleu
import tiktoken
from tqdm import tqdm
from openai import OpenAI

LANGUAGE = "maya"
ADD_DESC = False
N_EXAMPLES = 20
MODELNAME = "gpt-3.5-turbo-0125"
# MODELNAME = "gpt-4-0125-preview"
FILENAME = f"gpt3_{LANGUAGE}20_diffprompt"

# Replace the file path with your actual file path
DATA_PATH = ""

# Initialize OpenAI client
client = OpenAI(api_key='YOUR-API-KEY',)

# Function to load TSV data into pandas DataFrame
def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')


# Function to calculate BLEU and ChrF scores
def calculate_scores(source, target):
    bleu_score = sacrebleu.corpus_bleu(source, [target]).format(score_only=True)
    chrf_score = sacrebleu.corpus_chrf(source, [target]).format(score_only=True)
    return float(bleu_score), float(chrf_score)


# Function to generate a prompt prefix
def prompt_prefix():
    return [
        {"role": "system", "content": "You are a helpful assistant who follows the following pattern."},
        {"role": "user", "content": f"This is {LANGUAGE.capitalize()}. I will give you a source sentence and a grammar change and you have to output the correct change!"},
        {"role": "assistant", "content": "Ok!"},
    ]

# Function to create a user and assistant prompt
def create_prompts(source, change, target=None):
    user_prompt = {"role": "user", "content": f"Source Sentence: {source} / Grammar Change: {change}"}
    if target:
        assistant_prompt = {"role": "assistant", "content": target}
        return [user_prompt, assistant_prompt]
    return [user_prompt]

# Function to count overlapping elements between two columns
def count_overlapping_elements(row):
    set_change = set(row["Change"].split(","))
    set_current_change = set(row["Current_Change"].split(","))
    return len(set_change.intersection(set_current_change))

# Function to get examples
def get_examples(df, current_change, max_examples=20):
    df["Current_Change"] = current_change
    df['overlap_count'] = df.apply(count_overlapping_elements, axis=1)
    df = df[df['overlap_count'] > 0].sort_values(by='overlap_count', ascending=False)
    examples = []
    for _, row_train in df[:max_examples].iterrows():
        example = create_prompts(row_train["Source"], row_train["Change"], row_train["Target"])
        examples += example
    return examples

# Function to calculate token count
def calculate_token_count(prompts, encoding):
    token_count = sum(len(encoding.encode(unit["content"])) for prompt in prompts for unit in prompt)
    return token_count

# Function to save predictions and prompts
def save_predictions_and_prompts(df, predictions, prompts, filename):
    df["Predictions"] = predictions
    df["num_examples"] = [(len(prompt) - 4) // 2 for prompt in prompts]
    df.to_csv(filename, sep='\t', index=False)

def calc_acc(preds, target):
    # Calculate accuracy
    correct_predictions = sum(
        1 for pred, true in zip(preds, target) if pred == true)
    total_predictions = len(preds)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy


# Main function
def main():
    file_path_train = f"{DATA_PATH}{LANGUAGE}-train.tsv"
    file_path_dev = f"{DATA_PATH}{LANGUAGE}-dev.tsv"

    df_train = load_data(file_path_train)
    df_dev = load_data(file_path_dev)

    bleu, chrf = calculate_scores(list(df_dev["Source"]), list(df_dev["Target"]))
    print(f"Baseline: BLEU {bleu}, ChrF {chrf}")

    all_prompts = []
    for _, row in tqdm(df_dev.iterrows()):
        message = prompt_prefix()
        current_change = row["Change"]
        examples = get_examples(df_train, current_change, max_examples=N_EXAMPLES)
        test_example = create_prompts(row["Source"], row["Change"])
        all_prompts.append(message + examples + test_example)

    encoding = tiktoken.encoding_for_model(MODELNAME)
    token_count = calculate_token_count(all_prompts, encoding)
    print("Token count:", token_count)
    print("Average tokens per prompt:", token_count / len(all_prompts))
    print("Total Cost (GPT-4):", token_count / 1_000_000 * 10)
    print("Total Cost (GPT-3.5):", token_count / 1_000_000 * 0.5)

    save_answers = []
    save_prompt = []

    for prompt in tqdm(all_prompts):
        response = client.chat.completions.create(
            model=MODELNAME,
            messages=prompt,
            temperature=0,
        )
        current_prompt_str = "\n".join(f"{unit['role']}: {unit['content']}" for unit in prompt)
        save_prompt.append(current_prompt_str)
        save_answers.append(response.choices[0].message.content)

    save_predictions_and_prompts(df_dev, save_answers, save_prompt, f"{LANGUAGE}_{FILENAME}.tsv")

    accuracy = calc_acc(save_answers, list(df_dev["Target"]))
    print("Accuracy:", accuracy)

    bleu, chrf = calculate_scores(save_answers, list(df_dev["Target"]))
    print("BLEU:", bleu)
    print("ChrF:", chrf)

if __name__ == "__main__":
    main()

from dataclasses import dataclass
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tiktoken
from datasets import load_dataset
import argparse
import yaml
from constants import META_PROMPTS_DICT, NUM_TO_LABEL_DICT

DEBUG = False

@dataclass
class DatasetNames:
    headline: str = 'Headline-(Classification)'
    fpb: str = 'FPB-(Sentiment)'
    fiqa_sa: str = 'FiQA_SA-(Sentiment)'
    convfinqa: str = 'ConvFinQA-(Financial-Q&A)'
    finqa: str = 'FinQA-(Financial-Q&A)'
    twitter_topics: str = 'Twitter-(Topics)'
    twitter_sa: str = 'Twitter-(Sentiment)'
    fin_ner_cls: str = 'FinNER-(Classification)'
    open_orca: str = 'Open-Orca'
    orca_math: str = 'Orca-Math-(Numerical-Reasoning)'

def process_dataset(dataset_name, sample_limit=100):
    """
    Load, shuffle, and convert a dataset to a pandas DataFrame.

    Args:
        dataset_name (str): The name of the dataset to load.
        sample_limit (int): The number of samples to select from the dataset.

    Returns:
        pd.DataFrame: The processed dataset as a pandas DataFrame.
    """
    dataset = load_dataset(dataset_name, split="train").shuffle(seed=42)

    if sample_limit:
        sample_limit = min(sample_limit, len(dataset))
        dataset = dataset.select(range(sample_limit))

    dataset = dataset.to_pandas()
    return dataset

def process_math(dataset_name, sample_limit=100):
    """
    Process a math dataset, filter categories, shuffle, and convert to pandas DataFrame.

    Args:
        dataset_name (str): The name of the math dataset to load.
        sample_limit (int): The number of samples to select from the dataset.

    Returns:
        pd.DataFrame: The processed math dataset as a pandas DataFrame.
    """
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda x: x['category'] in ['general', 'gain'])

    dataset = dataset.shuffle(seed=42)
    if sample_limit:
        sample_limit = min(sample_limit, len(dataset))
        dataset = dataset.select(range(sample_limit))

    math_df = dataset.to_pandas()
    math_df = math_df[['Problem', 'options', 'correct', 'Rationale']]
    math_df = math_df.apply(extract_answer, axis=1)
    math_df['instruction'] = "You are given a math problem. What is the correct answer?"
    return math_df

def extract_answer(row):
    """
    Extract the correct answer and format the input and output for math problems.

    Args:
        row (pd.Series): A row from the DataFrame containing math problem data.

    Returns:
        pd.Series: A Series with 'input' and 'output' for the math problem.
    """
    options = row['options']
    problem_with_options = f"{row['Problem']} {options}"
    correct_option = next(
        (option.strip() for option in options.split(',')
         if option.strip().startswith(row['correct'])),
        None
    )
    if correct_option is None:
        correct_option = ''
    correct_answer = f"{correct_option}\nExplanation: {row['Rationale']}"
    return pd.Series({'input': problem_with_options, 'output': correct_answer})

def count_tokens(text, encoding):
    """
    Count the number of tokens in the given text using the specified encoding.

    Args:
        text (str): The text to encode.
        encoding: The encoding to use.

    Returns:
        int: The number of tokens in the text.
    """
    return len(encoding.encode(text))

def count_tokens_dict(datasets_dict):
    """
    Count tokens for 'input' and 'output' columns in each dataset.

    Args:
        datasets_dict (dict): A dictionary of datasets.

    Returns:
        dict: A dictionary containing token counts and statistics for each dataset.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_lens_dict = {}
    for key, df in datasets_dict.items():
        tokens_lens_dict[key] = {}
        input_tokens = df['input'].apply(lambda x: count_tokens(x, encoding))
        output_tokens = df['output'].apply(lambda x: count_tokens(x, encoding))

        tokens_lens_dict[key]['input_tokens'] = int(input_tokens.sum())
        tokens_lens_dict[key]['input_mean'] = float(input_tokens.mean())
        tokens_lens_dict[key]['input_max'] = int(input_tokens.max())
        tokens_lens_dict[key]['input_min'] = int(input_tokens.min())

        tokens_lens_dict[key]['output_tokens'] = int(output_tokens.sum())
        tokens_lens_dict[key]['output_mean'] = float(output_tokens.mean())
        tokens_lens_dict[key]['output_max'] = int(output_tokens.max())
        tokens_lens_dict[key]['output_min'] = int(output_tokens.min())
    return tokens_lens_dict

def visualize(ds_dict, name):
    """
    Create and save a pie chart visualization of dataset distributions.

    Args:
        ds_dict (dict): A dictionary of dataset names and their counts.
        name (str): The name to use when saving the plot.
    """
    ds_dict = dict(sorted(ds_dict.items(), key=lambda x: x[1], reverse=True))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']

    colors = sns.color_palette('deep')
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#f0f0f0')

    wedges, texts, autotexts = ax.pie(
        ds_dict.values(),
        labels=ds_dict.keys(),
        colors=colors,
        autopct='%1.1f%%',
        pctdistance=0.85,
        wedgeprops=dict(width=0.5, edgecolor='white')
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc='#f8f9fa')
    fig.gca().add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=16)

    ax.set_title("Dataset Distribution", fontsize=20, pad=20)

    total = sum(ds_dict.values())
    total_rounded = total - (total % 100)
    ax.text(0, 0, f'{total_rounded:,}\nSamples', ha='center', va='center', fontsize=20)

    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(f'{name}.png', dpi=300)

def load_datasets():
    """
    Load and process multiple datasets.

    Returns:
        dict: A dictionary containing processed datasets.
    """
    headline_df = process_dataset("FinGPT/fingpt-headline-cls", 10000)
    fpb_df = process_dataset("odedovadia/financial_phrasebank_split", 5000)
    fin_ner_df = process_dataset("FinGPT/fingpt-ner-cls", 5000)
    convfinqa_df = process_dataset("FinGPT/fingpt-convfinqa", 2000)
    twitter_topics_df = process_dataset("zeroshot/twitter-financial-news-topic", 2500)
    twitter_sa_df = process_dataset("zeroshot/twitter-financial-news-sentiment", 5000)
    finqa_df = process_dataset("ChanceFocus/flare-finqa", 2000)

    general_instructions_num = (
        len(headline_df) + len(fpb_df) + len(fin_ner_df) + len(convfinqa_df) +
        len(finqa_df) + len(twitter_topics_df) + len(twitter_sa_df)
    )
    math_instructions_num = general_instructions_num // 2

    orca_math_df = process_dataset("microsoft/orca-math-word-problems-200k", math_instructions_num)

    # Process FPB dataset
    fpb_df = fpb_df.rename(columns={"sentence": "input", "label": "output"})
    FPB_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
    fpb_df['output'] = fpb_df['output'].apply(lambda x: FPB_LABELS.get(x, x))

    fbp_instructions = META_PROMPTS_DICT['FPB'].replace("### Instruction:\n", "").split("\n### Input")[0]
    fpb_df['instruction'] = fbp_instructions

    # Process Open-Orca dataset
    open_orca_df = process_dataset("Open-Orca/OpenOrca", general_instructions_num)
    open_orca_df = open_orca_df.rename(
        columns={"system_prompt": "instruction", "question": "input", "response": "output"}
    ).drop(columns=["id"])

    # Process Twitter topics
    twitter_topics_df = twitter_topics_df.rename(columns={"text": "input", "label": "output"})
    meta_topics = list(NUM_TO_LABEL_DICT.get("Twitter_Topics", {}).values())
    twitter_topics_instruction = META_PROMPTS_DICT['Twitter_Topics'].replace(
        "### Instruction:\n", ""
    ).split("\n### Input")[0].format(labels=meta_topics)
    twitter_topics_df['instruction'] = twitter_topics_instruction
    twitter_topics_df['output'] = twitter_topics_df['output'].apply(
        lambda x: NUM_TO_LABEL_DICT['Twitter_Topics'].get(x, x)
    )

    # Process Twitter sentiment analysis
    twitter_sa_df = twitter_sa_df.rename(columns={"text": "input", "label": "output"})
    twitter_sa_df['instruction'] = META_PROMPTS_DICT['Twitter_SA'].replace(
        "### Instruction:\n", ""
    ).split("\n### Input")[0]
    twitter_sa_df['output'] = twitter_sa_df['output'].apply(
        lambda x: NUM_TO_LABEL_DICT['Twitter_SA'].get(x, x)
    )

    # Process Orca-Math dataset
    orca_math_df = orca_math_df.rename(
        columns={"question": "input", "answer": "output"}
    )
    orca_math_df['instruction'] = (
        "You are given a mathematical word problem. Solve it step by step and provide the answer."
    )

    # Process FinQA dataset
    finqa_df = finqa_df.rename(columns={"answer": "output", "query": "input"})
    finqa_prompt = "Please answer the given financial question based on the context."
    finqa_df['input'] = finqa_df['input'].apply(lambda x: x.split("\nContext: ")[1])
    finqa_df['instruction'] = finqa_prompt
    finqa_df = finqa_df.drop(columns=["id", "text"])

    datasets_dict = {
        "Orca-Math-(Numerical-Reasoning)": orca_math_df,
        "Headline-(Classification)": headline_df,
        "FPB-(Sentiment)": fpb_df,
        "FinNER-(Classification)": fin_ner_df,
        "ConvFinQA-(Financial-Q&A)": convfinqa_df,
        "FinQA-(Financial-Q&A)": finqa_df,
        "Twitter-(Topics)": twitter_topics_df,
        "Twitter-(Sentiment)": twitter_sa_df,
        "Open-Orca": open_orca_df
    }
    return datasets_dict

def create_cocktail(datasets_mix=None, model="microsoft/Phi-3-mini-128k-instruct"):
    """
    Create a cocktail dataset by mixing specified datasets.

    Args:
        datasets_mix (list): List of dataset names to mix.
        model (str): Name of the base model or path to checkpoint.
    """
    if datasets_mix is not None:
        for dataset_name in datasets_mix:
            if not hasattr(DatasetNames, dataset_name):
                raise ValueError(f"{dataset_name} not in DatasetNames")

    datasets_mix = [getattr(DatasetNames, name) for name in datasets_mix] if datasets_mix else None

    datasets_dict = load_datasets()

    # Filter datasets_dict based on datasets_mix
    datasets_dict = {key: df for key, df in datasets_dict.items() if key in datasets_mix} if datasets_mix else datasets_dict

    lens_dict = {key: len(df) for key, df in datasets_dict.items()}
    tokens_dict = count_tokens_dict(datasets_dict)

    # Combine all datasets into a single DataFrame
    combined_datasets_df = pd.concat(datasets_dict.values(), ignore_index=True)

    # Shuffle the combined dataset
    shuffled_combined_df = combined_datasets_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if DEBUG:
        shuffled_combined_df = shuffled_combined_df.head(300)

    # Save the combined dataset to a JSONL file
    cocktail_name = f"cocktail_{'+'.join(datasets_dict.keys())}"
    path_to_save = os.path.join("cocktails", cocktail_name)
    os.makedirs(path_to_save, exist_ok=True)

    output_file = os.path.join(path_to_save, cocktail_name + ".jsonl")
    shuffled_combined_df.to_json(output_file, orient="records", lines=True)

    visualize(lens_dict, os.path.join(path_to_save, cocktail_name))

    # Save dataset statistics
    with open(os.path.join(path_to_save, cocktail_name + "_cocktail_mix.json"), "w") as f:
        cocktail_mix = {cocktail_name: lens_dict}
        json.dump(cocktail_mix, f, indent=4)

    # Save token counts
    with open(os.path.join(path_to_save, cocktail_name + "_tokens.json"), "w") as f:
        json.dump(tokens_dict, f, indent=4)

    # Save dataset info for training
    with open(os.path.join(path_to_save, "dataset_info.json"), "w") as f:
        info_dict = {
            cocktail_name: {
                "file_name": cocktail_name + ".jsonl",
                "columns": {
                    "prompt": "instruction",
                    'query': 'input',
                    'response': 'output'
                }
            }
        }
        json.dump(info_dict, f, indent=4)

    # Update and save training config
    with open("training_configs/base_config.yaml") as f:
        config_file = yaml.safe_load(f)

    config_file["model_name_or_path"] = model
    config_file['dataset_dir'] = path_to_save
    config_file["dataset"] = cocktail_name
    config_file["output_dir"] = path_to_save
    if len(datasets_dict) > 3:
        config_file['warmup_steps'] = 10
        config_file['num_train_epochs'] = 2
    else:
        config_file['num_train_epochs'] = 3

    with open(os.path.join('training_configs', 'current_config.yaml'), "w") as f:
        yaml.dump(config_file, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", help="List of datasets to mix")
    parser.add_argument("--model", type=str, help="Name of the base model or path to checkpoint", default="microsoft/Phi-3-mini-128k-instruct")
    args = parser.parse_args()
    datasets_mix = args.datasets
    model = args.model

    print(f"Creating cocktail with datasets: {datasets_mix}")

    create_cocktail(datasets_mix, model)

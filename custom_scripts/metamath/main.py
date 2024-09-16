import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import List

import click

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts

from custom_scripts.metamath.cerebras_data_processing import (
    HdF5ProcessingSFT,  # noqa: E402
)
from custom_scripts.metamath.metamath import (  # noqa: E402
    answer_augmentation_pipeline,
    fobar_pipeline,
    rephrasing_pipeline,
    self_verification_pipeline,
)
from custom_scripts.visualize import run_ui_background  # noqa: E402
import multiprocessing as mp


current_file = Path(__file__).resolve()
math_data_path = current_file.parent.parent.parent / "data" / "MATH_train.json"
gsm8k_data_path = current_file.parent.parent.parent / "data" / "gsm8k_train.json"

with open(math_data_path, "r") as f:
    math_data = json.load(f)
with open(gsm8k_data_path, "r") as f:
    gsm8k_data = json.load(f)


def validate_augmentations(ctx, param, value):
    if not value:
        raise click.BadParameter('At least one augmentation must be specified.')
    return value


@click.command('MetaMath generation')
@click.option('--dataset_name', type=click.Choice(["GSM8K", "MATH", "GSM8K-MATH"]), help='Dataset name')
@click.option('--augmentations', type=click.Choice(['ans_aug', 'rephrase', 'sv', 'fobar']),
            callback=validate_augmentations, multiple=True, help='Augmentations')
@click.option('--jsonl_dir', type=str, help='Directory to store jsonl files')
@click.option('--output_dir', type=str, help='Output directory for hdf5 files')
@click.option('--api_key', type=str, help='OpenAI API key')
@click.option('--base_url', type=str, help='OpenAI base url')
@click.option('--model_name', type=str, help='Model name')
@click.option('--tokenizer_name', type=str, help='Tokenizer name')
@click.option('--temperature', default=0.7, type=float, help='Temperature for model generation')
@click.option('--n_processes', default=2, type=int, help='Number of processes')
@click.option('--max_seq_length', default=1024, type=int, help='Maximum sequence length')
def main(
    dataset_name: str,
    augmentations: List[str],
    jsonl_dir: str,
    output_dir: str,
    api_key: str,
    base_url: str,
    model_name: str,
    tokenizer_name: str,
    temperature: float,
    n_processes: int,
    max_seq_length: int
):
    """
    MetaMath Generation Tool

    This command-line tool generates MetaMath data based on the specified dataset
    and augmentations. It processes the data and saves the results in the specified
    output directory.

    Usage:
      python main.py --dataset_name [DATASET] --augmentations [AUG1] ... --output_dir [DIR]

    Example:
      python main.py --dataset_name GSM8K-MATH --augmentations ans_aug --augmentations rephrase --output_dir ./output
    """

    exec_dict = {
        "GSM8K": {
            'ans_aug': answer_augmentation_pipeline,
            'rephrase': rephrasing_pipeline,
            'sv': self_verification_pipeline,
            'fobar': fobar_pipeline,
            'load_dataset': LoadDataFromDicts
        },
        "MATH": {
            'ans_aug': answer_augmentation_pipeline,
            'rephrase': rephrasing_pipeline,
            'sv': self_verification_pipeline,
            'fobar': fobar_pipeline,
            'load_dataset': LoadDataFromDicts
        },
    }

    output_dir = output_dir or f"hdf5_data/{dataset_name}/{'_'.join(augmentations)}"
    jsonl_dir = jsonl_dir or f"jsonl_data/{dataset_name}/{'_'.join(augmentations)}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(jsonl_dir, exist_ok=True)

    dataset_name = dataset_name.split('-')
    api_key = api_key or os.environ.get('API_KEY')

    with Pipeline(name="metamath", cache_dir='./metamath_cache') as pipeline:
        # llm = vLLM(
        #     model='meta-llama/Meta-Llama-3.1-8B-Instruct',
        #     extra_kwargs={'max_model_len': 12000}
        # )

        llm = partial(
            OpenAILLM,
            model=model_name,
            api_key=api_key,
            generation_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_seq_length,
            }
        )

        if base_url:
            llm = llm(base_url=base_url)
        else:
            llm = llm()

        for d_name in dataset_name:
            load_dataset = exec_dict[d_name]['load_dataset'](data = math_data if d_name == 'MATH' else gsm8k_data)
            for aug in augmentations:
                exec_dict[d_name][aug](llm, load_dataset, d_name)

    hdf5_processing = HdF5ProcessingSFT(
        input_path=jsonl_dir,
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        n_processes=n_processes,
        max_seq_length=max_seq_length
    )
    # distiset = pipeline.dry_run(batch_size=3)
    run_ui_background(pipeline)
    distiset = pipeline.run()
    hdf5_processing.process(distiset)


if __name__ == '__main__':
    main()

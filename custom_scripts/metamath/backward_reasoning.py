import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from distilabel.steps import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepOutput
from pydantic import Field

from .utils import MATH_DS_LIST, UNKNOWN_VAR_GSM8K, UNKNOWN_VAR_MATH, answer_cleansing


class BackwardReasoning(Task):
    dataset_name: Literal["GSM8K", "MATH"]
    method: Literal["fobar", "sv"]
    unknown_var: str = Field(default=UNKNOWN_VAR_GSM8K) # type: ignore

    def __init__(self, **data):
        super().__init__(**data)
        if "MATH" in self.dataset_name:
            self.unknown_var = UNKNOWN_VAR_MATH

    @property
    def inputs(self) -> List[str]:
        return ["backward_question", "backward_answer", "answer"]

    @property
    def outputs(self) -> List[str]:
        return [f"{self.method}_question", f"{self.method}_answer", f"{self.method}_answer_short", "model_name"]

    @property
    def prompt(self):
        file_path = Path(__file__).resolve()
        prompts_path = file_path.parent / "prompts"

        prompt_file = os.path.join(prompts_path, f'{self.method}_cot_{self.dataset_name.lower()}.txt')
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def format_input(self, input) -> ChatType: # type: ignore
        def wrap(e):
            variable, special_token = (f"{self.unknown_var}", "") if "GSM8K" in self.dataset_name else (f"${self.unknown_var}$", "### ")
            if self.method == "fobar":
                wrap_q = f"""{e['backward_question']}\n{special_token}If we know the answer to the above question is {e['answer']}, what is the value of unknown variable {variable}?"""
            elif self.method == "sv":
                wrap_q = f"""{e['backward_question']} What is the value of unknown variable {variable}?"""
            else:
                raise ValueError(f"unknown method: {self.method}")
            return f"""{self.prompt}\n\nQuestion: {wrap_q}\nA: Let's think step by step.\n"""

        return [
            {
                'role': 'system',
                'content': 'Follow the given examples and answer the question.'
            },
            {
                'role': 'user',
                'content': wrap(input)
            }
        ]

    def format_output(self, output, input) -> Dict[str, Any]:
        def get_inv_split_str():
            if self.dataset_name in MATH_DS_LIST:
                return f"The value of ${self.unknown_var}$ is"
            else:
                return f"The value of {self.unknown_var} is"

        answer = output
        answer_cleaned = answer_cleansing(pred=answer, ds_name=self.dataset_name, split_str=get_inv_split_str())

        return {
            f"{self.method}_question": input['backward_question'],
            f"{self.method}_answer": answer,
            f"{self.method}_answer_short": answer_cleaned
        }

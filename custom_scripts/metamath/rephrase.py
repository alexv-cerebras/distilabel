import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from distilabel.steps import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepOutput
from .utils import answer_cleansing


class Rephrase(Task):
    dataset_name: Literal["GSM8K", "MATH"]

    @property
    def inputs(self) -> List[str]:
        return ["question", "answer"]

    @property
    def outputs(self) -> List[str]:
        return ["rephrased_question", "answer_short", "model_name"]

    @property
    def prompt(self):
        file_path = Path(__file__).resolve()
        prompts_path = file_path.parent / "prompts"

        prompt_file = os.path.join(prompts_path, f'rephrase_cot_{self.dataset_name.lower()}.txt')
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def format_input(self, input) -> ChatType: # type: ignore
        def wrap(e):
            return f"""{self.prompt}\n\nQuestion: {e['question']}\nRephrase the above question, provide only question without any explanations."""

        return [
            {
                'role': 'user',
                'content': wrap(input)
            }
        ]

    def format_output(self, output, input) -> Dict[str, Any]:
        answer_cleaned = answer_cleansing(pred=input['answer'], ds_name=self.dataset_name)
        return {
            "rephrased_question": output,
            "answer_short": answer_cleaned
        }

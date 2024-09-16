import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from distilabel.steps import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepOutput
from .utils import answer_cleansing


class SelfVerification(Task):
    dataset_name: Literal["GSM8K", "MATH"]

    @property
    def inputs(self) -> List[str]:
        return ["backward_question", "answer"]

    @property
    def outputs(self) -> List[str]:
        return ["backward_question_rephrased", "model_name"]

    @property
    def prompt(self):
        file_path = Path(__file__).resolve()
        prompts_path = file_path.parent / "prompts"

        prompt_file = os.path.join(prompts_path, f'sv_rewrite_question_prompt_{self.dataset_name.lower()}.txt')
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def format_input(self, input) -> ChatType: # type: ignore
        def wrap(e):
            text = e['backward_question'].replace(',', '.')
            position_fullstop = text[::-1].find('.')
            answer = answer_cleansing(e['answer'], ds_name=self.dataset_name)
            question = text[len(text) - position_fullstop:].strip()
            e['base_text'] = e['backward_question'][:len(text) - position_fullstop].strip()
            return f"{self.prompt}\n\nQuestion: {question} The answer is {answer}.\n Result: "

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
        backward_question_rephrased = f"{input['base_text']} {output}"

        return {
            'backward_question_rephrased': backward_question_rephrased
        }

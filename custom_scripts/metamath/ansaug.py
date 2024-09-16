import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from distilabel.steps import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps.typing import StepOutput

from .utils import answer_cleansing


class AnsAug(Task):
    dataset_name: Literal["GSM8K", "MATH"]

    @property
    def inputs(self) -> List[str]:
        return ["question", "answer"]

    @property
    def outputs(self) -> List[str]:
        return ["question", "augmented_answer", "augmented_answer_short", "model_name"]

    @property
    def prompt(self):
        file_path = Path(__file__).resolve()
        prompts_path = file_path.parent / "prompts"

        prompt_file = os.path.join(prompts_path, f'ansaug_cot_{self.dataset_name.lower()}.txt')
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def format_input(self, input) -> ChatType: # type: ignore
        def wrap(e):
            return "{}\n\nQuestion: {}\nA: Let's think step by step.\n".format(self.prompt, e['question'])

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
        answer = output
        answer_cleaned = answer_cleansing(pred=answer, ds_name=self.dataset_name)

        return {
            'question': input['question'],
            'augmented_answer': answer,
            'augmented_answer_short': answer_cleaned
        }

    # def process(self, inputs: StepInput) -> "StepOutput":
    #     # Cerebras API doesn't support generating multiple outputs at once
    #     output = []
    #     for _ in range(self.num_repeat):
    #         llm_generation = next(super().process(inputs))
    #         output.extend(llm_generation)
    #     yield output

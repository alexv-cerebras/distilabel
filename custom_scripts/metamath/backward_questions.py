import copy
import re
from typing import List

from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput

from .utils import UNKNOWN_VAR_GSM8K, UNKNOWN_VAR_MATH, _strip_string, delete_extra_zero

_string_number_dict = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifth": 5,
    "sixteen": 16, "half": "50%"
}


class BackwardQuestions(Step):
    @property
    def inputs(self) -> List[str]:
        return ["question", "answer"]

    @property
    def outputs(self) -> List[str]:
        return ["backward_question", "backward_answer"]

    def process(self, inputs: StepInput) -> "StepOutput":
        """Replace numbers in the question with 'x' and store the original question in 'inverse_question'
        """
        outputs = []
        for input in inputs:
            input = self.parse_example(input)

            token_list = input['question'].split(' ')
            # search for numbers in the question
            numbers_idx = [idx for idx, _ in enumerate(token_list) if self.search_number(_) is not None]
            if len(numbers_idx) > 0:
                for x_idx in numbers_idx:
                    _input = copy.deepcopy(input)
                    _token_list = copy.deepcopy(token_list)
                    inverse_question_answer = _token_list[x_idx]
                    # replace the number with 'x'
                    _token_list[x_idx] = self.replace_number_x(_token_list[x_idx])

                    _input['backward_question'] = " ".join(_token_list)
                    _input['backward_answer'] = self._clean_number(inverse_question_answer)
                    outputs.append(_input)
        yield outputs

    def parse_example(self, example: StepInput) -> "StepInput":
        return example

    def _clean_number(self, text):
        pattern_start = r'^[$,.\?!]+'
        pattern_end = r'[$,.\?!]+$'

        # First remove from start, then from end
        text = re.sub(pattern_start, '', text)
        text = re.sub(pattern_end, '', text)

        return text

    @staticmethod
    def search_number(s):
        if s in _string_number_dict:
            return True
        if re.search('[\d]', s) is not None:
            if re.search('[a-zA-Z]', s) or re.search('[\\n:\(\)-*\"+–-]', s):
                return None
            else:
                return True

    def replace_number_x(self, s):
        if s in _string_number_dict:
            s = str(_string_number_dict[s])
        if s[-1] in (",", ".", "?", ";", "”", "'", "!", "\"", "%"):
            try:
                mo = re.match('.*([0-9])[^0-9]*$', s)
                return UNKNOWN_VAR_GSM8K + s[mo.end(1):]
            except:
                breakpoint()
        elif s[0] in ("$"):
            return "$" + UNKNOWN_VAR_GSM8K
        else:
            return UNKNOWN_VAR_GSM8K


class GSM8KBackwardQuestions(BackwardQuestions):
    def parse_example(self, example: StepInput) -> "StepInput":
        a = example['answer']
        if a[-2:] == ".0":
            a = a[:-2]
        example["answer"] = delete_extra_zero(a)
        return example


class MATHBackwardQuestions(BackwardQuestions):
    def parse_example(self, example: StepInput) -> "StepInput":
        ans_detail = example['answer']
        example['answer'] = self.find_math_answer(ans_detail)
        return example

    @staticmethod
    def search_number(s):
        if re.search('[\d]', s) is not None:
            if re.search('[a-zA-Z]', s) or re.search('[\\n:\(\)-*\"+–-]', s):
                return None
            else:
                return True

    def find_math_answer(self, s):
        assert ('boxed' in s)
        # s = s.replace(",", "")
        ans = s.split('boxed')[-1]
        if (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0):
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        return a

    def replace_number_x(self, s):
        if s[-1] in (",", ".", "?", ";", "”", "'", "!", "\"", "%"):
            try:
                mo = re.match('.*([0-9])[^0-9]*$', s)
                return UNKNOWN_VAR_MATH + s[mo.end(1):]
            except:
                print(f"the string is {s}")
        else:
            return UNKNOWN_VAR_MATH

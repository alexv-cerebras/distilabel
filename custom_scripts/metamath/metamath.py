from distilabel.steps import KeepColumns

from .ansaug import AnsAug
from .backward_questions import (
    GSM8KBackwardQuestions,
    MATHBackwardQuestions,
)
from .backward_reasoning import BackwardReasoning
from .rephrase import Rephrase
from .self_verification import SelfVerification

_cache = {}
# For each of the pipeline steps, output data in the following format: {'question': '...', 'answer': '...'}

def answer_augmentation_pipeline(llm, load_pipeline, dataset_name, num_repeat=2):
    if dataset_name == "GSM8K":
        keep_columns = KeepColumns(
            name="answer-aug-gsm8k-filtered",
            columns=[
                'question',
                'augmented_answer',
                # 'augmented_answer_short',
                # 'model_name'
            ],
            output_mappings={'augmented_answer': 'answer'}
        )

        answer_augmentation = AnsAug(
            name="answer-aug-gsm8k",
            num_generations=num_repeat,
            dataset_name="GSM8K",
            llm=llm
        )

    else:
        keep_columns = KeepColumns(
            name="answer-aug-math-filtered",
            columns=[
                'problem',
                'augmented_answer',
                # 'augmented_answer_short',
                # 'model_name'
            ],
            output_mappings={'problem': 'question', 'augmented_answer': 'answer'}
        )

        answer_augmentation = AnsAug(
            name="answer-aug-math",
            num_generations=num_repeat,
            dataset_name="MATH",
            llm=llm,
            input_mappings={'question': 'problem', 'answer': 'solution'}
        )

    full_pipeline = load_pipeline >> answer_augmentation >> keep_columns
    return full_pipeline

def rephrasing_pipeline(llm, load_pipeline, dataset_name, num_repeat=2):
    if dataset_name == "GSM8K":
        keep_columns = KeepColumns(
            name="rephrase_gsm8k-filtered",
            columns=[
                'rephrased_question',
                'answer_detail',
                # 'answer',
                # 'model_name'
            ],
            output_mappings={'rephrased_question': 'question', 'answer_detail': 'answer'}
        )
        rephrase = Rephrase(
            name="rephrase_gsm8k",
            num_generations=num_repeat,
            dataset_name="GSM8K",
            llm=llm
        )
    else:
        keep_columns = KeepColumns(
            name="rephrase_math-filtered",
            columns=[
                'rephrased_question',
                'solution',
                # 'answer_short',
                # 'model_name'
            ],
            output_mappings={'rephrased_question': 'question', 'solution': 'answer'}
        )
        rephrase = Rephrase(
            name="rephrase_math",
            num_generations=num_repeat,
            dataset_name="MATH",
            llm=llm,
            input_mappings={'question': 'problem', 'answer': 'solution'}
        )

    full_pipeline = load_pipeline >> rephrase >> keep_columns
    return full_pipeline

def self_verification_pipeline(llm, load_pipeline, dataset_name, num_repeat=2):
    if dataset_name == "GSM8K":
        keep_columns = KeepColumns(
            name="self-verification-gsm8k-filtered",
            columns=[
                'sv_question',
                'sv_answer',
                # 'sv_answer_short',
                # 'model_name'
            ],
            output_mappings={'sv_question': 'question', 'sv_answer': 'answer'}
        )
        preprocessing_step = _self_verification_preprocessing(llm, load_pipeline, dataset_name, num_repeat)
        backward_sv = BackwardReasoning(
            name='self-verification-gsm8k',
            num_generations=num_repeat,
            dataset_name="GSM8K",
            method="sv",
            llm=llm,
            input_mappings={'backward_question': 'backward_question_rephrased'}
        )
    else:
        keep_columns = KeepColumns(
            name="self-verification-math-filtered",
            columns=[
                'sv_question',
                'sv_answer',
                # 'sv_answer_short',
                # 'model_name'
            ],
            output_mappings={'sv_question': 'question', 'sv_answer': 'answer'}
        )
        preprocessing_step = _self_verification_preprocessing(llm, load_pipeline, dataset_name, num_repeat)
        backward_sv = BackwardReasoning(
            name='self-verification-math',
            num_generations=num_repeat,
            dataset_name="MATH",
            method="sv",
            llm=llm,
            input_mappings={'backward_question': 'backward_question_rephrased', 'answer': 'solution'}
        )

    full_pipeline = preprocessing_step >> backward_sv >> keep_columns
    return full_pipeline

def fobar_pipeline(llm, load_pipeline, dataset_name, num_repeat=2):
    if dataset_name == "GSM8K":
        keep_columns = KeepColumns(
            name="fobar-gsm8k-filtered",
            columns=[
                'fobar_question',
                'fobar_answer',
                # 'fobar_answer_short',
                # 'model_name'
            ],
            output_mappings={'fobar_question': 'question', 'fobar_answer': 'answer'}
        )
        preprocessing_step = _self_verification_preprocessing(llm, load_pipeline, dataset_name, num_repeat)
        backward_fobar = BackwardReasoning(
            name='fobar-gsm8k',
            num_generations=num_repeat,
            dataset_name="GSM8K",
            method="fobar",
            llm=llm,
            input_mappings={'backward_question': 'backward_question_rephrased'}
        )
    else:
        keep_columns = KeepColumns(
            name="fobar-math-filtered",
            columns=[
                'fobar_question',
                'fobar_answer',
                # 'fobar_answer_short',
                # 'model_name'
            ],
            output_mappings={'fobar_question': 'question', 'fobar_answer': 'answer'}
        )
        preprocessing_step = _self_verification_preprocessing(llm, load_pipeline, dataset_name, num_repeat)
        backward_fobar = BackwardReasoning(
            name='fobar-math',
            num_generations=num_repeat,
            dataset_name="MATH",
            method="fobar",
            llm=llm,
            input_mappings={'backward_question': 'backward_question_rephrased', 'answer': 'solution'}
        )

    full_pipeline = preprocessing_step >> backward_fobar >> keep_columns
    return full_pipeline

def _self_verification_preprocessing(llm, load_pipeline, dataset_name, num_repeat=2):
    if dataset_name in _cache:
        return _cache[dataset_name]

    if dataset_name == "GSM8K":
        create_backward_questions = GSM8KBackwardQuestions(name="create_backward_questions_gsm8k")
        self_verification = SelfVerification(
            name='self-verification-preprocess-gsm8k',
            num_generations=num_repeat,
            dataset_name="GSM8K",
            llm=llm
        )
    else:
        create_backward_questions = MATHBackwardQuestions(
            name="create_backward_questions_math",
            input_mappings={'question': 'problem', 'answer': 'solution'}
        )
        self_verification = SelfVerification(
            name='self-verification-preprocess-math',
            num_generations=num_repeat,
            dataset_name="MATH",
            llm=llm,
            input_mappings={'answer': 'solution'}
        )

    pipe = load_pipeline >> create_backward_questions >> self_verification
    _cache[dataset_name] = pipe
    return pipe

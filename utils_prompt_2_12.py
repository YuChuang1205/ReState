from dataclasses import dataclass
from typing import List, Optional
import random
import sys
import re
import os
import json

from utils_solution_rules import (
    generate_negative_solution_from_single,
    generate_online_pos_neg_solutions_from_single,
    generate_positive_solution_from_single,
)

try:
    from utils_solution_augment_qwenvl import (
        get_augmented_solution_mixed,
        get_augmented_solution_positive,
    )
except Exception:
    get_augmented_solution_positive = None
    get_augmented_solution_mixed = None


EMPTY_SOLUTION = "[EMPTY_SOLUTION]"


def get_empty_solution_text() -> str:
    return EMPTY_SOLUTION


def is_empty_solution_text(text) -> bool:
    return str(text).strip() == EMPTY_SOLUTION

def _get_effective_self_refine_rounds_for_split(args, split="train") -> int:
    if split == "eval":
        rounds = int(getattr(args, "self_refine_eval_rounds_val", 0))
        if rounds <= 0:
            rounds = int(getattr(args, "self_refine_eval_rounds", 1))
    elif split == "test":
        rounds = int(getattr(args, "self_refine_eval_rounds", 1))
    else:
        rounds = int(getattr(args, "self_refine_eval_rounds", 1))
    return max(1, rounds)


def should_force_empty_solution_input(args, split="train", curr_le_data=None, forced_refine_round=None) -> bool:
    if split == "train" and forced_refine_round is not None:
        return int(forced_refine_round) == 1

    if split in {"eval", "test"} and curr_le_data is None:
        return _get_effective_self_refine_rounds_for_split(args, split=split) == 1

    return False


# ==============================================================
# QCML-AE / QCMG-AE mixed single-stage training helpers
# ==============================================================
def should_mix_qcml_qcmg_ae_train(args, split="train", curr_le_data=None):
    return bool(getattr(args, "mix_qcml_qcmg_ae_train", False)) and split == "train" and curr_le_data is None


def choose_qcml_qcmg_ae_prompt_format(args):
    prob = float(getattr(args, "qcml_ae_prob", 0.5))
    prob = max(0.0, min(1.0, prob))
    return "QCML-AE" if random.random() < prob else "QCMG-AE"


def resolve_effective_prompt_format(args, split="train", curr_le_data=None, prompt_format=None):
    if prompt_format is not None:
        return prompt_format
    if should_mix_qcml_qcmg_ae_train(args, split=split, curr_le_data=curr_le_data):
        return choose_qcml_qcmg_ae_prompt_format(args)
    return getattr(args, "prompt_format", "QCM-A")


def extract_solution_from_ae_output(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    m = re.search(r"BECAUSE\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^Solution\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^The answer is \([A-Z]\)\.?", "", text, flags=re.IGNORECASE).strip()
    return text


def _get_student_history_dir(args):
    history_dir = getattr(args, "student_history_dir", None)
    if history_dir is None:
        return None
    history_dir = str(history_dir).strip()
    return history_dir or None


def _read_student_history_candidates(qid, args):
    history_dir = _get_student_history_dir(args)
    if history_dir is None:
        return []

    path = os.path.join(history_dir, f"{qid}.jsonl")
    if not os.path.exists(path):
        return []

    max_size = int(getattr(args, "student_history_max_size", 10))
    max_size = max(1, max_size)

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception:
        return []

    lines = lines[-max_size:]
    cands = []
    seen = set()
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
        except Exception:
            text = str(line).strip()
        if not text:
            continue
        if is_empty_solution_text(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        cands.append(text)
    return cands


def maybe_get_student_history_solution(qid, args, split="train", curr_le_data=None):
    if split != "train" or curr_le_data is not None:
        return None
    prob = float(getattr(args, "student_history_prob", 0.0))
    prob = max(0.0, min(1.0, prob))
    if prob <= 0.0 or random.random() >= prob:
        return None
    cands = _read_student_history_candidates(qid, args)
    if not cands:
        return None
    return random.choice(cands)


##############################################################
def get_question_text(problem):
    question = problem['question'].replace("\n", "\\n")
    return question


def get_question_text_random(problem, max_choose_num=4, curr_le_data=None):
    if curr_le_data != None:
        question = problem['question'].replace("\n", "\\n")
    else:
        random_num = random.randint(0, max_choose_num - 1)
        if random_num == 0:
            question = problem['question'].replace("\n", "\\n")
        else:
            adjusted_question_name = 'adjusted_question_' + str(random_num)
            question = problem[adjusted_question_name].replace("\n", "\\n")
    return question


def get_context_text(problem, use_caption):
    txt_context = problem['hint']
    img_context = problem['caption'] if use_caption else ""
    context = " ".join([txt_context, img_context]).strip()
    if context == "":
        context = "N/A"
    return context


def get_context_text_random(problem, use_caption, max_choose_num=4, curr_le_data=None):
    img_context = problem['caption'] if use_caption else ""
    if curr_le_data != None:
        txt_context = problem['hint']
        context = " ".join([txt_context, img_context]).strip()

    else:
        random_num = random.randint(0, max_choose_num - 1)
        if random_num == 0:
            txt_context = problem['hint']
            context = " ".join([txt_context, img_context]).strip()
        else:
            adjusted_hint_name = 'adjusted_hint_' + str(random_num)
            hint = problem[adjusted_hint_name].replace("\n", "\\n")
            context = " ".join([hint, img_context]).strip()

    if context == "":
        context = "N/A"
    else:
        pass
    return context


def get_choice_text(probelm, options):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    # print(choice_txt)
    return choice_txt


def get_origin_answer(problem, options):
    return problem['choices'][problem['answer']]


def get_answer(problem, options):
    return options[problem['answer']]


def get_lecture_text(problem):
    # \\n: GPT-3 can generate the lecture with more tokens.
    lecture = problem['lecture'].replace("\n", "\\n")
    return lecture


def get_lecture_text_random(problem, max_choose_num=4, curr_le_data=None):
    # \\n: GPT-3 can generate the lecture with more tokens.
    if curr_le_data != None:
        lecture = problem['lecture'].replace("\n", "\\n")
    else:
        random_num = random.randint(0, max_choose_num - 1)
        if random_num == 0:
            lecture = problem['lecture'].replace("\n", "\\n")
        else:
            adjusted_lecture_name = 'adjusted_lecture_' + str(random_num)
            lecture = problem[adjusted_lecture_name].replace("\n", "\\n")
    return lecture


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def get_solution_text_random(problem, max_choose_num=4, curr_le_data=None):
    # \\n: GPT-3 can generate the solution with more tokens
    if curr_le_data != None:
        solution = problem['solution'].replace("\n", "\\n")
    else:
        max_choose_num = max(1, int(max_choose_num))
        random_num = random.randint(0, max_choose_num - 1)
        if random_num == 0:
            solution = problem['solution'].replace("\n", "\\n")
        else:
            adjusted_solution_name = 'adjusted_solution_' + str(random_num)
            solution = problem.get(adjusted_solution_name, problem['solution']).replace("\n", "\\n")
    return solution


def get_solution_text_random_add_neg(problem, max_choose_num=4, curr_le_data=None, neg_prob=0.5):
    if curr_le_data != None:
        solution = problem['solution'].replace("\n", "\\n")
        Flag_pos = True
    else:
        max_choose_num = max(1, int(max_choose_num))
        neg_prob = max(0.0, min(1.0, float(neg_prob)))
        Flag_pos = True
        random_num = random.randint(0, max_choose_num - 1)
        if random_num == 0:
            solution = problem['solution'].replace("\n", "\\n")
        else:
            if random.random() < neg_prob:
                neg_solution_name = 'neg_solution_' + str(random_num)
                solution = problem.get(neg_solution_name, problem['solution']).replace("\n", "\\n")
                Flag_pos = False
            else:
                adjusted_solution_name = 'adjusted_solution_' + str(random_num)
                solution = problem.get(adjusted_solution_name, problem['solution']).replace("\n", "\\n")
    return solution, Flag_pos


def is_qwenvl_solution_mode(args) -> bool:
    return getattr(args, "solution_gen_mode", "rule") in {"qwen_vl_cache", "qwen_vl_online"}


def is_preprocessed_solution_mode(args) -> bool:
    return getattr(args, "solution_gen_mode", "rule") == "preprocessed"


def get_solution_text_random_only_neg(problem, max_choose_num=4, curr_le_data=None):
    if curr_le_data != None:
        solution = problem['solution'].replace("\n", "\\n")
    else:
        Flag_pos = True
        random_num = random.randint(0, max_choose_num - 1)
        random_pos_neg = random.random()
        if random_num == 0:
            solution = problem['solution'].replace("\n", "\\n")
        else:
            if random_pos_neg<0:
                adjusted_solution_name = 'adjusted_solution_' + str(random_num)
                solution = problem[adjusted_solution_name].replace("\n", "\\n")
            else:
                neg_solution_name = 'neg_solution_' + str(random_num)
                solution = problem[neg_solution_name].replace("\n", "\\n")
                Flag_pos = False
    return solution, Flag_pos


def get_solutions_pos_neg_random(problem, test_choose_num=10):
    choose_unique_number_1 = random.sample(range(1, 501), test_choose_num)
    choose_unique_number_2 = random.sample(range(1, 501), test_choose_num)
    pos_solutions = []
    neg_solutions = []
    for i in choose_unique_number_1:
        pos_solution_name = 'adjusted_solution_' + str(i)
        pos_solution = problem[pos_solution_name].replace("\n", "\\n")
        pos_solutions.append(pos_solution)
    for j in choose_unique_number_2:
        neg_solution_name = 'neg_solution_' + str(j)
        neg_solution = problem[neg_solution_name].replace("\n", "\\n")
        neg_solutions.append(neg_solution)
    return pos_solutions, neg_solutions



def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True,
                       WithOutput=False, curr_le_data=None):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {get_empty_solution_text()}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"

    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        elif output_format == 'AE':
            text = input + f'Answer + Solution:'
        else:
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text



def create_one_example_output_pos(format, question, context, choice, answer, lecture, solution, solution_phase2_pos_out,test_example=True,
                       WithOutput=False, curr_le_data=None):
    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nLecture: {lecture}\nSolution: {get_empty_solution_text()}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution_phase2_pos_out}"
    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution_phase2_pos_out}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution_phase2_pos_out} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution_phase2_pos_out} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution_phase2_pos_out} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution_phase2_pos_out} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution_phase2_pos_out}."

    elif output_format == 'E':
        output = f"Solution: {solution_phase2_pos_out}"

    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        elif output_format == 'AE':
            text = input + f'Answer + Solution:'
        else:
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output

    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text



def build_train_pair(problems, test_qid, args, curr_le_data=None, prompt_format=None, split="train", forced_refine_round=None):
    phase1_cot_num = args.phase1_cot_num
    phase2_cot_num = args.phase2_cot_num
    prompt_format = resolve_effective_prompt_format(args, split=split, curr_le_data=curr_le_data, prompt_format=prompt_format)
    examples = []

    if curr_le_data is None:
        question = get_question_text_random(problems[test_qid], max_choose_num=1, curr_le_data=curr_le_data)
        context = get_context_text_random(problems[test_qid], args.use_caption, max_choose_num=1, curr_le_data=curr_le_data)
        lecture = get_lecture_text_random(problems[test_qid], max_choose_num=1, curr_le_data=curr_le_data)

        problem_for_aug = dict(problems[test_qid])
        problem_for_aug["qid"] = test_qid

        use_qwenvl_aug = (
            is_qwenvl_solution_mode(args)
            and get_augmented_solution_positive is not None
        )
        use_preprocessed_aug = is_preprocessed_solution_mode(args)

        if args.use_txt_enhancement_phase1:
            solution_origin = get_solution_text(problems[test_qid])
            if use_preprocessed_aug:
                solution = get_solution_text_random(
                    problems[test_qid],
                    max_choose_num=phase1_cot_num,
                    curr_le_data=curr_le_data,
                )
            elif use_qwenvl_aug:
                solution = get_augmented_solution_positive(problem_for_aug, test_qid, args=args)
            else:
                solution = generate_positive_solution_from_single(solution_origin, args=args)
            if random.random() < getattr(args, "solution_origin_pos_prob", 0.0):
                solution = solution_origin
            solution_enhance_input = solution

        elif args.use_txt_enhancement_phase2:
            solution_origin = get_solution_text(problems[test_qid])
            if use_preprocessed_aug:
                solution = get_solution_text_random(
                    problems[test_qid],
                    max_choose_num=phase2_cot_num,
                    curr_le_data=curr_le_data,
                )
            elif use_qwenvl_aug:
                solution = get_augmented_solution_positive(problem_for_aug, test_qid, args=args)
            else:
                solution = generate_positive_solution_from_single(solution_origin, args=args)

            if random.random() < getattr(args, "solution_origin_pos_prob", 0.0):
                solution = solution_origin

            if use_preprocessed_aug:
                solution_enhance_input, flag_pos = get_solution_text_random_add_neg(
                    problems[test_qid],
                    max_choose_num=phase2_cot_num,
                    curr_le_data=curr_le_data,
                    neg_prob=getattr(args, "solution_neg_prob", 0.5),
                )
                # if flag_pos:
                #     solution_enhance_input = solution
            elif use_qwenvl_aug:
                solution_enhance_input = get_augmented_solution_mixed(
                    problem_for_aug,
                    test_qid,
                    neg_prob=getattr(args, "solution_neg_prob", 0.5),
                    args=args,
                )
            else:
                solution_enhance_input = generate_online_pos_neg_solutions_from_single(
                    solution_origin,
                    probability=getattr(args, "solution_neg_prob", 0.5),
                    args=args,
                )

            student_history_solution = maybe_get_student_history_solution(
                test_qid, args, split=split, curr_le_data=curr_le_data
            )
            if student_history_solution is not None:
                solution_enhance_input = student_history_solution

            if should_force_empty_solution_input(
                args, split=split, curr_le_data=curr_le_data, forced_refine_round=forced_refine_round
            ):
                solution_enhance_input = get_empty_solution_text()
            elif split == "train" and forced_refine_round is not None and int(forced_refine_round) == 2:
                if student_history_solution is not None:
                    solution_enhance_input = student_history_solution
                if not str(solution_enhance_input).strip():
                    solution_enhance_input = get_empty_solution_text()

        else:
            solution = get_solution_text(problems[test_qid])
            solution_enhance_input = solution

    else:
        question = get_question_text(problems[test_qid])
        context = get_context_text(problems[test_qid], args.use_caption)
        lecture = get_lecture_text(problems[test_qid])
        solution = get_solution_text(problems[test_qid])
        solution_enhance_input = curr_le_data

    choice = get_choice_text(problems[test_qid], args.options)
    answer_option = get_answer(problems[test_qid], args.options)
    answer = "(" + answer_option + ")"

    if args.use_txt_enhancement_phase2 == False:
        test_example, target = create_one_example(
            prompt_format,
            question,
            context,
            choice,
            answer,
            lecture,
            solution,
            test_example=False,
            WithOutput=True,
            curr_le_data=curr_le_data,
        )
    else:
        test_example, target = create_one_example_output_pos(
            prompt_format,
            question,
            context,
            choice,
            answer,
            lecture,
            solution_enhance_input,
            solution,
            test_example=False,
            WithOutput=True,
            curr_le_data=curr_le_data,
        )

    examples.append(test_example)
    target = target.replace("Answer:", "").strip()
    prompt_input = '\n\n'.join(examples)
    return prompt_input, target

def build_train_pair_new(problems, test_qid, args, curr_le_data=None, prompt_format=None, split="train"):
    examples = []
    per_pos_neg = args.per_pos_neg
    phase1_cot_num = args.phase1_cot_num

    prompt_format = resolve_effective_prompt_format(args, split=split, curr_le_data=curr_le_data, prompt_format=prompt_format)
    if curr_le_data == None:
        input_format, output_format = prompt_format.split("-")
        question = get_question_text_random(problems[test_qid], max_choose_num=1,
                                            curr_le_data=curr_le_data)  ##When max_choose_num=1 is origin
        context = get_context_text_random(problems[test_qid], args.use_caption, max_choose_num=1,
                                          curr_le_data=curr_le_data)  ##when max_choose_num=1 is origin
        lecture = get_lecture_text_random(problems[test_qid], max_choose_num=1,
                                          curr_le_data=curr_le_data)  ##when max_choose_num=1 is origin
        solution = get_solution_text_random(problems[test_qid], max_choose_num=phase1_cot_num,
                                            curr_le_data=curr_le_data)  ##when max_choose_num=1 is origin
        choice = get_choice_text(problems[test_qid], args.options)
        answer_option = get_answer(problems[test_qid], args.options)
        answer = "(" + answer_option + ")"

        test_example, target = create_one_example(prompt_format,
                                                  question,
                                                  context,
                                                  choice,
                                                  answer,
                                                  lecture,
                                                  solution,
                                                  test_example=False, WithOutput=True, curr_le_data=curr_le_data)
        examples.append(test_example)
        target = target.replace("Answer:", "").strip()
        # create the prompt input
        prompt_input = '\n\n'.join(examples)

        pos_targets = []
        neg_targets = []
        if output_format == 'E' and args.hardsamplemine == True:
            pos_solutions, neg_solutions = get_solutions_pos_neg_random(problems[test_qid], test_choose_num=per_pos_neg)
            for i in range(len(pos_solutions)):
                pos_targets.append(f"Solution: {pos_solutions[i]}")
            for i in range(len(neg_solutions)):
                neg_targets.append(f"Solution: {neg_solutions[i]}")
            return prompt_input, target,pos_targets, neg_targets
        else:
            return prompt_input, target

    else:
        # test example
        question = get_question_text(problems[test_qid])
        context = get_context_text(problems[test_qid], args.use_caption)
        lecture = get_lecture_text(problems[test_qid])
        solution = get_solution_text(problems[test_qid])
        choice = get_choice_text(problems[test_qid], args.options)
        answer_option = get_answer(problems[test_qid], args.options)
        answer = "(" + answer_option + ")"
        test_example, target = create_one_example(prompt_format,
                                                  question,
                                                  context,
                                                  choice,
                                                  answer,
                                                  lecture,
                                                  solution,
                                                  test_example=False, WithOutput=True, curr_le_data=curr_le_data)
        examples.append(test_example)

        target = target.replace("Answer:", "").strip()
        # create the prompt input
        prompt_input = '\n\n'.join(examples)

        return prompt_input, target





@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

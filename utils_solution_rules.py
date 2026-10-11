import random
import re


def normalize_solution_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", "\\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_solution_sentences(text: str):
    """
    Light sentence splitting.
    Supports punctuation and \n.
    """
    text = normalize_solution_text(text)
    text = text.replace("\\n", " \\n ")
    parts = re.split(r'(?<=[\.\?\!])\s+|\\n', text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def join_solution_sentences(sent_list):
    text = " ".join([s.strip() for s in sent_list if s and s.strip()])
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Positive-view augmentations
# ----------------------------
def _pos_add_prefix(text: str) -> str:
    prefixes = [
        "In summary,",
        "Briefly,",
        "To explain,",
        "More specifically,",
        "In other words,",
    ]
    if random.random() < 0.35:
        return random.choice(prefixes) + " " + text
    return text


def _pos_merge_short_sentences(sent_list):
    if len(sent_list) < 2:
        return sent_list

    out = []
    i = 0
    while i < len(sent_list):
        if (
            i < len(sent_list) - 1
            and len(sent_list[i].split()) <= 8
            and len(sent_list[i + 1].split()) <= 14
            and random.random() < 0.35
        ):
            out.append(sent_list[i] + " " + sent_list[i + 1])
            i += 2
        else:
            out.append(sent_list[i])
            i += 1
    return out


def _pos_drop_low_info_phrases(text: str) -> str:
    removable_phrases = [
        "therefore",
        "thus",
        "overall",
        "in fact",
        "for example",
        "as a result",
        "generally",
    ]
    for phrase in removable_phrases:
        if random.random() < 0.12:
            text = re.sub(rf"\b{re.escape(phrase)}\b[,]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pos_remove_last_sentence_if_long(sent_list):
    if len(sent_list) >= 4 and random.random() < 0.25:
        return sent_list[:-1]
    return sent_list


def generate_positive_solution_from_single(solution: str, args=None) -> str:
    """
    Positive = semantic-preserving augmentation.
    Keep meaning close to the original solution.
    """
    solution = normalize_solution_text(solution)
    if not solution:
        return solution

    sent_list = split_solution_sentences(solution)

    mode = random.choice([
        "light_rephrase",
        "compressed",
        "reorganized",
    ])

    if mode == "light_rephrase":
        sent_list = _pos_merge_short_sentences(sent_list)
        pos_text = join_solution_sentences(sent_list)
        pos_text = _pos_drop_low_info_phrases(pos_text)
        pos_text = _pos_add_prefix(pos_text)

    elif mode == "compressed":
        sent_list = _pos_remove_last_sentence_if_long(sent_list)
        pos_text = join_solution_sentences(sent_list)
        pos_text = _pos_add_prefix(pos_text)

    else:  # reorganized
        sent_list = _pos_merge_short_sentences(sent_list)
        pos_text = join_solution_sentences(sent_list)
        pos_text = _pos_add_prefix(pos_text)

    pos_text = normalize_solution_text(pos_text)
    if not pos_text:
        pos_text = solution

    return pos_text


# ----------------------------
# Negative-view augmentations
# ----------------------------
def _neg_shuffle_middle(sent_list):
    if len(sent_list) <= 3:
        return sent_list
    middle = sent_list[1:-1]
    random.shuffle(middle)
    return [sent_list[0]] + middle + [sent_list[-1]]


def _neg_truncate(sent_list):
    if len(sent_list) <= 1:
        return sent_list
    keep_num = random.randint(1, max(1, len(sent_list) - 1))
    return sent_list[:keep_num]


def _neg_drop_last_sentence(sent_list):
    if len(sent_list) >= 2:
        return sent_list[:-1]
    return sent_list


def _resolve_options(problem, args=None):
    if args is not None and hasattr(args, "options"):
        return list(getattr(args, "options"))
    return ["A", "B", "C", "D", "E"]


def _neg_conflict_conclusion(text: str, problem=None, args=None) -> str:
    """
    Add a conclusion inconsistent with the rationale.
    If choices are available, force a mismatched option.
    """
    if problem is not None and "choices" in problem:
        gold_idx = problem.get("answer", None)
        num_choices = len(problem["choices"])
        if gold_idx is not None and num_choices > 1:
            cand = [i for i in range(num_choices) if i != gold_idx]
            if cand:
                wrong_idx = random.choice(cand)
                options = _resolve_options(problem, args=args)
                wrong_letter = options[wrong_idx] if wrong_idx < len(options) else str(wrong_idx)
                return text + f" Therefore, the correct answer should be ({wrong_letter})."
    return text + " Therefore, the conclusion above supports a different answer."


def _neg_flip_words(text: str):
    flip_dict = {
        r"\bmore\b": "less",
        r"\bless\b": "more",
        r"\blarger\b": "smaller",
        r"\bsmaller\b": "larger",
        r"\bincrease\b": "decrease",
        r"\bdecrease\b": "increase",
        r"\bhigher\b": "lower",
        r"\blower\b": "higher",
        r"\bpositive\b": "negative",
        r"\bnegative\b": "positive",
        r"\bcorrect\b": "incorrect",
        r"\bincorrect\b": "correct",
        r"\bis\b": "is not",
        r"\bare\b": "are not",
        r"\bcan\b": "cannot",
        r"\bcauses\b": "does not cause",
        r"\bsupports\b": "does not support",
    }

    items = list(flip_dict.items())
    random.shuffle(items)

    changed = False
    for pattern, repl in items[:4]:
        if random.random() < 0.45:
            new_text, n = re.subn(pattern, repl, text, count=1, flags=re.IGNORECASE)
            if n > 0:
                text = new_text
                changed = True
    return text, changed


def _neg_add_confusing_prefix(text: str) -> str:
    prefixes = [
        "At first glance,",
        "It may seem that",
        "One possible explanation is that",
        "A confusing point is that",
    ]
    if random.random() < 0.45:
        return random.choice(prefixes) + " " + text
    return text


def generate_negative_solution_from_single(solution: str, problem=None, args=None) -> str:
    """
    Negative = structure-broken / conclusion-conflicting / semantic-flipped view.
    """
    solution = normalize_solution_text(solution)
    if not solution:
        return solution

    sent_list = split_solution_sentences(solution)

    neg_mode = random.choice([
        "truncate",
        "shuffle",
        #"drop_conclusion",
        "flip",
        "conflict_conclusion",
        "mixed",
    ])

    if neg_mode == "truncate":
        sent_list = _neg_truncate(sent_list)
        neg_text = join_solution_sentences(sent_list)

    elif neg_mode == "shuffle":
        sent_list = _neg_shuffle_middle(sent_list)
        neg_text = join_solution_sentences(sent_list)

    elif neg_mode == "drop_conclusion":
        sent_list = _neg_drop_last_sentence(sent_list)
        neg_text = join_solution_sentences(sent_list)

    elif neg_mode == "flip":
        neg_text, changed = _neg_flip_words(solution)
        if not changed:
            sent_list = _neg_truncate(sent_list)
            neg_text = join_solution_sentences(sent_list)

    elif neg_mode == "conflict_conclusion":
        neg_text = _neg_conflict_conclusion(solution, problem=problem, args=args)

    else:  # mixed
        if len(sent_list) > 2:
            sent_list = _neg_shuffle_middle(sent_list)
        sent_list = _neg_truncate(sent_list)
        neg_text = join_solution_sentences(sent_list)
        neg_text, _ = _neg_flip_words(neg_text)
        if random.random() < 0.5:
            neg_text = _neg_conflict_conclusion(neg_text, problem=problem, args=args)

    neg_text = _neg_add_confusing_prefix(neg_text)
    neg_text = normalize_solution_text(neg_text)

    if not neg_text or neg_text == solution:
        neg_text = "Possibly, " + solution

    return neg_text


def generate_online_pos_neg_solutions_from_single(solution: str, probability: float = 0.5, problem=None, args=None) -> str:
    if random.random() < probability:
        return generate_negative_solution_from_single(solution, problem=problem, args=args)
    return generate_positive_solution_from_single(solution, args=args)

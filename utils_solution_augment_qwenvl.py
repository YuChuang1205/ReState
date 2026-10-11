import json
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from model import load_qwen_vl_base
from utils_solution_rules import (
    generate_negative_solution_from_single,
    generate_positive_solution_from_single,
)


# ----------------------------
# small utils
# ----------------------------
def normalize_solution_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", "\\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_solution_prefix(text: str) -> str:
    text = normalize_solution_text(text)
    text = re.sub(r"^(Solution\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(Answer\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(BECAUSE\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^The input solution is (correct|incorrect)!\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def parse_tagged_solution_outputs(raw_text: str, tag: str, repeat_num: int = 1) -> List[str]:
    raw_text = str(raw_text or "")
    raw_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
    raw_text = raw_text.strip()
    raw_text = re.sub(r"(?:\s*~~~\s*)+$", "", raw_text)

    pieces = [p.strip() for p in re.split(r"\s*~~~\s*", raw_text) if p and p.strip()]
    cleaned: List[str] = []
    tag_pattern = re.compile(rf"^\**\s*{re.escape(tag)}\s*:\**\s*", flags=re.IGNORECASE)

    for piece in pieces:
        piece = tag_pattern.sub("", piece).strip()
        piece = strip_solution_prefix(piece)
        if piece:
            cleaned.append(piece)

    if not cleaned and raw_text:
        fallback = tag_pattern.sub("", raw_text).strip()
        fallback = strip_solution_prefix(fallback)
        if fallback:
            cleaned.append(fallback)

    if repeat_num is not None and repeat_num > 0:
        cleaned = cleaned[:repeat_num]
    return cleaned


def _try_join(*parts) -> Optional[str]:
    p = os.path.join(*[str(x) for x in parts if x is not None and str(x) != ""])
    return p


def _get_dataset_name(args) -> str:
    return str(getattr(args, "dataset_name", "m3cot") or "m3cot").lower().strip()


def resolve_dataset_image_path(problem: dict, data_root: str, dataset_name: str = "m3cot") -> Optional[str]:
    dataset_name = str(dataset_name or "m3cot").lower().strip()

    img = problem.get("image_path", None)
    if not img:
        img = problem.get("image", None)

    if img is None:
        return None
    img = str(img).strip()
    if img == "" or img.lower() in {"null", "none"}:
        return None

    if os.path.isabs(img) and os.path.exists(img):
        return os.path.abspath(img)

    qid = problem.get("qid", None)
    if qid is None:
        qid = problem.get("id", None) or problem.get("question_id", None)
    qid = str(qid) if qid is not None else None

    base = os.path.basename(img)
    cands: List[str] = []
    if qid is not None:
        cands.append(_try_join(data_root, dataset_name, "images", qid, base))
    cands.append(_try_join(data_root, img))
    cands.append(_try_join(data_root, dataset_name, img))
    cands.append(_try_join(data_root, dataset_name, "images", img))
    if qid is not None:
        cands.append(_try_join(data_root, dataset_name, "images", qid, img))
    cands.append(_try_join(data_root, dataset_name, "images", base))

    for p in cands:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def resolve_m3cot_image_path(problem: dict, data_root: str) -> Optional[str]:
    # Backward-compatible alias. New code should call resolve_dataset_image_path(...).
    return resolve_dataset_image_path(problem, data_root, dataset_name="m3cot")


def _safe_atomic_write_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@contextmanager
def _file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    f = open(lock_path, "a+")
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f.close()


# ----------------------------
# singleton loader
# ----------------------------
_AUGMENTOR = None


def get_solution_augmentor(args):
    global _AUGMENTOR
    if _AUGMENTOR is None:
        _AUGMENTOR = QwenVLSolutionAugmentor(args)
    return _AUGMENTOR


class QwenVLSolutionAugmentor:
    def __init__(self, args):
        self.args = args
        self.model_name = getattr(args, "solution_gen_model", "Qwen/Qwen3-VL-8B-Instruct")
        self.cache_dir = getattr(args, "solution_cache_dir", "./solution_aug_cache")
        self.use_image = bool(getattr(args, "solution_gen_use_image", False))
        self.use_caption = bool(getattr(args, "solution_gen_use_caption", False))
        self.temperature = float(getattr(args, "solution_gen_temperature", 0.9))
        self.top_p = float(getattr(args, "solution_gen_top_p", 0.9))
        self.max_new_tokens = int(getattr(args, "solution_gen_max_new_tokens", 256))
        self.do_sample = bool(getattr(args, "solution_gen_do_sample", False))
        self.read_only = bool(getattr(args, "solution_cache_read_only", False))
        self.strict_filter = bool(getattr(args, "solution_gen_strict_filter", False))
        self.edit_from_origin = bool(getattr(args, "solution_edit_from_origin", False))
        self.multi_gen_num = max(1, int(getattr(args, "solution_multi_gen_num", 1)))

        self._processor = None
        self._model = None
        self._device = None
        #self._dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self._dtype = torch.float16

    # ---------- cache ----------
    def _cache_path(self, qid: str) -> str:
        return os.path.join(self.cache_dir, f"{qid}.json")

    def _lock_path(self, qid: str) -> str:
        return os.path.join(self.cache_dir, f"{qid}.lock")

    def _read_cache(self, qid: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(qid)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "pos" not in data:
                data["pos"] = []
            if "neg" not in data:
                data["neg"] = []
            return data
        except Exception:
            return None

    def _write_cache(self, qid: str, bank: Dict[str, Any]):
        _safe_atomic_write_json(self._cache_path(qid), bank)


    def _ensure_model(self):
        if self._model is not None and self._processor is not None:
            return

        if torch.cuda.is_available():
            try:
                use_bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                use_bf16 = False
            self._dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            self._dtype = torch.float16

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        device_map = None
        self._model = load_qwen_vl_base(
            self.model_name,
            torch_dtype=self._dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        self._model.eval()

        device_str = getattr(self.args, "solution_gen_device", None)
        if device_str:
            self._device = torch.device(device_str)
            self._model.to(self._device)
        else:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                self._model.to(self._device)
            else:
                self._device = torch.device("cpu")
                self._model.to(self._device)


    # ---------- prompt packing ----------
    def _gold_answer_letter(self, problem: Dict[str, Any]) -> str:
        options = getattr(self.args, "options", ["A", "B", "C", "D", "E"])
        idx = int(problem["answer"])
        if idx < len(options):
            return options[idx]
        return str(idx)

    def _gold_answer_text(self, problem: Dict[str, Any]) -> str:
        idx = int(problem["answer"])
        try:
            return str(problem["choices"][idx])
        except Exception:
            return ""

    def _origin_solution_is_empty(self, problem: Dict[str, Any]) -> bool:
        origin = normalize_solution_text(problem.get("solution", ""))
        return origin == ""

    def _option_info(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        choices = problem.get("choices", [])
        options = getattr(self.args, "options", ["A", "B", "C", "D", "E"])
        gold_idx = int(problem["answer"])

        gold_letter = options[gold_idx] if gold_idx < len(options) else str(gold_idx)
        gold_text = str(choices[gold_idx]) if gold_idx < len(choices) else ""

        wrong_items = []
        for i, c in enumerate(choices):
            if i == gold_idx:
                continue
            letter = options[i] if i < len(options) else str(i)
            wrong_items.append({
                "idx": i,
                "letter": letter,
                "text": str(c),
            })

        wrong_text = " ".join([f"({x['letter']}) {x['text']}" for x in wrong_items]) if wrong_items else "N/A"
        near_wrong = random.choice(wrong_items) if wrong_items else None

        return {
            "gold_idx": gold_idx,
            "gold_letter": gold_letter,
            "gold_text": gold_text,
            "wrong_items": wrong_items,
            "wrong_text": wrong_text,
            "near_wrong": near_wrong,
        }

    def _get_negative_variant_pool(self) -> List[str]:
        return [
            "single_step_flip",
            "final_hop_wrong",
            "evidence_swap",
            "near_miss_option",
            "single_step_flip",
            "final_hop_wrong",
            "evidence_swap",
            "structure_reframe_wrong",
            "alternative_reasoning_wrong",
            "option_retarget_wrong",
        ]

    def _get_image(self, problem: Dict[str, Any]) -> Optional[Image.Image]:
        if not self.use_image:
            return None
        image_path = resolve_dataset_image_path(problem, self.args.data_root, dataset_name=_get_dataset_name(self.args))
        if image_path is None:
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None

    def _build_common_problem_text(self, problem: Dict[str, Any]) -> str:
        q = str(problem.get("question", "")).replace("\n", "\\n").strip()
        hint = str(problem.get("hint", "") or "").replace("\n", "\\n").strip()
        caption = str(problem.get("caption", "") or "").replace("\n", "\\n").strip()
        solution = normalize_solution_text(problem.get("solution", "")) or "N/A"

        choices = problem.get("choices", [])
        options = getattr(self.args, "options", ["A", "B", "C", "D", "E"])
        choice_text = " ".join([f"({options[i]}) {c}" for i, c in enumerate(choices)])
        info = self._option_info(problem)

        pieces = [
            "[Question]",
            q,
            "[Context]",
            hint if hint else "N/A",
        ]
        if self.use_caption and caption:
            pieces += ["[Caption]", caption]

        pieces += [
            "[Options]",
            choice_text,
            "[Gold answer]",
            f"({info['gold_letter']}) {info['gold_text']}",
            "[Original solution to edit]",
            solution,
        ]
        return "\n".join(pieces)

    def _multi_output_tag(self, task: str) -> str:
        return "Adjusted Solution" if task == "positive" else "Negative Solution"

    def _build_edit_from_origin_messages(self, problem: Dict[str, Any], task: str, repeat_num: int) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
        image = self._get_image(problem)
        q = str(problem.get("question", "")).replace("\n", "\\n").strip()
        hint = str(problem.get("hint", "") or "").replace("\n", "\\n").strip() or "N/A"
        caption = str(problem.get("caption", "") or "").replace("\n", "\\n").strip()
        solution = normalize_solution_text(problem.get("solution", "")) or "N/A"

        choices = problem.get("choices", [])
        options = getattr(self.args, "options", ["A", "B", "C", "D", "E"])
        choice_text = " ".join([f"({options[i]}) {c}" for i, c in enumerate(choices)])
        info = self._option_info(problem)
        answer_text = f"({info['gold_letter']}) {info['gold_text']}"

        base_blocks = [
            f"Question: {q}",
            f"Context: {hint}",
        ]
        if self.use_caption and caption:
            base_blocks.append(f"Caption: {caption}")
        base_blocks += [
            f"Options: {choice_text}",
            f"Correct Answer: {answer_text}",
            f"Solution: {solution}",
        ]
        sample_text = "\n".join(base_blocks)

        if task == "positive":
            instruct = (
                "You are an intelligent agent with both perception and reasoning abilities. "
                f'Based on the given context, image, and options, please make random content adjustments to "{solution}" '
                "within a range of 10% to 50% while ensuring that the semantics remain unchanged and the gold answer remains logically supported. "
                f"Please output {repeat_num} different solutions. Each output format is 'Adjusted Solution:'. "
                "Use '\n\n~~~\n\n' to separate them. The output must be in the required format and contain no extra commentary."
            )
        else:
            instruct = (
                "You are an intelligent agent with both perception and reasoning abilities. "
                f'"{solution}" is the explanation for the above problem. Based on the given context, image, and options, '
                f'please make minor edits to the "{solution}" to reverse its meaning and ensure the correct answer cannot be logically derived, '
                "while keeping most of the original words and structure intact. "
                f"Please output {repeat_num} different solutions. Each output format is 'Negative Solution:'. "
                "Use '\n\n~~~\n\n' to separate them. The output must be in the required format and contain no extra commentary."
            )

        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": sample_text + "\n\n" + instruct})
        messages = [{"role": "user", "content": content}]
        return messages, image

    def _build_messages(self, problem: Dict[str, Any], task: str, variant: str) -> Tuple[List[Dict[str, Any]], Optional[Image.Image]]:
        image = self._get_image(problem)
        common_text = self._build_common_problem_text(problem)
        info = self._option_info(problem)

        near_wrong = info["near_wrong"]
        near_wrong_text = f"({near_wrong['letter']}) {near_wrong['text']}" if near_wrong is not None else "a wrong option"
        wrong_options_text = info["wrong_text"]

        if task == "positive":
            system_prompt = (
                "You are a careful multimodal reasoning editor for data augmentation. "
                "Your job is to transform the original correct solution into a substantially different but still correct solution. "
                "The rewritten solution may differ a lot in wording, sentence structure, reasoning order, and presentation style, "
                "but it must remain fully grounded in the question, context, image, and options, and it must still support the gold answer. "
                "Do not write from scratch blindly; use the original solution as a semantic anchor. "
                "Return only the rewritten solution text."
            )

            extra = {
                "logic_reframe": (
                    "Reorganize the reasoning flow substantially. "
                    "You may change the order of steps, merge or split steps, and reframe the logic, "
                    "but the final reasoning must still support the gold answer."
                ),
                "evidence_first": (
                    "Rewrite the solution in an evidence-first style: first summarize the key visual/contextual clues, "
                    "then derive the answer from them."
                ),
                "option_elimination": (
                    "Rewrite the solution as an elimination-style rationale: explain why competing options are less suitable "
                    "and why the gold answer is best."
                ),
                "abductive_style": (
                    "Rewrite the solution in a more inferential style: start from the observed facts and infer the most plausible correct answer."
                ),
                "concise_but_complete": (
                    "Rewrite the solution into a shorter but still complete rationale. "
                    "It may be much more compact than the original, but it must preserve all decisive logic."
                ),
                "step_explicit": (
                    "Rewrite the solution by making implicit reasoning steps explicit, so the logic becomes more structured and easier to follow."
                ),
            }.get(
                variant,
                "Rewrite the original correct solution into a clearly different but still correct rationale."
            )

            user_text = (
                f"{common_text}\n\n"
                "[Editing goal]\n"
                f"{extra}\n\n"
                "[Hard constraints]\n"
                "1. The final reasoning must still support the gold answer.\n"
                "2. The output should be noticeably different from the original solution in expression, structure, or reasoning organization.\n"
                "3. Larger edits are allowed, including reordering steps, changing presentation style, compressing, expanding, or reframing.\n"
                "4. However, do not invent unsupported facts.\n"
                "5. Keep the key evidence consistent with the question, context, image, and options.\n"
                "6. Do not add meta-talk such as 'rewritten', 'edited', 'as an AI', or explanations about the task.\n"
                "7. Output only the rewritten solution text.\n"
                "8. If the original solution is N/A, generate a correct rationale directly from the problem information."
            )
        else:
            system_prompt = (
                "You are a careful multimodal reasoning editor for hard-negative data creation. "
                "Most negative rationales should stay close to the original correct solution and only introduce one or two key reasoning mistakes, "
                "so they function as hard negatives. "
                "However, some negative rationales may also use a moderately different wording or local structure to improve diversity, "
                "as long as they remain relevant to the question, context, image, and options, and the final reasoning is incorrect. "
                "Do not mention that the rationale is intentionally wrong. "
                "Return only the edited solution text."
            )

            extra = {
                "single_step_flip": (
                    "Keep the rationale very close to the original solution, but flip one key inference step so the final reasoning becomes incorrect."
                ),
                "final_hop_wrong": (
                    f"Keep almost all earlier reasoning intact, but change the final decision step so that it leans toward {near_wrong_text} instead of the gold answer."
                ),
                "evidence_swap": (
                    "Keep the structure close to the original, but subtly misread or swap one important visual/contextual clue."
                ),
                "near_miss_option": (
                    f"Produce a near-miss hard negative: the rationale should stay very close to the original, "
                    f"but a small local mistake should make it favor a plausible wrong option such as {near_wrong_text}."
                ),
                "structure_reframe_wrong": (
                    "Rewrite part of the reasoning flow with moderate variation in wording or local structure, "
                    "but keep it grounded in the same question and evidence, and make one crucial conclusion step wrong."
                ),
                "alternative_reasoning_wrong": (
                    "Use a different but still relevant reasoning path from the original solution, "
                    "while ensuring the final answer supported by the rationale is incorrect."
                ),
                "option_retarget_wrong": (
                    f"Rewrite the rationale so that it plausibly supports a wrong option such as {near_wrong_text}, "
                    "with moderate wording variation while staying relevant to the question, context, image, and options."
                ),
            }.get(
                variant,
                "Generate a relevant but incorrect rationale."
            )

            user_text = (
                f"{common_text}\n\n"
                "[Wrong options for reference]\n"
                f"{wrong_options_text}\n\n"
                "[Editing goal]\n"
                f"{extra}\n\n"
                "[Hard constraints]\n"
                "1. The result must become inconsistent with the gold answer.\n"
                "2. Most variants should stay close to the original solution, but moderate local diversity is allowed when requested.\n"
                "3. Keep the rationale relevant, coherent, and plausible.\n"
                "4. Prefer subtle reasoning errors over obvious nonsense.\n"
                "5. Do not say it is intentionally wrong.\n"
                "6. Do not add meta-talk such as 'rewritten', 'edited', or 'as an AI'.\n"
                "7. Output only the edited solution text.\n"
                "8. If the original solution is N/A, generate a plausible but incorrect rationale directly from the problem information."
            )

        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        return messages, image

    def _prepare_request(self, problem: Dict[str, Any], task: str, variant: str, repeat_num: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_model()
        effective_repeat_num = max(1, int(repeat_num if repeat_num is not None else self.multi_gen_num))
        if self.edit_from_origin:
            messages, image = self._build_edit_from_origin_messages(problem, task=task, repeat_num=effective_repeat_num)
            parse_mode = "multi_tagged"
        else:
            messages, image = self._build_messages(problem, task=task, variant=variant)
            parse_mode = "single"
        chat_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "chat_text": chat_text,
            "image": image,
            "task": task,
            "repeat_num": effective_repeat_num,
            "parse_mode": parse_mode,
        }

    def _generate_batch_raw(self, request_list: List[Dict[str, Any]]) -> List[str]:
        self._ensure_model()
        if not request_list:
            return []

        outputs: List[Optional[str]] = [None] * len(request_list)

        groups = {True: [], False: []}
        for idx, req in enumerate(request_list):
            has_image = req.get("image", None) is not None
            groups[has_image].append((idx, req))

        for has_image, items in groups.items():
            if not items:
                continue

            texts = [x[1]["chat_text"] for x in items]
            if has_image:
                images = [x[1]["image"] for x in items]
                inputs = self._processor(
                    text=texts,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=texts,
                    padding=True,
                    return_tensors="pt",
                )

            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature if self.do_sample else None,
                    top_p=self.top_p if self.do_sample else None,
                    pad_token_id=getattr(self._processor.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self._processor.tokenizer, "eos_token_id", None),
                )

            gen_ids = out[:, input_len:]
            texts_out = self._processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for (orig_idx, _), text in zip(items, texts_out):
                outputs[orig_idx] = str(text or "")

        return [x if x is not None else "" for x in outputs]

    def _generate_batch(self, request_list: List[Dict[str, Any]]) -> List[str]:
        raw_outputs = self._generate_batch_raw(request_list)
        final_outputs: List[str] = []
        for req, raw_text in zip(request_list, raw_outputs):
            if req.get("parse_mode") == "multi_tagged":
                tag = self._multi_output_tag(req.get("task", "positive"))
                parsed = parse_tagged_solution_outputs(raw_text, tag=tag, repeat_num=int(req.get("repeat_num", 1)))
                final_outputs.append(parsed[0] if parsed else "")
            else:
                final_outputs.append(strip_solution_prefix(raw_text))
        return final_outputs

    def _generate_once(self, problem: Dict[str, Any], task: str, variant: str) -> str:
        req = self._prepare_request(problem, task=task, variant=variant, repeat_num=1)
        return self._generate_batch([req])[0]

    def _generate_multi_candidates(self, problem: Dict[str, Any], task: str, repeat_num: Optional[int] = None) -> List[str]:
        effective_repeat_num = max(1, int(repeat_num if repeat_num is not None else self.multi_gen_num))
        req = self._prepare_request(problem, task=task, variant="rewrite_from_origin", repeat_num=effective_repeat_num)
        raw_text = self._generate_batch_raw([req])[0]
        if req.get("parse_mode") == "multi_tagged":
            return parse_tagged_solution_outputs(raw_text, tag=self._multi_output_tag(task), repeat_num=effective_repeat_num)
        one = strip_solution_prefix(raw_text)
        return [one] if one else []

    # ---------- filtering ----------
    def _positive_ok(self, text: str, origin: str) -> bool:
        text = strip_solution_prefix(text)
        origin = strip_solution_prefix(origin)
        if not text:
            return False
        if len(text.split()) < 6:
            return False
        if text.lower() == origin.lower():
            return False
        banned = ["as an ai", "i cannot", "i'm sorry", "rewrite", "rewritten"]
        if any(b in text.lower() for b in banned):
            return False
        return True

    def _negative_ok(self, text: str, origin: str) -> bool:
        text = strip_solution_prefix(text)
        origin = strip_solution_prefix(origin)
        if not text:
            return False
        if len(text.split()) < 6:
            return False
        if text.lower() == origin.lower():
            return False
        banned = ["as an ai", "i cannot", "intentionally wrong", "incorrect rationale"]
        if any(b in text.lower() for b in banned):
            return False
        return True

    # ---------- fallback ----------
    def _rule_positive_fallback(self, solution: str) -> str:
        return generate_positive_solution_from_single(solution, args=self.args)

    def _rule_negative_fallback(self, solution: str, problem: Dict[str, Any]) -> str:
        return generate_negative_solution_from_single(solution, problem=problem, args=self.args)

    # ---------- bank build ----------
    def _build_bank(self, problem: Dict[str, Any], qid: str) -> Dict[str, Any]:
        origin = normalize_solution_text(problem.get("solution", ""))
        desired_pos = int(getattr(self.args, "solution_pos_num", 6))
        desired_neg = int(getattr(self.args, "solution_neg_num", 6))

        bank = {
            "qid": str(qid),
            "solution_origin": origin,
            "pos": [],
            "neg": [],
            "meta": {
                "model": self.model_name,
                "use_image": self.use_image,
                "use_caption": self.use_caption,
                "edit_from_origin": self.edit_from_origin,
                "multi_gen_num": self.multi_gen_num,
            },
        }

        if origin == "":
            bank["pos"] = [origin for _ in range(max(desired_pos, 1))]
            bank["neg"] = [origin for _ in range(max(desired_neg, 1))]
            bank["meta"]["empty_origin_solution"] = True
            return bank

        pos_variants = [
            "logic_reframe",
            "evidence_first",
            "option_elimination",
            "abductive_style",
            "concise_but_complete",
            "step_explicit",
        ]
        neg_variants = self._get_negative_variant_pool()

        pos_trials = max(desired_pos * 3, 12)
        neg_trials = max(desired_neg * 3, 12)

        if self.edit_from_origin:
            for _ in range(pos_trials):
                if len(bank["pos"]) >= desired_pos:
                    break
                try:
                    texts = self._generate_multi_candidates(problem, task="positive", repeat_num=self.multi_gen_num)
                except Exception:
                    texts = [self._rule_positive_fallback(origin)]
                for text in texts:
                    text = strip_solution_prefix(text)
                    if self._positive_ok(text, origin) and text not in bank["pos"]:
                        bank["pos"].append(text)
                        if len(bank["pos"]) >= desired_pos:
                            break

            for _ in range(neg_trials):
                if len(bank["neg"]) >= desired_neg:
                    break
                try:
                    texts = self._generate_multi_candidates(problem, task="negative", repeat_num=self.multi_gen_num)
                except Exception:
                    texts = [self._rule_negative_fallback(origin, problem)]
                for text in texts:
                    text = strip_solution_prefix(text)
                    if self._negative_ok(text, origin) and text not in bank["neg"]:
                        bank["neg"].append(text)
                        if len(bank["neg"]) >= desired_neg:
                            break
        else:
            for _ in range(pos_trials):
                if len(bank["pos"]) >= desired_pos:
                    break
                variant = random.choice(pos_variants)
                try:
                    text = self._generate_once(problem, task="positive", variant=variant)
                except Exception:
                    text = self._rule_positive_fallback(origin)
                text = strip_solution_prefix(text)
                if self._positive_ok(text, origin) and text not in bank["pos"]:
                    bank["pos"].append(text)

            for _ in range(neg_trials):
                if len(bank["neg"]) >= desired_neg:
                    break
                variant = random.choice(neg_variants)
                try:
                    text = self._generate_once(problem, task="negative", variant=variant)
                except Exception:
                    text = self._rule_negative_fallback(origin, problem)
                text = strip_solution_prefix(text)
                if self._negative_ok(text, origin) and text not in bank["neg"]:
                    bank["neg"].append(text)

        # robust fallback (avoid infinite loops when fallback keeps returning duplicates)
        pos_fill_trials = 0
        max_pos_fill_trials = max(desired_pos * 5, 20)
        while len(bank["pos"]) < desired_pos and pos_fill_trials < max_pos_fill_trials:
            text = strip_solution_prefix(self._rule_positive_fallback(origin))
            if self._positive_ok(text, origin) and text not in bank["pos"]:
                bank["pos"].append(text)
            pos_fill_trials += 1

        neg_fill_trials = 0
        max_neg_fill_trials = max(desired_neg * 5, 20)
        while len(bank["neg"]) < desired_neg and neg_fill_trials < max_neg_fill_trials:
            text = strip_solution_prefix(self._rule_negative_fallback(origin, problem))
            if self._negative_ok(text, origin) and text not in bank["neg"]:
                bank["neg"].append(text)
            neg_fill_trials += 1

        # final size guarantee: allow duplicates rather than risking dead loop
        if len(bank["pos"]) < desired_pos:
            pos_fill = strip_solution_prefix(self._rule_positive_fallback(origin))
            if not self._positive_ok(pos_fill, origin):
                pos_fill = origin
            while len(bank["pos"]) < desired_pos:
                bank["pos"].append(pos_fill)

        if len(bank["neg"]) < desired_neg:
            neg_fill = strip_solution_prefix(self._rule_negative_fallback(origin, problem))
            if not self._negative_ok(neg_fill, origin):
                neg_fill = "This reasoning is relevant to the question but leads to the wrong answer."
            while len(bank["neg"]) < desired_neg:
                bank["neg"].append(neg_fill)

        return bank

    def load_or_build_bank(self, problem: Dict[str, Any], qid: str) -> Dict[str, Any]:
        qid = str(qid)
        bank = self._read_cache(qid)
        desired_pos = int(getattr(self.args, "solution_pos_num", 6))
        desired_neg = int(getattr(self.args, "solution_neg_num", 6))
        enough = bank is not None and len(bank.get("pos", [])) >= desired_pos and len(bank.get("neg", [])) >= desired_neg
        if enough:
            return bank

        if self.read_only:
            origin = normalize_solution_text(problem.get("solution", ""))
            if bank is None:
                if origin == "":
                    bank = {
                        "qid": qid,
                        "solution_origin": origin,
                        "pos": [origin for _ in range(max(desired_pos, 1))],
                        "neg": [origin for _ in range(max(desired_neg, 1))],
                        "meta": {"fallback": True, "empty_origin_solution": True},
                    }
                else:
                    bank = {
                        "qid": qid,
                        "solution_origin": origin,
                        "pos": [self._rule_positive_fallback(origin)],
                        "neg": [self._rule_negative_fallback(origin, problem)],
                        "meta": {"fallback": True},
                    }
            return bank

        with _file_lock(self._lock_path(qid)):
            bank = self._read_cache(qid)
            enough = bank is not None and len(bank.get("pos", [])) >= desired_pos and len(bank.get("neg", [])) >= desired_neg
            if not enough:
                bank = self._build_bank(problem, qid)
                self._write_cache(qid, bank)
        return bank

    # ---------- public API ----------
    def sample_positive(self, problem: Dict[str, Any], qid: str) -> str:
        if self._origin_solution_is_empty(problem):
            return ""
        bank = self.load_or_build_bank(problem, qid)
        pos = bank.get("pos", [])
        #print(len(pos))
        if pos:
            return random.choice(pos)
        return normalize_solution_text(problem.get("solution", ""))

    def sample_negative(self, problem: Dict[str, Any], qid: str) -> str:
        if self._origin_solution_is_empty(problem):
            return ""
        bank = self.load_or_build_bank(problem, qid)
        neg = bank.get("neg", [])
        if neg:
            return random.choice(neg)

        origin = normalize_solution_text(problem.get("solution", ""))
        text = strip_solution_prefix(self._rule_negative_fallback(origin, problem))
        if self._negative_ok(text, origin):
            return text
        return "This reasoning is relevant to the question but leads to the wrong answer."

    def sample_mixed(self, problem: Dict[str, Any], qid: str, neg_prob: float = 0.5) -> str:
        if random.random() < neg_prob:
            return self.sample_negative(problem, qid)
        return self.sample_positive(problem, qid)


def get_augmented_solution_positive(problem: Dict[str, Any], qid: str, args) -> str:
    aug = get_solution_augmentor(args)
    return aug.sample_positive(problem, qid)


def get_augmented_solution_negative(problem: Dict[str, Any], qid: str, args) -> str:
    aug = get_solution_augmentor(args)
    return aug.sample_negative(problem, qid)


def get_augmented_solution_mixed(problem: Dict[str, Any], qid: str, neg_prob: float, args) -> str:
    aug = get_solution_augmentor(args)
    return aug.sample_mixed(problem, qid, neg_prob=neg_prob)


# ----------------------------
# optional prewarm helper
# ----------------------------
def prewarm_solution_cache(problems: Dict[str, Dict[str, Any]], qids: List[str], args):
    aug = get_solution_augmentor(args)
    for qid in qids:
        print(qid)
        problem = dict(problems[qid])
        problem["qid"] = qid
        aug.load_or_build_bank(problem, qid)

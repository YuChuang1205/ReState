import argparse
import copy
import glob
import inspect
import json
import os
import random
import re
import tempfile
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from model import (QwenVLForScienceQA, T5ForMultimodalGeneration,
                   load_qwen_vl_base)
from rich import box
from rich.console import Console
from rich.table import Column, Table
from transformers import (AutoProcessor, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration, TrainerCallback)
from utils_data import QwenVLCollator  # patched Qwen dataset/collator
from utils_data import (ScienceQADatasetImg, ScienceQADatasetQwenVL,
                        ScienceQADatasetStd, img_shape, load_data_img,
                        load_data_std, resolve_dataset_file_path)
from utils_evaluate import get_scores
from utils_prompt_2_12 import *

console = Console(record=True)

import logging

import evaluate
import nltk

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="m3cot", choices=["m3cot", "aokvqa", "scienceqa", "versacot10k"],
                        help="dataset to use under data_root; keeps m3cot as the default")
    parser.add_argument("--data_file", type=str, default=None,
                        help="optional dataset json path; relative paths are resolved under data_root")
    parser.add_argument("--split_file", type=str, default=None,
                        help="optional pid_splits.json path; relative paths are resolved under data_root")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--model", type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--start_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--input_len", type=int, default=4096)
    parser.add_argument("--output_len", type=int, default=64)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--eval_acc", type=int, default=None, help="evaluate accumulation step")
    parser.add_argument("--train_split", type=str, default="minitrain", choices=["train", "trainval", "minitrain"])
    parser.add_argument("--val_split", type=str, default="minival", choices=["test", "val", "minival"])
    parser.add_argument("--test_split", type=str, default="minitest", choices=["test", "minitest"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps before optimizer update")

    parser.add_argument("--solution_origin_pos_prob", type=float, default=0.0, help="probability of using original solution as positive target")

    parser.add_argument("--per_pos_neg", type=int, default=1)
    parser.add_argument("--phase1_cot_num", type=int, default=1)
    parser.add_argument("--phase2_cot_num", type=int, default=1)
    parser.add_argument("--phase2_use_neg_input", action="store_true", help="Whether to perform negative rationale input in the phase2")
    parser.add_argument("--phase2_use_neg_input_only", action="store_true", help="Whether to only perform negative rationale input in the phase2")
    parser.add_argument("--phase2_only_pos_have_rationale", action="store_true", help="phase2_only_pos_have_rationale")
    parser.add_argument("--phase2_only_neg_have_rationale", action="store_true", help="phase2_only_neg_have_rationale")
    parser.add_argument("--phase2_use_pos_output", action="store_true", help="Whether to perform postivate rationale output in the phase2")
    parser.add_argument("--phase2_cot_out_choose_num", type=int, default=1)

    parser.add_argument("--use_txt_enhancement_phase1", action="store_true", help="phase1 use_txt_enhancement")
    parser.add_argument("--use_txt_enhancement_phase2", action="store_true", help="phase2 use_txt_enhancement")


    parser.add_argument("--hardsamplemine", action="store_true", help="hard sample mine")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=1)

    parser.add_argument("--use_answer_token_weight", action="store_true",
                        help="enable an extra answer-focused loss to upweight answer tokens")
    parser.add_argument("--answer_focused_loss_weight", type=float, default=0.5,
                        help="weight coefficient for the extra answer-focused loss")

    parser.add_argument("--use_generate", action="store_true", help="only for baseline to improve inference speed")
    parser.add_argument("--final_eval", action="store_true", help="only evaluate the model at the final epoch")
    parser.add_argument("--user_msg", type=str, default="baseline", help="experiment type in the save_dir")


    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed config json (e.g., ZeRO-3).")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by torchrun/deepspeed.")

    parser.add_argument(
        "--img_type",
        type=str,
        default=None,
        choices=[
            "detr",
            "clip",
            "resnet",
            "vit",
            "dinov2_giant",
            "dinov2_large",
            "blip_large",
            "blip2_flant5_xl",
            "blip2_flant5_xxl",
            "clip_b16",
            "clip_l14_336",
            "convnextv2_huge",
            "sam_base",
            "sam_large",
            "sam_huge",
        ],
        help="type of image features",
    )
    parser.add_argument("--eval_le", type=str, default=None, help="generated rationale for the dev set")
    parser.add_argument("--test_le", type=str, default=None, help="generated rationale for the test set")
    parser.add_argument("--evaluate_dir", type=str, default=None, help="the directory of model for evaluation")

    parser.add_argument("--caption_file", type=str, default="data/captions.json")
    parser.add_argument("--use_caption", action="store_true", help="use image captions or not")

    parser.add_argument(
        "--prompt_format",
        type=str,
        default="QCM-A",
        help="prompt format template",
        choices=["QCM-A", "QCM-E", "QCM-LE", "QCMG-A", "QCMG-AE", "QCML-AE", "QCM-LEA", "QCM-ALE"],
    )
    parser.add_argument("--mix_qcml_qcmg_ae_train", action="store_true", help="mix QCML-AE and QCMG-AE on the training split only")
    parser.add_argument("--qcml_ae_prob", type=float, default=0.3, help="probability of sampling QCML-AE when mixed single-stage training is enabled")
    parser.add_argument("--self_refine_eval_rounds", type=int, default=2, help="number of eval/test self-refine rounds. 1 disables iterative refinement; 2 means QCML-AE -> QCMG-AE once")
    parser.add_argument("--self_refine_eval_rounds_val", type=int, default=0, help="self-refine rounds used only for epoch-end validation; 0 means follow self_refine_eval_rounds")
    parser.add_argument("--eval_max_new_tokens", type=int, default=0, help="max new tokens used only for epoch-end validation; 0 means follow output_len")

    parser.add_argument(
        "--self_refine_init_file",
        type=str,
        default=None,
        help="Optional saved predictions json used as the initial solution input for extra self-refine rounds.",
    )
    parser.add_argument(
        "--self_refine_extra_rounds",
        type=int,
        default=0,
        help="If >0, continue self-refine for this many extra rounds from --self_refine_init_file.",
    )
    parser.add_argument(
        "--self_refine_init_round",
        type=int,
        default=0,
        help="Round index of --self_refine_init_file. 0 means infer from filename or json metadata.",
    )
    parser.add_argument(
        "--self_refine_continue_only",
        action="store_true",
        help="Only run extra self-refine continuation and skip the normal final test predict. Requires --self_refine_init_file and --self_refine_extra_rounds > 0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--qwen_vl", action="store_true", help="Enable Qwen2.5-VL/Qwen3-VL backend")


    parser.add_argument("--use_image_aug", action="store_true", help="enable random image augmentation on training images")
    parser.add_argument("--image_aug_prob", type=float, default=0.5, help="probability of applying image augmentation per training sample")

 
    # ===== Qwen3-VL solution augmentation =====
    parser.add_argument(
        "--solution_gen_mode",
        type=str,
        default="rule",
        choices=["rule", "preprocessed", "qwen_vl_cache", "qwen_vl_online"],
        help="solution augmentation backend: rule / preprocessed / qwen_vl_cache / qwen_vl_online",
    )
    parser.add_argument(
        "--solution_gen_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model used for solution augmentation",
    )
    parser.add_argument(
        "--solution_cache_dir",
        type=str,
        default="./solution_aug_cache",
        help="Directory to store/load generated solution augmentation cache",
    )
    parser.add_argument("--solution_pos_num", type=int, default=1, help="number of positive solution variants")
    parser.add_argument("--solution_neg_num", type=int, default=1, help="number of negative solution variants")
    parser.add_argument("--solution_neg_prob", type=float, default=0.5, help="probability of sampling negative solution in phase2 input")
    parser.add_argument("--solution_refresh_prob", type=float, default=0.05, help="probability of refreshing solution bank")
    parser.add_argument("--solution_bank_max_size", type=int, default=10, help="max cached bank size per qid")
    parser.add_argument("--solution_gen_max_new_tokens", type=int, default=256, help="max new tokens for augmentation generation")
    parser.add_argument("--solution_gen_temperature", type=float, default=0.9, help="generation temperature")
    parser.add_argument("--solution_gen_top_p", type=float, default=0.9, help="generation top_p")
    parser.add_argument("--solution_gen_do_sample", action="store_true", help="whether to use sampling for augmentation generation")
    parser.add_argument("--solution_gen_use_image", action="store_true", help="use image for Qwen3-VL augmentation")
    parser.add_argument("--solution_gen_use_caption", action="store_true", help="use caption for Qwen3-VL augmentation")
    parser.add_argument("--solution_edit_from_origin", action="store_true", help="rewrite the original solution directly when generating positive/negative banks")
    parser.add_argument("--solution_multi_gen_num", type=int, default=1, help="number of solution candidates to ask for in a single augmentation dialogue")
    parser.add_argument("--solution_gen_strict_filter", action="store_true", help="apply strict post-filter to generated solutions")
    parser.add_argument(
        "--solution_gen_device",
        type=str,
        default=None,
        help="Dedicated device for augmentation model, e.g. cuda:0. Leave None for auto single-device.",
    )
    parser.add_argument(
        "--solution_cache_read_only",
        action="store_true",
        help="Read prebuilt cache only; never build cache during training. Recommended for multi-GPU training.",
    )
    parser.add_argument(
        "--student_history_dir",
        type=str,
        default=None,
        help="Directory for saving/loading student-generated solution history (.jsonl per qid).",
    )
    parser.add_argument(
        "--student_history_prob",
        type=float,
        default=0.0,
        help="Probability of sampling a saved student solution as phase2 input during training.",
    )
    parser.add_argument(
        "--student_history_max_size",
        type=int,
        default=10,
        help="Maximum recent student-generated solutions kept/used per qid.",
    )
    parser.add_argument(
        "--student_history_save_steps",
        type=int,
        default=5,
        help="If >0, periodically save training predictions into student history.",
    )
    parser.add_argument(
        "--student_history_save_mode",
        type=str,
        default="argmax",
        choices=["generate", "argmax"],
        help="How to save training-time student history: "
             "'generate' = greedy autoregressive decode; "
             "'argmax' = token-wise argmax from training forward logits.",
    )
    parser.add_argument(
        "--student_history_start_step",
        type=int,
        default=200,
        help="Only start saving student history when global_step is greater than this value.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir or experiment dir. Supports both .../checkpoint-xxxx and experiment root.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="If set and --resume_from_checkpoint is None, auto-resume from latest checkpoint under current save_dir.",
    )
    parser.add_argument(
        "--round1_repeat",
        type=int,
        default=1,
        help="repeat times for round-1 empty-solution samples in dual-round training",
    )
    parser.add_argument(
        "--round2_repeat",
        type=int,
        default=1,
        help="repeat times for round-2 non-empty/noisy-solution samples in dual-round training",
    )
    parser.add_argument("--qwen_image_min_tokens", type=int, default=256)
    parser.add_argument("--qwen_image_max_tokens", type=int, default=1280)


    args = parser.parse_args()
    return args



def _find_latest_ckpt_dir(root: str) -> str:
    # root=.../checkpoint-3000 or root=.../exp_dir
    if os.path.isfile(os.path.join(root, "pytorch_model.bin")) or os.path.isfile(os.path.join(root, "model.safetensors")):
        return root

    # checkpoint-xxxx
    ckpts = glob.glob(os.path.join(root, "checkpoint-*"))
    if not ckpts:
        return root

    def _step(p):
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_step)
    return ckpts[-1]


def _resolve_resume_ckpt(args, save_dir: str):
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)

    if resume_ckpt is not None and str(resume_ckpt).strip() != "":
        resume_ckpt = _find_latest_ckpt_dir(str(resume_ckpt).strip())
        if os.path.exists(resume_ckpt):
            return resume_ckpt
        raise FileNotFoundError(f"--resume_from_checkpoint not found: {resume_ckpt}")

    if getattr(args, "auto_resume", False):
        auto_ckpt = _find_latest_ckpt_dir(save_dir)
        if auto_ckpt != save_dir and os.path.exists(auto_ckpt):
            return auto_ckpt

    return None


def load_finetuned_weights_into_wrapper(wrapper_model, ckpt_dir: str):
    ckpt_dir = _find_latest_ckpt_dir(ckpt_dir)

    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    st_path = os.path.join(ckpt_dir, "model.safetensors")

    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path)
    elif os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        candidates = glob.glob(os.path.join(ckpt_dir, "**", "*model_states.pt"), recursive=True)
        raise FileNotFoundError(
            f"Cannot find pytorch_model.bin/model.safetensors in: {ckpt_dir}\n"
            f"Found possible deepspeed shards: {candidates[:3]}\n"
            f"Suggestion: point --evaluate_dir to a HF checkpoint folder (checkpoint-xxxx) "
            f"that contains pytorch_model.bin, or convert ZeRO shards to fp32/16 single weight."
        )

    missing, unexpected = wrapper_model.load_state_dict(state, strict=False)
    print(f"[Load FT] Loaded weights from: {ckpt_dir}")
    if missing:
        print(f"[Load FT] Missing keys (show up to 20): {missing[:20]}")
    if unexpected:
        print(f"[Load FT] Unexpected keys (show up to 20): {unexpected[:20]}")




# =========================
# utils
# =========================
def safe_format(value):
    if isinstance(value, (int, float)):
        return f"{value:.5f}"
    return value


def build_qwen_dataset_safely(dataset_cls, **kwargs):
    sig = inspect.signature(dataset_cls.__init__)
    params = sig.parameters

    filtered = {k: v for k, v in kwargs.items() if k in params}

    return dataset_cls(**filtered)

def build_collator_safely(collator_cls, **kwargs):
    sig = inspect.signature(collator_cls.__init__)
    params = sig.parameters
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return collator_cls(**filtered)


def get_qwen_vl_pixel_factor(model_name: str) -> int:
    name = model_name.lower()
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name or "qvq" in name:
        return 28
    return 32


def build_seq2seq_training_args(output_dir: str, **kwargs):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters


    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if "logging_strategy" in kwargs and "logging_strategy" not in params:
        kwargs.pop("logging_strategy")
    if "save_strategy" in kwargs and "save_strategy" not in params:
        kwargs.pop("save_strategy")

    filtered = {k: v for k, v in kwargs.items() if k in params}

    if "output_dir" in params:
        return Seq2SeqTrainingArguments(output_dir=output_dir, **filtered)
    else:
        return Seq2SeqTrainingArguments(output_dir, **filtered)


# =========================
# callback
# =========================
class DelayEvalSaveCallback(TrainerCallback):
    def __init__(self, start_epoch: int = 100):
        self.start_epoch = start_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch < self.start_epoch:
            control.should_evaluate = False
            control.should_save = False
        return control


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, raw_args=None, processor_ref=None, **kwargs):
        self.raw_args = raw_args
        self.processor_ref = processor_ref
        super().__init__(*args, **kwargs)

        self._reset_last_predict_round_cache()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.args.output_dir, f"loss_log_{current_time}.txt")

        with open(self.log_filename, "w") as log_file:
            log_file.write(f"Training Loss Log - {current_time}\n")
            log_file.write("Step,   Total Loss,        CE Loss,           Contrastive Loss,   Answer Focused Loss\n")

    def _reset_last_predict_round_cache(self):
        self._last_predict_round_outputs = None
        self._last_predict_round_prompt_formats = None
        self._last_predict_global_round_indices = None

    def _get_loss_tokenizer(self):
        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):  # Qwen processor
            tokenizer = tokenizer.tokenizer
        return tokenizer

    def _extract_answer_prefix_text(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).strip()
        if text == "":
            return ""

        if (
            re.search(r"The answer is\s*\([A-Z]\)", text, flags=re.IGNORECASE) is None
            and re.search(r"^Answer\s*:", text, flags=re.IGNORECASE) is None
        ):
            return ""

        m = re.search(r"\bBECAUSE\s*:", text, flags=re.IGNORECASE)
        if m is not None:
            text = text[:m.start()].strip()
        return text

    def _compute_answer_focused_loss(self, logits, labels):
        if self.raw_args is None or not bool(getattr(self.raw_args, "use_answer_token_weight", False)):
            return None
        if logits is None or labels is None:
            return None
        if not torch.is_tensor(logits) or not torch.is_tensor(labels):
            return None

        tokenizer = self._get_loss_tokenizer()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        ce_labels = labels.detach().clone().to(logits.device)
        ce_labels[ce_labels == pad_id] = -100

        decode_labels = torch.where(
            ce_labels != -100,
            ce_labels,
            torch.full_like(ce_labels, pad_id),
        )

        answer_mask = torch.zeros_like(ce_labels, dtype=logits.dtype, device=logits.device)
        has_answer_tokens = False

        for i in range(decode_labels.size(0)):
            label_ids = decode_labels[i].tolist()
            text = tokenizer.decode(
                label_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            answer_prefix = self._extract_answer_prefix_text(text)
            if not answer_prefix:
                continue

            answer_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
            if len(answer_ids) == 0:
                continue

            valid_len = int((ce_labels[i] != -100).sum().item())
            if valid_len <= 0:
                continue

            focus_len = min(len(answer_ids), valid_len)
            if focus_len <= 0:
                continue

            answer_mask[i, :focus_len] = 1.0
            has_answer_tokens = True

        if not has_answer_tokens:
            return None

        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ce_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(ce_labels)

        focus_mask = answer_mask * (ce_labels != -100).to(logits.dtype)
        denom = focus_mask.sum()
        if denom.item() <= 0:
            return None

        return (token_loss * focus_mask).sum() / denom

    def _get_student_history_dir(self):
        if self.raw_args is None:
            return None
        history_dir = getattr(self.raw_args, "student_history_dir", None)
        if history_dir is None:
            save_steps = int(getattr(self.raw_args, "student_history_save_steps", 0))
            history_prob = float(getattr(self.raw_args, "student_history_prob", 0.0))
            if save_steps > 0 or history_prob > 0:
                history_dir = os.path.join(self.args.output_dir, "student_solution_history")
        if history_dir is None:
            return None
        os.makedirs(history_dir, exist_ok=True)
        return history_dir


    def _should_save_student_history(self):
        if self.raw_args is None:
            return False
        save_steps = int(getattr(self.raw_args, "student_history_save_steps", 0))
        return save_steps > 0 and self.model.training

    def _should_trigger_student_history_save(self, step: int) -> bool:
        if self.raw_args is None:
            return False

        save_steps = int(getattr(self.raw_args, "student_history_save_steps", 0))
        start_step = int(getattr(self.raw_args, "student_history_start_step", 0))

        if save_steps <= 0:
            return False

        if step <= start_step:
            return False

        return (step - start_step) % save_steps == 0


    def _append_student_history_batch(self, qids, pred_token_ids, target_labels=None):
        history_dir = self._get_student_history_dir()
        if history_dir is None or not qids:
            return

        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):  # Qwen processor
            tokenizer = tokenizer.tokenizer

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos_id = getattr(tokenizer, "eos_token_id", None)

        pred_token_ids = pred_token_ids.detach().clone()
        pred_token_ids = torch.where(
            pred_token_ids >= 0,
            pred_token_ids,
            torch.full_like(pred_token_ids, pad_id)
        )

        if target_labels is not None and torch.is_tensor(target_labels):
            target_labels = target_labels.detach().clone()

        max_size = max(1, int(getattr(self.raw_args, "student_history_max_size", 10)))

        def _first_stop_idx(seq_ids, stop_ids):
            for i, tid in enumerate(seq_ids):
                if tid in stop_ids:
                    return i
            return len(seq_ids)

        for idx, qid in enumerate(qids):
            pred_ids = pred_token_ids[idx].tolist()

            if target_labels is not None:
                label_ids = target_labels[idx].tolist()
                valid_len = _first_stop_idx(label_ids, {pad_id})
                pred_ids = pred_ids[:valid_len]

            stop_ids = {pad_id}
            if eos_id is not None:
                stop_ids.add(eos_id)
            pred_ids = pred_ids[:_first_stop_idx(pred_ids, stop_ids)]

            if not pred_ids:
                continue

            text = tokenizer.decode(
                pred_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            solution_text = extract_solution_from_ae_output(text).strip()
            if not solution_text or is_empty_solution_text(solution_text):
                continue

            path = os.path.join(history_dir, f"{qid}.jsonl")
            try:
                existing_texts = []
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                old_text = str(obj.get("text", "")).strip()
                            except Exception:
                                old_text = ""
                            if old_text:
                                existing_texts.append(old_text)

                if solution_text in existing_texts:
                    continue

                existing_texts.append(solution_text)
                existing_texts = existing_texts[-max_size:]

                with open(path, "w", encoding="utf-8") as f:
                    for t in existing_texts:
                        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            except Exception:
                pass


    def _argmax_student_history_batch(self, outputs):
        logits = getattr(outputs, "logits", None)
        if logits is None:
            return None
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits is None or not torch.is_tensor(logits):
            return None
        return logits.detach().argmax(dim=-1)

    def _generate_student_history_batch(self, model, inputs):
        gen_model = model
        if not hasattr(gen_model, "generate"):
            if hasattr(gen_model, "module") and hasattr(gen_model.module, "generate"):
                gen_model = gen_model.module
            elif hasattr(self, "model") and hasattr(self.model, "generate"):
                gen_model = self.model
            else:
                return None

        gen_inputs = {}
        for k, v in inputs.items():
            if k in {"labels", "pos_labels", "neg_labels", "qids", "decoder_input_ids", "decoder_attention_mask"}:
                continue
            gen_inputs[k] = v

        max_new_tokens = self._get_effective_eval_max_new_tokens()
        if max_new_tokens is None:
            max_new_tokens = int(getattr(self.raw_args, "output_len", 64)) if self.raw_args is not None else 64

        was_training = bool(getattr(gen_model, "training", False))
        try:
            if was_training and hasattr(gen_model, "eval"):
                gen_model.eval()
            with torch.no_grad():
                pred_token_ids = gen_model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                )
            return pred_token_ids
        finally:
            if was_training and hasattr(gen_model, "train"):
                gen_model.train()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        qids = inputs.pop("qids", None)
        outputs = model(**inputs)
        loss = outputs.loss

        answer_focused_loss = self._compute_answer_focused_loss(
            getattr(outputs, "logits", None),
            inputs.get("labels", None),
        )
        answer_focused_loss_weight = 0.0
        if self.raw_args is not None:
            answer_focused_loss_weight = float(getattr(self.raw_args, "answer_focused_loss_weight", 0.5))
        weighted_answer_focused_loss = None
        if answer_focused_loss is not None:
            weighted_answer_focused_loss = answer_focused_loss * answer_focused_loss_weight
            loss = loss + weighted_answer_focused_loss

        if self._should_save_student_history() and qids is not None:
            step = self.state.global_step
            if self._should_trigger_student_history_save(step):
                save_mode = str(getattr(self.raw_args, "student_history_save_mode", "generate")).lower()

                if save_mode == "argmax":
                    pred_token_ids = self._argmax_student_history_batch(outputs)
                else:
                    pred_token_ids = self._generate_student_history_batch(model, inputs)

                if pred_token_ids is not None:
                    self._append_student_history_batch(qids, pred_token_ids, target_labels=None)

        step = self.state.global_step
        if step % 100 == 0:
            pv = inputs.get("pixel_values", None)
            g  = inputs.get("image_grid_thw", None)
            #print(f"[rank? step={self.state.global_step}] pixel_values={pv is not None}, image_grid_thw={g is not None}, keys={list(inputs.keys())}")
            total_loss = loss
            ce = getattr(outputs, "ce_loss", None)
            ct = getattr(outputs, "contrastive_loss", None)
            afl = weighted_answer_focused_loss

            total_loss_value = total_loss.mean().item() if total_loss is not None else "N/A"
            ce_loss_value = ce.mean().item() if ce is not None else "N/A"
            contrastive_loss_value = ct.mean().item() if ct is not None else "N/A"
            answer_focused_loss_value = afl.mean().item() if afl is not None else "N/A"

            with open(self.log_filename, "a") as log_file:
                log_file.write(f"{step}, {total_loss_value}, {ce_loss_value}, {contrastive_loss_value}, {answer_focused_loss_value}\n")

        return (loss, outputs) if return_outputs else loss

    def _get_effective_self_refine_rounds(self, metric_key_prefix="eval") -> int:
        if self.raw_args is None:
            return 1
        if metric_key_prefix == "eval":
            rounds = int(getattr(self.raw_args, "self_refine_eval_rounds_val", 0))
            if rounds <= 0:
                rounds = int(getattr(self.raw_args, "self_refine_eval_rounds", 1))
        else:
            rounds = int(getattr(self.raw_args, "self_refine_eval_rounds", 1))
        return max(1, rounds)

    def _get_effective_eval_max_new_tokens(self) -> int:
        if self.raw_args is None:
            return None
        val_max_new_tokens = int(getattr(self.raw_args, "eval_max_new_tokens", 0))
        if val_max_new_tokens > 0:
            return val_max_new_tokens
        return getattr(self.raw_args, "output_len", None)


    def _should_use_iterative_refine(self, dataset, metric_key_prefix="eval") -> bool:
        if dataset is None or self.raw_args is None:
            return False

        rounds = self._get_effective_self_refine_rounds(metric_key_prefix=metric_key_prefix)
        if rounds <= 1:
            return False

        if not bool(getattr(self.raw_args, "qwen_vl", False)):
            return False

        if self.processor_ref is None:
            return False

        if not (hasattr(dataset, "problems") and hasattr(dataset, "qids")):
            return False

        prompt_format = str(getattr(self.raw_args, "prompt_format", ""))
        use_phase2_txt = bool(getattr(self.raw_args, "use_txt_enhancement_phase2", False))

        return use_phase2_txt or prompt_format in {"QCMG-AE", "QCML-AE"}

    def _decode_predictions_and_targets(self, predict_output):
        if getattr(self.raw_args, "use_generate", False):
            preds, targets = predict_output.predictions, predict_output.label_ids
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = predict_output.predictions[0]
            targets = predict_output.label_ids
            preds = preds.argmax(axis=2)

        #tokenizer = self.tokenizer
        tokenizer = self.processing_class
        if hasattr(tokenizer, "tokenizer"):  # Qwen processor
            tokenizer = tokenizer.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets_text = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return preds_text, targets_text

    def _write_round_solution_file(self, preds_text, expected_len: int = None):
        # DDP/DeepSpeed prediction may pad examples across ranks. Keep only real qids.
        if expected_len is not None:
            expected_len = int(expected_len)
            if len(preds_text) > expected_len:
                # print(
                #     f"[Self-Refine Round Truncate] round preds={len(preds_text)}, "
                #     f"qids={expected_len}. Truncate to true qids length."
                # )
                preds_text = preds_text[:expected_len]
            elif len(preds_text) < expected_len:
                raise ValueError(
                    f"Current round produced only {len(preds_text)} preds, "
                    f"but dataset has {expected_len} qids."
                )

        refined = [extract_solution_from_ae_output(p).strip() for p in preds_text]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="self_refine_round_", dir=self.args.output_dir, delete=False)
        json.dump({"preds": refined}, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return tmp.name

    def _build_round_dataset(self, base_dataset, prompt_format: str, test_le_file: str = None):
        round_args = argparse.Namespace(**vars(self.raw_args))
        round_args.prompt_format = prompt_format
        round_args.mix_qcml_qcmg_ae_train = False
        round_args.eval_le = None
        round_args.test_le = None

        return build_qwen_dataset_safely(
            base_dataset.__class__,
            problems=base_dataset.problems,
            qids=base_dataset.qids,
            processor=self.processor_ref,
            source_len=base_dataset.source_len,
            target_len=base_dataset.target_len,
            args=round_args,
            test_le=test_le_file,
            split=getattr(base_dataset, "split", "eval"),
        )

    def _read_saved_preds_in_qid_order(self, prediction_file: str, qids):
        with open(prediction_file, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            preds = (
                obj.get("preds", None)
                or obj.get("predictions", None)
                or obj.get("rationales", None)
                or obj.get("results", None)
            )
        else:
            preds = obj

        if isinstance(preds, dict):
            ordered = []
            for qid in qids:
                sqid = str(qid)
                if sqid not in preds:
                    raise KeyError(f"Missing qid={sqid} in saved prediction file: {prediction_file}")
                ordered.append(preds[sqid])
            preds = ordered

        if not isinstance(preds, list):
            raise ValueError(
                f"Unsupported saved prediction format in {prediction_file}. "
                f"Expected a list/dict under key 'preds' or a raw list."
            )

        true_len = len(qids)
        if len(preds) > true_len:
            preds = preds[:true_len]
        elif len(preds) < true_len:
            raise ValueError(
                f"Saved prediction file has only {len(preds)} preds, "
                f"but dataset has {true_len} qids: {prediction_file}"
            )
        return [str(x).strip() for x in preds]

    def _prepare_continue_solution_file(self, prediction_file: str, dataset):
        # Saved round files contain full model outputs. Reuse the exact same
        # internal round-to-round handoff function, so continuation from a saved
        # round is identical to the original self-refine test path.
        preds = self._read_saved_preds_in_qid_order(prediction_file, dataset.qids)
        return self._write_round_solution_file(
            preds,
            expected_len=len(dataset.qids),
        )

    def _infer_self_refine_round_from_file(self, prediction_file: str) -> int:
        # Priority: explicit CLI value > json metadata > filename pattern roundXX > 0.
        if self.raw_args is not None:
            explicit_round = int(getattr(self.raw_args, "self_refine_init_round", 0))
            if explicit_round > 0:
                return explicit_round

        try:
            with open(prediction_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for key in ["round_idx", "self_refine_round", "self_refine_rounds"]:
                    value = obj.get(key, None)
                    if value is not None:
                        value = int(value)
                        if value > 0:
                            return value
        except Exception:
            pass

        name = os.path.basename(str(prediction_file))
        m = re.search(r"round(\d+)", name, flags=re.IGNORECASE)
        if m is not None:
            return int(m.group(1))
        return 0

    def predict_from_self_refine_file(
        self,
        test_dataset,
        init_prediction_file: str,
        extra_rounds: int,
        ignore_keys=None,
        metric_key_prefix="test",
        max_new_tokens=None,
    ):
        self._reset_last_predict_round_cache()

        extra_rounds = int(extra_rounds)
        if extra_rounds <= 0:
            raise ValueError("extra_rounds must be > 0 when continuing self-refine.")
        if init_prediction_file is None or str(init_prediction_file).strip() == "":
            raise ValueError("--self_refine_init_file is required for extra self-refine continuation.")
        if not bool(getattr(self.raw_args, "qwen_vl", False)):
            raise ValueError("Extra self-refine continuation is implemented for Qwen-VL mode only.")
        if self.processor_ref is None:
            raise ValueError("processor_ref is required for extra self-refine continuation.")
        if not (hasattr(test_dataset, "problems") and hasattr(test_dataset, "qids")):
            raise ValueError("test_dataset must expose problems and qids for extra self-refine continuation.")

        temp_files = []
        round_outputs = []
        round_prompt_formats = []
        round_global_indices = []
        unified_prompt_format = "QCMG-AE"
        start_round = self._infer_self_refine_round_from_file(init_prediction_file)
        final_output = None

        try:
            prev_solution_file = self._prepare_continue_solution_file(init_prediction_file, test_dataset)
            temp_files.append(prev_solution_file)

            for eidx in range(extra_rounds):
                round_dataset = self._build_round_dataset(
                    test_dataset,
                    prompt_format=unified_prompt_format,
                    test_le_file=prev_solution_file,
                )

                current_output = super().predict(
                    test_dataset=round_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                    max_new_tokens=max_new_tokens,
                )
                final_output = current_output

                round_outputs.append(current_output)
                round_prompt_formats.append(unified_prompt_format)
                round_global_indices.append(start_round + eidx + 1)

                if eidx < extra_rounds - 1:
                    preds_text, _ = self._decode_predictions_and_targets(current_output)
                    prev_solution_file = self._write_round_solution_file(
                        preds_text,
                        expected_len=len(test_dataset.qids),
                    )
                    temp_files.append(prev_solution_file)

            self._last_predict_round_outputs = round_outputs
            self._last_predict_round_prompt_formats = round_prompt_formats
            self._last_predict_global_round_indices = round_global_indices

            if final_output is not None and hasattr(final_output, "metrics") and final_output.metrics is not None:
                final_output.metrics[f"{metric_key_prefix}_self_refine_continue_from_round"] = start_round
                final_output.metrics[f"{metric_key_prefix}_self_refine_extra_rounds"] = extra_rounds
                final_output.metrics[f"{metric_key_prefix}_self_refine_total_rounds"] = start_round + extra_rounds

            return final_output

        finally:
            for path in temp_files:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass


    def _iterative_refine_predict(self, dataset, ignore_keys=None, metric_key_prefix="eval", max_new_tokens=None):
        self._reset_last_predict_round_cache()

        rounds = self._get_effective_self_refine_rounds(metric_key_prefix=metric_key_prefix)
        temp_files = []
        final_output = None
        used_rounds = 0

        round_outputs = []
        round_prompt_formats = []

        unified_prompt_format = "QCMG-AE"

        try:
            # round-0 [EMPTY_SOLUTION]
            empty_preds = [get_empty_solution_text()] * len(dataset.qids)
            init_solution_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix="self_refine_round_init_",
                dir=self.args.output_dir,
                delete=False,
            )
            json.dump({"preds": empty_preds}, init_solution_file, ensure_ascii=False, indent=2)
            init_solution_file.close()

            prev_solution_file = init_solution_file.name
            temp_files.append(prev_solution_file)

            for ridx in range(rounds):
                round_dataset = self._build_round_dataset(
                    dataset,
                    prompt_format=unified_prompt_format,
                    test_le_file=prev_solution_file,
                )

                # For epoch-end validation, intermediate self-refine rounds are only used
                # to build the next-round solution file, so skip compute_metrics to save time.
                # The final round still keeps compute_metrics for model selection.
                original_compute_metrics = self.compute_metrics
                skip_intermediate_metrics = (metric_key_prefix == "eval" and ridx < rounds - 1)
                old_expected_len = getattr(self.raw_args, "_metric_expected_len", None) if self.raw_args is not None else None
                had_expected_len = self.raw_args is not None and hasattr(self.raw_args, "_metric_expected_len")
                try:
                    if self.raw_args is not None:
                        setattr(self.raw_args, "_metric_expected_len", len(dataset.qids))
                    if skip_intermediate_metrics:
                        self.compute_metrics = None

                    current_output = super().predict(
                        test_dataset=round_dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=metric_key_prefix,
                        max_new_tokens=max_new_tokens,
                    )
                finally:
                    self.compute_metrics = original_compute_metrics
                    if self.raw_args is not None:
                        if had_expected_len:
                            setattr(self.raw_args, "_metric_expected_len", old_expected_len)
                        elif hasattr(self.raw_args, "_metric_expected_len"):
                            delattr(self.raw_args, "_metric_expected_len")

                final_output = current_output
                used_rounds = ridx + 1

                round_outputs.append(current_output)
                round_prompt_formats.append(unified_prompt_format)

                if ridx < rounds - 1:
                    preds_text, _ = self._decode_predictions_and_targets(current_output)
                    prev_solution_file = self._write_round_solution_file(
                        preds_text,
                        expected_len=len(dataset.qids),
                    )
                    temp_files.append(prev_solution_file)

            self._last_predict_round_outputs = round_outputs
            self._last_predict_round_prompt_formats = round_prompt_formats

            if final_output is not None and hasattr(final_output, "metrics") and final_output.metrics is not None:
                final_output.metrics[f"{metric_key_prefix}_self_refine_rounds"] = used_rounds
                final_output.metrics[f"{metric_key_prefix}_self_refine_prompt"] = unified_prompt_format

            return final_output

        finally:
            for path in temp_files:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if inputs is not None:
            inputs = dict(inputs)
            inputs.pop("qids", None)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self._should_use_iterative_refine(dataset, metric_key_prefix=metric_key_prefix):
            output = self._iterative_refine_predict(
                dataset=dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                max_new_tokens=self._get_effective_eval_max_new_tokens(),
            )
            metrics = output.metrics if output is not None else {}
            if metrics:
                self.log(metrics)
            return metrics
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", max_new_tokens=None):
        self._reset_last_predict_round_cache()

        if self._should_use_iterative_refine(test_dataset, metric_key_prefix=metric_key_prefix):
            return self._iterative_refine_predict(
                dataset=test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                max_new_tokens=max_new_tokens,
            )
        return super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            max_new_tokens=max_new_tokens,
        )


# =========================
# Trainer
# =========================
def T5Trainer(dataframe, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # # evaluate_dir
    # if args.evaluate_dir is not None:
    #     args.model = args.evaluate_dir

    # auto qwen_vl
    if (("qwen" in args.model.lower()) and ("vl" in args.model.lower())):
        args.qwen_vl = True

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    problems = dataframe["problems"]
    qids = dataframe["qids"]
    train_qids = qids["train"]
    test_qids = qids["test"]
    val_qids = qids["val"]
   
    model_name = args.model.replace("/", "-")
    gpu_count = torch.cuda.device_count()
    save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    # ======================
    # 1) tokenizer/processor + model + dataset/collator
    # ======================
    if args.qwen_vl:
        if args.img_type is not None:
            print(f"[Warning] Qwen-VL backend ignores --img_type={args.img_type}. Please remove it from .sh for cleanliness.")

        factor = get_qwen_vl_pixel_factor(args.model)
        min_pixels = args.qwen_image_min_tokens * factor * factor
        max_pixels = args.qwen_image_max_tokens * factor * factor

        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        if hasattr(processor, "image_processor"):
            processor.image_processor.size = {
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token


        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        is_train = (args.evaluate_dir is None)  

        device_map = None if (is_distributed or is_train) else "auto"

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

        base_model = load_qwen_vl_base(
            args.model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

        model = QwenVLForScienceQA(base_model=base_model, tokenizer=tokenizer)


        if args.evaluate_dir is not None:
            load_finetuned_weights_into_wrapper(model, args.evaluate_dir)


        if is_train:
            if hasattr(model.model, "config"):
                model.model.config.use_cache = False
            # if hasattr(model.model, "gradient_checkpointing_enable"):
            #     model.model.gradient_checkpointing_enable()
            if hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        train_set = build_qwen_dataset_safely(
            ScienceQADatasetQwenVL,
            problems=problems,
            qids=train_qids,
            processor=processor,
            source_len=args.input_len,      
            target_len=args.output_len,     
            args=args,
            test_le=None,   
            split="train",
        )

        eval_set = build_qwen_dataset_safely(
            ScienceQADatasetQwenVL,
            problems=problems,
            qids=val_qids,
            processor=processor,
            source_len=args.input_len,      
            target_len=args.output_len,     
            args=args,
            test_le=args.eval_le,
            split="eval",
        )

        test_set = build_qwen_dataset_safely(
            ScienceQADatasetQwenVL,
            problems=problems,
            qids=test_qids,
            processor=processor,
            source_len=args.input_len,      
            target_len=args.output_len,     
            args=args,
            test_le=args.test_le,
            split="test",
        )

        datacollator = build_collator_safely(
            QwenVLCollator,
            processor=processor,
            tokenizer=getattr(processor, "tokenizer", None),
            source_len=args.input_len,
        )


    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        processor = None

        if args.img_type is not None:
            patch_size = img_shape[args.img_type]
            model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)

            name_maps = dataframe["name_maps"]
            image_features = dataframe["image_features"]

            train_set = ScienceQADatasetImg(
                problems,
                train_qids,
                name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
                split="train",
            )
            eval_set = ScienceQADatasetImg(
                problems,
                val_qids,
                name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
                args.eval_le,
                split="eval",
            )
            test_set = ScienceQADatasetImg(
                problems,
                test_qids,
                name_maps,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                image_features,
                args.test_le,
                split="test",
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model)
            train_set = ScienceQADatasetStd(
                problems,
                train_qids,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                split="train",
            )
            eval_set = ScienceQADatasetStd(
                problems,
                val_qids,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                args.eval_le,
                split="eval",
            )
            test_set = ScienceQADatasetStd(
                problems,
                test_qids,
                tokenizer,
                args.input_len,
                args.output_len,
                args,
                args.test_le,
                split="test",
            )

        datacollator = DataCollatorForSeq2Seq(tokenizer)


    if args.hardsamplemine and hasattr(model, "set_hardsamplemine"):
        model.set_hardsamplemine(args.hardsamplemine, margin=args.margin, alpha=args.alpha, topk=args.topk)

    if hasattr(model, "num_parameters"):
        print("model parameters: ", model.num_parameters())

    def extract_ans(ans):
        pattern = re.compile(r"The answer is \(([A-Z])\).*")
        res = pattern.findall(ans)
        if len(res) > 0:
            return res[0]
        return "FAILED"

    def build_test_output_data_from_predict_output(predict_output, eval_qids):
        if args.use_generate:
            preds, targets = predict_output.predictions, predict_output.label_ids
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = predict_output.predictions[0]
            targets = predict_output.label_ids
            preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        true_len = len(eval_qids)
        preds = preds[:true_len]
        targets = targets[:true_len]

        results_ans, results_rationale, results_reference = {}, {}, {}
        num_fail = 0


        for idx, qid in enumerate(eval_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)

            num_choices = len(
                problems[str(qid)]["choices"]
            )
            valid_options = args.options[:num_choices]

            if extract_pred in valid_options:
                extract_pred = valid_options.index(
                    extract_pred
                )
            else:
                num_fail += 1
                extract_pred = random.randrange(
                    num_choices
                )

            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        scores = get_scores(
            results_ans,
            results_rationale,
            results_reference,
            resolve_dataset_file_path(args),
            test_qids=eval_qids,
        )

        preds_strip = [p.strip() for p in preds]
        output_data = {
            "num_fail": num_fail,
            "scores": scores,
            "preds": preds_strip,
            "labels": targets,
            "self_refine_rounds": int(getattr(args, "self_refine_eval_rounds", 1)),
        }
        return output_data

    def save_test_output_data(output_data, split="test", round_idx=None, prompt_format=None):
        if round_idx is None:
            filename = f"predictions_ans_{split}.json"
        else:
            fmt = (prompt_format or "unknown").lower().replace("-", "_")
            filename = f"predictions_ans_{split}_round{int(round_idx):02d}_{fmt}.json"

        output_prediction_file = os.path.join(save_dir, filename)
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(output_data, indent=4, ensure_ascii=False))
        return output_prediction_file

    def resolve_self_refine_init_file(path):
        if path is None or str(path).strip() == "":
            return None
        path = str(path).strip()
        candidates = [path]
        if not os.path.isabs(path):
            candidates.append(os.path.join(save_dir, path))
            candidates.append(os.path.join(args.output_dir, path))
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            f"--self_refine_init_file not found: {path}. "
            f"Checked: {candidates}"
        )

    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # DDP/DeepSpeed prediction may pad extra examples. In iterative eval/predict,
        # CustomSeq2SeqTrainer sets args._metric_expected_len to the real qid count.
        expected_len = getattr(args, "_metric_expected_len", None)
        if expected_len is not None:
            expected_len = int(expected_len)
            preds = preds[:expected_len]
            targets = targets[:expected_len]

        correct = 0
        total = 0
        num_fail = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = extract_ans(targets[idx])
            if reference not in args.options:
                # This should only happen for invalid/padded labels. Do not let it become a false correct.
                continue
            reference_idx = args.options.index(reference)

            extract_pred = extract_ans(pred)
            if extract_pred in args.options:
                pred_idx = args.options.index(extract_pred)
            else:
                # Keep epoch-end eval consistent with final test: FAILED/invalid prediction = random guess.
                num_fail += 1
                #pred_idx = random.choice(range(len(args.options)))
                pred_idx = -1

            if reference_idx == pred_idx:
                correct += 1
            total += 1

        return {
            "accuracy": 1.0 * correct / max(total, 1),
            "num_fail": num_fail,
        }

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(tokenizer.encode(p)) for p in preds]
        result["gen_len"] = float(np.mean(prediction_lens))
        return result


    use_qwenvl_solution_aug = (
            getattr(args, "solution_gen_mode", "rule") in {"qwen_vl_cache", "qwen_vl_online"}
            and (getattr(args, "use_txt_enhancement_phase1", False) or getattr(args, "use_txt_enhancement_phase2",
                                                                               False)))
    solution_aug_read_only = bool(getattr(args, "solution_cache_read_only", False))

    if use_qwenvl_solution_aug and (not solution_aug_read_only):
        train_num_workers = 0
        train_pin_memory = False
    else:
        train_num_workers = 16
        train_pin_memory = True


    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    eval_gen_max_new_tokens = args.eval_max_new_tokens if getattr(args, "eval_max_new_tokens", 0) > 0 else args.output_len
    effective_deepspeed = args.deepspeed_config if args.evaluate_dir is None else None
    if args.final_eval:
        training_args = build_seq2seq_training_args(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_new_tokens=eval_gen_max_new_tokens,
            report_to="none",
            dataloader_num_workers=train_num_workers,
            dataloader_pin_memory=train_pin_memory,
            ddp_find_unused_parameters=True,
            bf16=use_bf16,
            fp16=(not use_bf16),
            gradient_checkpointing=True,
            deepspeed=effective_deepspeed,
            save_safetensors=False,
            remove_unused_columns=False,
        )
    else:
        training_args = build_seq2seq_training_args(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format in ["QCMG-A", "QCM-A", "QCMG-AE", "QCML-AE"] else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_new_tokens=eval_gen_max_new_tokens,
            load_best_model_at_end=True,
            report_to="none",
            dataloader_num_workers=train_num_workers,
            dataloader_pin_memory=train_pin_memory,
            ddp_find_unused_parameters=True,
            bf16=use_bf16,
            fp16=(not use_bf16),
            gradient_checkpointing=True,
            deepspeed=effective_deepspeed,
            save_safetensors=False,
            remove_unused_columns=False,
        )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        #tokenizer=tokenizer,
        processing_class=processor if processor is not None else tokenizer,
        compute_metrics=compute_metrics_acc if args.prompt_format in ["QCMG-A", "QCM-A", "QCMG-AE", "QCML-AE"] else compute_metrics_rougel,
        callbacks=[DelayEvalSaveCallback(start_epoch=args.start_epoch)],
        raw_args=args,
        processor_ref=processor,
    )

    # ---------------------------------------------------------
    # DDP + gradient checkpointing may hit "marked ready twice".
    # Workaround: static graph
    # ---------------------------------------------------------
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if dist.is_available() and dist.is_initialized():
        if isinstance(trainer.model, DDP):
            # PyTorch recommended workaround for reentrant checkpointing + DDP
            trainer.model._set_static_graph()



    if args.evaluate_dir is None:
        resume_ckpt = _resolve_resume_ckpt(args, save_dir)
        if resume_ckpt is not None:
            print(f"[Resume] resume_from_checkpoint = {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()

        trainer.save_model(save_dir)


    if hasattr(model, "set_hardsamplemine"):
        model.set_hardsamplemine(False)


    continue_only = bool(getattr(args, "self_refine_continue_only", False))
    extra_rounds = int(getattr(args, "self_refine_extra_rounds", 0))
    init_file = resolve_self_refine_init_file(getattr(args, "self_refine_init_file", None))

    if continue_only and (init_file is None or extra_rounds <= 0):
        raise ValueError(
            "--self_refine_continue_only requires --self_refine_init_file and "
            "--self_refine_extra_rounds > 0."
        )

    predict_results = None
    if not continue_only:
        predict_results = trainer.predict(test_dataset=test_set, max_new_tokens=args.output_len)
        if trainer.is_world_process_zero():
            round_outputs = getattr(trainer, "_last_predict_round_outputs", None)
            round_prompt_formats = getattr(trainer, "_last_predict_round_prompt_formats", None)

            if round_outputs is not None and len(round_outputs) > 0:
                for ridx, round_output in enumerate(round_outputs, start=1):
                    prompt_format = None
                    if round_prompt_formats is not None and (ridx - 1) < len(round_prompt_formats):
                        prompt_format = round_prompt_formats[ridx - 1]

                    output_data = build_test_output_data_from_predict_output(round_output, test_qids)
                    output_data["round_idx"] = int(ridx)
                    output_data["self_refine_rounds"] = int(ridx)
                    save_test_output_data(
                        output_data,
                        split="test",
                        round_idx=ridx,
                        prompt_format=prompt_format,
                    )
            else:
                output_data = build_test_output_data_from_predict_output(predict_results, test_qids)
                save_test_output_data(output_data, split="test")
    else:
        print("[Self-Refine Continue] Skip normal final test predict and only run continuation.")

    # Optional compatibility path: continue extra self-refine rounds from a saved round file.
    # Defaults keep the original training/test logic unchanged.
    if init_file is not None and extra_rounds > 0:
        print(
            f"[Self-Refine Continue] init_file={init_file}, "
            f"extra_rounds={extra_rounds}"
        )
        continue_results = trainer.predict_from_self_refine_file(
            test_dataset=test_set,
            init_prediction_file=init_file,
            extra_rounds=extra_rounds,
            metric_key_prefix="test",
            max_new_tokens=args.output_len,
        )

        if trainer.is_world_process_zero():
            round_outputs = getattr(trainer, "_last_predict_round_outputs", None)
            round_prompt_formats = getattr(trainer, "_last_predict_round_prompt_formats", None)
            round_global_indices = getattr(trainer, "_last_predict_global_round_indices", None)

            if round_outputs is not None and len(round_outputs) > 0:
                for local_idx, round_output in enumerate(round_outputs, start=1):
                    prompt_format = None
                    if round_prompt_formats is not None and (local_idx - 1) < len(round_prompt_formats):
                        prompt_format = round_prompt_formats[local_idx - 1]

                    global_round_idx = local_idx
                    if round_global_indices is not None and (local_idx - 1) < len(round_global_indices):
                        global_round_idx = int(round_global_indices[local_idx - 1])

                    output_data = build_test_output_data_from_predict_output(round_output, test_qids)
                    output_data["round_idx"] = int(global_round_idx)
                    output_data["self_refine_rounds"] = int(global_round_idx)
                    output_data["continued_from_file"] = init_file
                    output_data["extra_round_idx"] = int(local_idx)
                    save_test_output_data(
                        output_data,
                        split="test",
                        round_idx=global_round_idx,
                        prompt_format=prompt_format,
                    )

        if "continue_results" in locals():
            del continue_results

    if args.prompt_format in ["QCM-LE", "QCM-E"]:
        torch.cuda.empty_cache()
        #del predict_results, preds, targets
        if "predict_results" in locals():
            del predict_results
        if "preds" in locals():
            del preds
        if "targets" in locals():
            del targets
        predict_results = trainer.predict(test_dataset=eval_set, max_new_tokens=args.output_len)
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds, targets = predict_results.predictions, predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds = [p.strip() for p in preds]
            output_data = {"preds": preds, "labels": targets}
            output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))


if __name__ == "__main__":
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    # print("args", args)
    # print("====Input Arguments====")
    # print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if (("qwen" in args.model.lower()) and ("vl" in args.model.lower())):
        args.qwen_vl = True

    if args.qwen_vl:
        problems, qids = load_data_std(args)
        dataframe = {"problems": problems, "qids": qids}
    else:
        if args.img_type is not None:
            problems, qids, name_maps, image_features = load_data_img(args)
            dataframe = {"problems": problems, "qids": qids, "name_maps": name_maps, "image_features": image_features}
        else:
            problems, qids = load_data_std(args)
            dataframe = {"problems": problems, "qids": qids}

    if args.evaluate_dir is None and args.start_epoch >= args.epoch:
        raise ValueError("The start_epoch should be smaller than the total epoch!!!")

    T5Trainer(dataframe=dataframe, args=args)

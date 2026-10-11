import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt_2_12 import *
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps



DATASET_DEFAULT_FILES = {
    "m3cot": "m3cot/multi_cot_merge_235B_V3_R1Q8B_R1_500_500_cleaning_M3CoT.json",
    "aokvqa": "aokvqa/multi_cot_merge_235B_V3_R1Q8B_R1_1000_1000_cleaning_AOKVQA_test_true_answer.json",
    "scienceqa": "scienceqa/multi_cot_merge_235B_V3_R1Q8B_R1_1000_1000_cleaning.json",
    "versacot10k": "versacot10k/multi_cot_merge_235B_V3_R1Q32B_R1_80_80_cleaning_VersaCoT10k.json",
}


def get_dataset_name(args) -> str:
    dataset_name = str(getattr(args, "dataset_name", "m3cot") or "m3cot").lower().strip()
    if dataset_name not in DATASET_DEFAULT_FILES:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name}. "
            f"Supported: {sorted(DATASET_DEFAULT_FILES.keys())}"
        )
    return dataset_name


def resolve_dataset_file_path(args) -> str:
    data_file = getattr(args, "data_file", None)
    if data_file is None or str(data_file).strip() == "":
        data_file = DATASET_DEFAULT_FILES[get_dataset_name(args)]
    data_file = str(data_file)
    if os.path.isabs(data_file):
        return data_file
    return os.path.join(args.data_root, data_file)


def resolve_split_file_path(args) -> str:
    split_file = getattr(args, "split_file", None)
    if split_file is None or str(split_file).strip() == "":
        split_file = os.path.join(get_dataset_name(args), "pid_splits.json")
    split_file = str(split_file)
    if os.path.isabs(split_file):
        return split_file
    return os.path.join(args.data_root, split_file)


img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
    "dinov2_giant": (257, 1536),
    "dinov2_large": (257, 1024),
    "blip_large": (577, 1024),
    "blip2_flant5_xl": (257, 1408),
    "blip2_flant5_xxl": (257, 1408),
    "clip_b16": (197, 768),
    "clip_l14_336": (577, 1024),
    "convnextv2_huge": (2816, 256),
    "sam_base": (256, 4096),
    "sam_large": (256, 4096),
    "sam_huge": (256, 4096),
}



def _build_dual_round_sample_plan(qids, args, split, test_le_data=None):
    if split == "train" and test_le_data is None and getattr(args, "use_txt_enhancement_phase2", False):
        round1_repeat = max(1, int(getattr(args, "round1_repeat", 1)))
        round2_repeat = max(1, int(getattr(args, "round2_repeat", 1)))

        plan = []
        for qid in qids:
            for _ in range(round1_repeat):
                plan.append((qid, 1))
            for _ in range(round2_repeat):
                plan.append((qid, 2))
        return plan

    return [(qid, None) for qid in qids]


def load_data_std(args):
    dataset_name = get_dataset_name(args)
    data_file = resolve_dataset_file_path(args)
    split_file = resolve_split_file_path(args)

    print(f"[Dataset] dataset_name={dataset_name}")
    print(f"[Dataset] data_file={data_file}")
    print(f"[Dataset] split_file={split_file}")

    problems = json.load(open(data_file, "r", encoding="utf-8"))
    pid_splits = json.load(open(split_file, "r", encoding="utf-8"))

    captions = {}
    caption_file = getattr(args, "caption_file", None)
    if caption_file is not None and str(caption_file).strip() != "" and os.path.exists(str(caption_file)):
        caption_obj = json.load(open(caption_file, "r", encoding="utf-8"))
        captions = caption_obj.get("captions", caption_obj)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids,

def load_data_img(args):
    problems = json.load(open(resolve_dataset_file_path(args), "r", encoding="utf-8"))
    pid_splits = json.load(open(resolve_split_file_path(args), "r", encoding="utf-8"))
    caption_file = getattr(args, "caption_file", None)
    captions = {}
    if caption_file is not None and str(caption_file).strip() != "" and os.path.exists(str(caption_file)):
        caption_obj = json.load(open(caption_file, "r", encoding="utf-8"))
        captions = caption_obj.get("captions", caption_obj)
    name_maps = json.load(open('../data/name_map.json'))

    # check
    if args.img_type == "resnet":
        image_features = np.load('../vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('../vision_features/clip.npy')
    elif args.img_type == "detr":
        image_features = np.load('../vision_features/detr.npy')
    elif args.img_type == "vit":
        image_features = torch.load("../vision_features/vit.pth")
    elif args.img_type == "dinov2_giant":
        image_features = torch.load("../vision_features/dinov2_giant.pth")
    elif args.img_type == "dinov2_large":
        image_features = torch.load("../vision_features/dinov2_large.pth")
    elif args.img_type == "blip_large":
        image_features = torch.load("../vision_features/blip_large.pth")
    elif args.img_type == "blip2_flant5_xl":
        image_features = torch.load("../vision_features/blip2_flant5_xl.pth")
    elif args.img_type == "blip2_flant5_xxl":
        image_features = torch.load("../vision_features/blip2_flant5_xxl.pth")
    elif args.img_type == "clip_b16":
        image_features = torch.load("../vision_features/clip_b16.pth")
    elif args.img_type == "clip_l14_336":
        image_features = torch.load("../vision_features/clip_l14_336.pth")
    elif args.img_type == "convnextv2_huge":
        image_features = torch.load("../vision_features/convnextv2_huge.pth")
    elif args.img_type == "sam_base":
        image_features = torch.load("../vision_features/sam_base.pth")
    elif args.img_type == "sam_large":
        image_features = torch.load("../vision_features/sam_large.pth")
    elif args.img_type == "sam_huge":
        image_features = torch.load("../vision_features/sam_huge.pth")
    else:
        image_features = np.load('../vision_features/detr.npy')
    print("img_features size: ", image_features.shape)

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features


class ScienceQADatasetStd(Dataset):
    def __init__(self, problems, qids, tokenizer, source_len, target_len, args, test_le=None, split="train"):
        self.tokenizer = tokenizer
        self.problems = problems
        self.qids = qids
        self.source_len = source_len
        self.target_len = target_len
        self.args = args
        self.split = split

        if test_le is not None:
            self.test_le_data = json.load(open(test_le))["preds"]
        else:
            self.test_le_data = None

        self.sample_plan = _build_dual_round_sample_plan(self.qids, self.args, self.split, self.test_le_data)

    def __len__(self):
        return len(self.sample_plan)

    def __getitem__(self, index):
        qid, forced_refine_round = self.sample_plan[index]
        problem = self.problems[qid]
        curr_le_data = None
        if self.test_le_data is not None:
            curr_le_data = self.test_le_data[index]

        effective_prompt_format = resolve_effective_prompt_format(self.args, split=self.split, curr_le_data=curr_le_data)
        prompt, target = build_train_pair(
            self.problems, qid, self.args, curr_le_data,
            prompt_format=effective_prompt_format, split=self.split,
            forced_refine_round=forced_refine_round,
        )

        source_text = " ".join(prompt.strip().split())
        target_text = " ".join(target.strip().split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }




class ScienceQADatasetImg(Dataset):
    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None, split="train"
    ):
        self.tokenizer = tokenizer
        self.problems = problems
        self.qids = qids
        self.name_maps = name_maps
        self.image_features = image_features
        self.source_len = source_len
        self.target_len = target_len
        self.args = args
        self.split = split
        self.hardsamplemine = args.hardsamplemine
        self.test_le_data = None

        if test_le is not None:
            self.test_le_data = json.load(open(test_le))["preds"]

        self.sample_plan = _build_dual_round_sample_plan(self.qids, self.args, self.split, self.test_le_data)
        self.img_shape = img_shape[args.img_type]
        self.image_cache = self._cache_image_features()

    def _cache_image_features(self):
        image_ids = {}
        for qid in self.qids:
            if str(qid) in self.name_maps:
                vec = self.image_features[int(self.name_maps[str(qid)])]
                image_ids[str(qid)] = torch.tensor(vec).float()
            else:
                image_ids[str(qid)] = torch.zeros(self.img_shape, dtype=torch.float)
        return image_ids

    def __len__(self):
        return len(self.sample_plan)

    def __getitem__(self, index):
        qid, forced_refine_round = self.sample_plan[index]
        problem = self.problems[qid]
        curr_le_data = None
        if self.test_le_data:
            curr_le_data = self.test_le_data[index]
        image_ids = self.image_cache[str(qid)]

        if self.hardsamplemine and curr_le_data is None:
            effective_prompt_format = resolve_effective_prompt_format(self.args, split=self.split, curr_le_data=curr_le_data)
            prompt, target, pos_targets, neg_targets = build_train_pair_new(
                self.problems, qid, self.args, curr_le_data, prompt_format=effective_prompt_format, split=self.split
            )
        else:
            effective_prompt_format = resolve_effective_prompt_format(self.args, split=self.split, curr_le_data=curr_le_data)
            prompt, target = build_train_pair(
                self.problems, qid, self.args, curr_le_data,
                prompt_format=effective_prompt_format, split=self.split,
                forced_refine_round=forced_refine_round,
            )
            pos_targets, neg_targets = [], []

        # Tokenize input prompt
        prompt = " ".join(prompt.strip().split())
        source = self.tokenizer(
            prompt,
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        # Tokenize target
        target = " ".join(target.strip().split())
        tgt = self.tokenizer(
            target,
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_ids = tgt["input_ids"].squeeze().tolist()

        # Hard mining case
        if self.hardsamplemine and curr_le_data is None:
            pos_encodings = self.tokenizer.batch_encode_plus(
                pos_targets,
                max_length=self.target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            neg_encodings = self.tokenizer.batch_encode_plus(
                neg_targets,
                max_length=self.target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            pos_target_ids = pos_encodings["input_ids"].tolist()
            neg_target_ids = neg_encodings["input_ids"].tolist()

            return {
                "input_ids": source_ids,
                "attention_mask": source_mask,
                "image_ids": image_ids,
                "labels": target_ids,
                "pos_labels": pos_target_ids,
                "neg_labels": neg_target_ids,
            }

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }



# ============================================================
# Qwen2.5-VL / Qwen3-VL support (image-text-to-text, chat-template)
#  - Keeps the original evaluation pipeline unchanged:
#    * Dataset returns: input_ids/attention_mask (+vision tensors) and labels as fixed-length token ids
#    * Model wrapper concatenates prompt+labels internally for loss (so labels can stay "target-only")
# ============================================================

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

def _try_join(*parts):
    p = os.path.join(*[str(x) for x in parts if x is not None])
    return p


def resolve_dataset_image_path(problem: dict, data_root: str, dataset_name: str = "m3cot") -> Optional[str]:
    """Resolve image path under {data_root}/{dataset_name}/images/... with robust fallbacks."""
    dataset_name = str(dataset_name or "m3cot").lower().strip()

    img = problem.get("image_path", None)
    if not img:
        img = problem.get("image", None)

    if img is None:
        return None
    img = str(img).strip()
    if img == "" or img.lower() == "null" or img.lower() == "none":
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



class ScienceQADatasetQwenVL(Dataset):
    """
    Return prompt-only inputs (with chat template + generation prompt) and target-only labels
    to keep the original Seq2SeqTrainer + compute_metrics code unchanged.

    Vision tensors are produced by processor.apply_chat_template(..., tokenize=True).
    """
    def __init__(self, problems, qids, processor, source_len, target_len, args, test_le=None, split="train"):
        self.problems = problems
        self.qids = qids
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.args = args
        self.split = split
        self.hardsamplemine = args.hardsamplemine
        self.test_le_data = None
        if test_le is not None:
            self.test_le_data = json.load(open(test_le))["preds"]
        self.sample_plan = _build_dual_round_sample_plan(self.qids, self.args, self.split, self.test_le_data)

    def __len__(self):
        return len(self.sample_plan)

    def _should_apply_image_aug(self) -> bool:
        if self.split != "train":
            return False
        if not bool(getattr(self.args, "use_image_aug", False)):
            return False
        prob = float(getattr(self.args, "image_aug_prob", 0.5))
        prob = max(0.0, min(1.0, prob))
        return random.random() < prob

    def _apply_gamma(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(0.85, 1.15)
        inv_gamma = 1.0 / max(gamma, 1e-6)
        lut = [min(255, max(0, int(((i / 255.0) ** inv_gamma) * 255.0))) for i in range(256)]
        return image.point(lut * 3)

    def _apply_gaussian_noise(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image).astype(np.float32)
        sigma = random.uniform(2.0, 6.0)
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        quality = random.randint(60, 88)
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_resize_down_up(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if w < 32 or h < 32:
            return image
        scale = random.uniform(0.75, 0.95)
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        image = image.resize((w, h), Image.BICUBIC)
        return image

    def _apply_light_blur(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.7)))

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if image is None:
            return image

        ops_done = set()

        photometric_pool = [
            ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.78, 1.28))),
            ("brightness", lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.88, 1.16))),
            ("color", lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.88, 1.15))),
            ("sharpness", lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.88, 1.22))),
            ("autocontrast", lambda img: ImageOps.autocontrast(img)),
            ("gamma", self._apply_gamma),
        ]

        degradation_pool = [
            ("jpeg", self._apply_jpeg_compression),
            ("gaussian_noise", self._apply_gaussian_noise),
            ("resize_down_up", self._apply_resize_down_up),
            ("light_blur", self._apply_light_blur),
        ]

        num_photo = 1 if random.random() < 0.65 else 2
        num_degrade = 1 if random.random() < 0.30 else 0
        if random.random() < 0.08:
            num_degrade = min(1, num_degrade + 1)

        for name, fn in random.sample(photometric_pool, k=min(num_photo, len(photometric_pool))):
            try:
                image = fn(image)
                ops_done.add(name)
            except Exception:
                pass

        if num_degrade > 0:
            for name, fn in random.sample(degradation_pool, k=min(num_degrade, len(degradation_pool))):
                try:
                    image = fn(image)
                    ops_done.add(name)
                except Exception:
                    pass

        if "autocontrast" not in ops_done and random.random() < 0.08:
            try:
                image = ImageOps.autocontrast(image)
            except Exception:
                pass
        elif "sharpness" not in ops_done and random.random() < 0.08:
            try:
                image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.95, 1.10))
            except Exception:
                pass

        return image

    def _qwen_output_constraint(self, prompt_format=None,num_choices=None) -> str:
        """Build strict-format instruction based on your original prompt_format.

        IMPORTANT:
          - We do NOT change your original prompt construction (build_train_pair/build_train_pair_new).
          - We ONLY add a format constraint so Qwen outputs match the target style that your
            evaluation (extract_ans / rouge) already assumes.
        """
        fmt = prompt_format or getattr(self.args, "prompt_format", "QCM-A")
        try:
            _, out_fmt = fmt.split("-", 1)
        except Exception:
            out_fmt = "A"

        options = getattr(self.args, "options", ["A", "B", "C", "D", "E"])
        if num_choices is not None:
            options = options[:int(num_choices)]
        opt_str = ", ".join(options)

        # Stage-1 (rationale) formats: E / LE
        if out_fmt in ["E", "LE"]:
            return (
                "Output FORMAT (strict):\n"
                "- Return ONLY the rationale.\n"
                "- The output MUST start with: Solution: \n"
                "- Do NOT output any answer option letter, and do NOT include 'The answer is'."
            )

        # Stage-2 (answer) formats
        if out_fmt == "A":
            return (
                "Output FORMAT (strict):\n"
                f"- Return EXACTLY one line: The answer is (X). where X is one of {opt_str}.\n"
                "- Do NOT output anything else (no explanation, no extra lines, no prefixes like 'Answer:')."
            )
        if out_fmt == "AE":
            return (
                "Output FORMAT (strict):\n"
                f"- First, output: The answer is (X). where X is one of {opt_str}.\n"
                "- Then, continue in the SAME line or next text with: BECAUSE: <rationale>.\n"
                "- Do NOT add other headings." 
            )
        if out_fmt == "ALE":
            return (
                "Output FORMAT (strict):\n"
                f"- First, output: The answer is (X). where X is one of {opt_str}.\n"
                "- Then, output: BECAUSE: <lecture> <solution>.\n"
                "- Do NOT add other headings."
            )
        if out_fmt == "LEA":
            # Target is: "<lecture> <solution> The answer is (X)." (no 'Solution:' prefix)
            return (
                "Output FORMAT (strict):\n"
                "- First, output the explanation (lecture + solution) as plain text (NO 'Solution:' prefix).\n"
                f"- End with: The answer is (X). where X is one of {opt_str}.\n"
                "- Do NOT add other headings."
            )

        # Fallback: keep evaluation-safe answer rule
        return (
            "Output FORMAT (strict):\n"
            f"- Your output MUST contain 'The answer is (X)' with X in {opt_str}.\n"
            "- Avoid extra prefixes." 
        )


    def _build_user_messages(self, prompt_text: str, image, caption: str, prompt_format=None, num_choices=None) -> List[Dict[str, Any]]:
        """Qwen chat style wrapper (image MUST be a PIL.Image or None)."""
        system_prompt = getattr(self.args, "qwen_system_prompt", None) or "You are a helpful vision-language assistant."
        constraint = self._qwen_output_constraint(prompt_format=prompt_format, num_choices=num_choices)

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})

        caption_mode = getattr(self.args, "qwen_caption_mode", "none")
        if caption and caption_mode in ["segment", "both"]:
            content.append({"type": "text", "text": f"Image caption: {caption}"})

        user_text = f"{prompt_text}\n\n{constraint}"
        content.append({"type": "text", "text": user_text})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def __getitem__(self, index):
        qid, forced_refine_round = self.sample_plan[index]
        problem = self.problems[qid]
        problem = dict(problem)
        problem["qid"] = qid

        curr_le_data = self.test_le_data[index] if self.test_le_data else None
        effective_prompt_format = resolve_effective_prompt_format(
            self.args, split=self.split, curr_le_data=curr_le_data
        )

        if self.hardsamplemine and curr_le_data is None:
            prompt, target, pos_targets, neg_targets = build_train_pair_new(
                self.problems, qid, self.args, curr_le_data,
                prompt_format=effective_prompt_format, split=self.split
            )
        else:
            prompt, target = build_train_pair(
                self.problems, qid, self.args, curr_le_data,
                prompt_format=effective_prompt_format, split=self.split,
                forced_refine_round=forced_refine_round,
            )
            pos_targets, neg_targets = [], []

        prompt = " ".join(prompt.strip().split())
        target = " ".join(target.strip().split())

        caption = problem.get("caption", "") or ""
        if getattr(self.args, "qwen_caption_mode", "none") in ["context", "both"]:
            if caption:
                prompt = f"Image caption: {caption}\n\n{prompt}"

        image_path = resolve_dataset_image_path(
            problem,
            self.args.data_root,
            dataset_name=get_dataset_name(self.args),
        )

        try:
            if image_path is None:
                raise FileNotFoundError("No image path")
            image = Image.open(image_path).convert("RGB")
            if self._should_apply_image_aug():
                image = self._augment_image(image)
        except Exception:
            image = Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))

        messages = self._build_user_messages(
            prompt, image, caption, prompt_format=effective_prompt_format, num_choices=len(problem["choices"])
        )

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.tokenizer.eos_token is not None:
            if not target.strip().endswith(self.tokenizer.eos_token):
                target = target.rstrip() + self.tokenizer.eos_token

        tgt = self.tokenizer(
            target,
            max_length=self.target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"][0].tolist()

        example = {
            "chat_text": chat_text,
            "image": image,
            "labels": labels,
            "qid": str(qid),
        }

        if self.hardsamplemine and curr_le_data is None:
            pos_enc = self.tokenizer.batch_encode_plus(
                pos_targets,
                max_length=self.target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].tolist()

            neg_enc = self.tokenizer.batch_encode_plus(
                neg_targets,
                max_length=self.target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].tolist()

            example["pos_labels"] = pos_enc
            example["neg_labels"] = neg_enc

        return example


from typing import List, Dict, Any, Optional

class QwenVLCollator:
    def __init__(self, processor, tokenizer=None, source_len=None, pad_to_multiple_of=None):
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.source_len = source_len
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch

        texts = [f["chat_text"] for f in features]
        images = [f["image"] for f in features]

        proc_kwargs = {
            "text": texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.source_len is not None:
            proc_kwargs["truncation"] = True
            proc_kwargs["max_length"] = self.source_len

        proc = self.processor(**proc_kwargs)
        proc.pop("token_type_ids", None)

        batch = dict(proc)

        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        if any("qid" in f for f in features):
            batch["qids"] = [str(f.get("qid", "")) for f in features]

        for k in ["pos_labels", "neg_labels"]:
            if any(k in f for f in features):
                batch[k] = [f.get(k, []) for f in features]

        return batch

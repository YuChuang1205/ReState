import argparse
import random

import numpy as np
import torch

from utils_data import load_data_std
from utils_solution_augment_qwenvl import prewarm_solution_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="m3cot", choices=["m3cot", "aokvqa", "scienceqa", "versacot10k"],
                        help="dataset to use under data_root; keeps m3cot as the default")
    parser.add_argument("--data_file", type=str, default=None,
                        help="optional dataset json path; relative paths are resolved under data_root")
    parser.add_argument("--split_file", type=str, default=None,
                        help="optional pid_splits.json path; relative paths are resolved under data_root")
    parser.add_argument("--caption_file", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "trainval", "minitrain"])
    parser.add_argument("--train_split", type=str, default="minitrain", choices=["train", "trainval", "minitrain"])
    parser.add_argument("--val_split", type=str, default="minival", choices=["test", "val", "minival"])
    parser.add_argument("--test_split", type=str, default="minitest", choices=["test", "minitest"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--solution_gen_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--solution_cache_dir", type=str, default="./solution_aug_cache")
    parser.add_argument("--solution_pos_num", type=int, default=6)
    parser.add_argument("--solution_neg_num", type=int, default=6)
    parser.add_argument("--solution_gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--solution_gen_temperature", type=float, default=0.9)
    parser.add_argument("--solution_gen_top_p", type=float, default=0.9)
    parser.add_argument("--solution_gen_do_sample", action="store_true")
    parser.add_argument("--solution_gen_use_image", action="store_true")
    parser.add_argument("--solution_gen_use_caption", action="store_true")
    parser.add_argument("--solution_edit_from_origin", action="store_true", help="rewrite the original solution directly when generating positive/negative banks")
    parser.add_argument("--solution_multi_gen_num", type=int, default=1, help="number of solution candidates to ask for in a single augmentation dialogue")
    parser.add_argument("--solution_gen_device", type=str, default="cuda:0")

    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--options", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    parser.add_argument("--solution_refresh_prob", type=float, default=0.0)
    parser.add_argument("--solution_bank_max_size", type=int, default=10)
    parser.add_argument("--solution_cache_read_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    problems, qids = load_data_std(args)

    if args.split == "trainval":
        use_qids = list(qids["train"]) + list(qids["val"])
    elif args.split in ["train", "minitrain"]:
        use_qids = list(qids["train"])
    elif args.split in ["val", "minival"]:
        use_qids = list(qids["val"])
    elif args.split in ["test", "minitest"]:
        use_qids = list(qids["test"])
    else:
        raise ValueError(f"Unsupported split={args.split}")
    
    print(f"[CacheBuild] split={args.split}, num_qids={len(use_qids)}, cache_dir={args.solution_cache_dir}")
    prewarm_solution_cache(problems, use_qids, args)
    print("[CacheBuild] done")

import argparse
import os
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

from utils_data import load_data_std
from utils_solution_augment_qwenvl import (
    get_solution_augmentor,
    normalize_solution_text,
    parse_tagged_solution_outputs,
    strip_solution_prefix,
)


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
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "trainval", "minitrain", "minival", "minitest"])
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "trainval", "minitrain"])
    parser.add_argument("--val_split", type=str, default="val", choices=["val", "test", "minival"])
    parser.add_argument("--test_split", type=str, default="test", choices=["test", "minitest"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--solution_gen_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--solution_cache_dir", type=str, default="./solution_aug_cache")
    parser.add_argument("--solution_pos_num", type=int, default=6)
    parser.add_argument("--solution_neg_num", type=int, default=6)
    parser.add_argument("--solution_bank_max_size", type=int, default=20)
    parser.add_argument("--solution_gen_max_new_tokens", type=int, default=256)
    parser.add_argument("--solution_gen_temperature", type=float, default=0.9)
    parser.add_argument("--solution_gen_top_p", type=float, default=0.9)
    parser.add_argument("--solution_gen_do_sample", action="store_true")
    parser.add_argument("--solution_gen_use_image", action="store_true")
    parser.add_argument("--solution_gen_use_caption", action="store_true")
    parser.add_argument("--solution_edit_from_origin", action="store_true", help="rewrite the original solution directly when generating positive/negative banks")
    parser.add_argument("--solution_multi_gen_num", type=int, default=1, help="number of solution candidates to ask for in a single augmentation dialogue")
    parser.add_argument("--solution_gen_strict_filter", action="store_true")
    parser.add_argument("--solution_gen_device", type=str, default="cuda:0")
    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--options", nargs="+", default=["A", "B", "C", "D", "E", "F", "G", "H", "I"])

    parser.add_argument("--mode", type=str, default="loop", choices=["once", "loop"])
    parser.add_argument("--num_rounds", type=int, default=0, help="0 means unlimited when --mode loop")
    parser.add_argument("--qid_batch_size", type=int, default=32, help="How many qids to refresh per round")
    parser.add_argument("--gen_batch_size", type=int, default=8, help="How many generation requests to send to the model at once")
    parser.add_argument("--sleep_seconds", type=float, default=0, help="Sleep between rounds in loop mode")
    parser.add_argument("--shuffle_each_round", action="store_true")
    parser.add_argument("--start_from_cache_only", action="store_true", help="Only refresh qids that already have cache files")
    parser.add_argument("--build_missing_bank", action="store_true", help="If cache missing, build a full bank first for that qid")
    parser.add_argument("--refresh_pos_per_qid", type=int, default=1)
    parser.add_argument("--refresh_neg_per_qid", type=int, default=1)
    parser.add_argument("--refresh_new_qid_limit", type=int, default=0, help="If >0, limit number of missing-cache qids to initialize per round")

    # keep writer enabled; training side should be read_only
    parser.add_argument("--solution_cache_read_only", action="store_true")
    return parser.parse_args()


POS_VARIANTS = [
    "logic_reframe",
    "evidence_first",
    "option_elimination",
    "abductive_style",
    "concise_but_complete",
    "step_explicit",
]

NEG_VARIANTS = [
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


class dummy_lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def resolve_use_qids(qids: Dict[str, List[str]], split: str) -> List[str]:
    if split == "trainval":
        return list(qids["train"]) + list(qids["val"])
    if split in ["train", "minitrain"]:
        return list(qids["train"])
    if split in ["val", "minival"]:
        return list(qids["val"])
    if split in ["test", "minitest"]:
        return list(qids["test"])
    raise ValueError(f"Unsupported split={split}")


def _make_bank_stub(aug, qid: str, origin: str) -> Dict[str, Any]:
    return {
        "qid": qid,
        "solution_origin": origin,
        "pos": [],
        "neg": [],
        "meta": {
            "model": aug.model_name,
            "use_image": aug.use_image,
            "use_caption": aug.use_caption,
            "refresh_script": True,
        },
    }


def prepare_refresh_for_qid(aug, problem: Dict[str, Any], qid: str, args) -> Dict[str, Any]:
    qid = str(qid)
    origin = normalize_solution_text(problem.get("solution", ""))
    max_bank_size = int(getattr(args, "solution_bank_max_size", 20))
    desired_pos = max(1, int(getattr(args, "solution_pos_num", 6)))
    desired_neg = max(1, int(getattr(args, "solution_neg_num", 6)))

    bank = aug._read_cache(qid)
    if bank is None:
        if args.start_from_cache_only:
            return {
                "qid": qid,
                "result": {"qid": qid, "status": "skip_no_cache", "appended_pos": 0, "appended_neg": 0, "errors": []},
            }
        if args.build_missing_bank:
            bank = aug._build_bank(problem, qid)
            aug._write_cache(qid, bank)
        else:
            bank = _make_bank_stub(aug, qid, origin)

    old_pos = len(bank.get("pos", []))
    old_neg = len(bank.get("neg", []))

    if origin == "":
        with aug._file_lock(aug._lock_path(qid)) if hasattr(aug, "_file_lock") else dummy_lock():
            latest = aug._read_cache(qid) or bank
            latest.setdefault("pos", [])
            latest.setdefault("neg", [])
            latest.setdefault("meta", {})
            latest["solution_origin"] = origin
            latest["pos"] = [origin for _ in range(min(max_bank_size, desired_pos))]
            latest["neg"] = [origin for _ in range(min(max_bank_size, desired_neg))]
            latest["meta"]["empty_origin_solution"] = True
            latest["meta"]["refresh_script"] = True
            aug._write_cache(qid, latest)

        return {
            "qid": qid,
            "result": {
                "qid": qid,
                "status": "ok_empty_origin",
                "old_pos": old_pos,
                "old_neg": old_neg,
                "new_pos": min(max_bank_size, desired_pos),
                "new_neg": min(max_bank_size, desired_neg),
                "appended_pos": 0,
                "appended_neg": 0,
                "errors": [],
            },
        }

    jobs: List[Dict[str, Any]] = []
    for _ in range(int(args.refresh_pos_per_qid)):
        jobs.append({
            "qid": qid,
            "kind": "pos",
            "task": "positive",
            "variant": random.choice(POS_VARIANTS),
            "problem": problem,
        })
    for _ in range(int(args.refresh_neg_per_qid)):
        jobs.append({
            "qid": qid,
            "kind": "neg",
            "task": "negative",
            "variant": random.choice(NEG_VARIANTS),
            "problem": problem,
        })

    return {
        "qid": qid,
        "problem": problem,
        "origin": origin,
        "bank": bank,
        "old_pos": old_pos,
        "old_neg": old_neg,
        "jobs": jobs,
    }


def _run_single_bucket_generation(aug, jobs: List[Dict[str, Any]], gen_batch_size: int) -> List[Dict[str, Any]]:
    if not jobs:
        return jobs

    for start in range(0, len(jobs), gen_batch_size):
        chunk = jobs[start : start + gen_batch_size]
        prepared = []
        prepared_jobs = []

        for job in chunk:
            try:
                req = aug._prepare_request(job["problem"], task=job["task"], variant=job["variant"])
                prepared.append(req)
                prepared_jobs.append(job)
            except Exception as e:
                job["generated_text"] = None
                job["error"] = f"PREP:{type(e).__name__}:{e}"

        if not prepared_jobs:
            continue

        try:
            if bool(getattr(aug, "edit_from_origin", False)):
                raw_texts = aug._generate_batch_raw(prepared)
                for job, req, raw_text in zip(prepared_jobs, prepared, raw_texts):
                    tag = aug._multi_output_tag(job["task"])
                    texts = parse_tagged_solution_outputs(raw_text, tag=tag, repeat_num=int(req.get("repeat_num", 1)))
                    job["generated_texts"] = texts
                    job["generated_text"] = texts[0] if texts else ""
            else:
                texts = aug._generate_batch(prepared)
                for job, text in zip(prepared_jobs, texts):
                    job["generated_text"] = text
        except Exception as e:
            for job in prepared_jobs:
                job["generated_text"] = None
                job["generated_texts"] = []
                job["error"] = f"BATCH:{type(e).__name__}:{e}"

    return jobs


def run_batch_generation(aug, jobs: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    if not jobs:
        return jobs

    gen_batch_size = max(1, int(getattr(args, "gen_batch_size", 8)))

    # Bucket by task to avoid mixing positive/negative editing objectives in the same batch.
    pos_jobs = [job for job in jobs if job.get("task") == "positive"]
    neg_jobs = [job for job in jobs if job.get("task") == "negative"]
    other_jobs = [job for job in jobs if job.get("task") not in {"positive", "negative"}]

    _run_single_bucket_generation(aug, pos_jobs, gen_batch_size)
    _run_single_bucket_generation(aug, neg_jobs, gen_batch_size)
    _run_single_bucket_generation(aug, other_jobs, gen_batch_size)

    return jobs


def apply_generated_results_for_qid(aug, prep: Dict[str, Any], args) -> Dict[str, Any]:
    qid = prep["qid"]
    origin = prep["origin"]
    bank = prep["bank"]
    max_bank_size = int(getattr(args, "solution_bank_max_size", 20))

    appended_pos = 0
    appended_neg = 0
    errors = []

    with aug._file_lock(aug._lock_path(qid)) if hasattr(aug, "_file_lock") else dummy_lock():
        latest = aug._read_cache(qid) or bank
        latest.setdefault("pos", [])
        latest.setdefault("neg", [])
        latest.setdefault("meta", {})

        for job in prep.get("jobs", []):
            if job.get("error"):
                errors.append(job["error"])
                continue

            candidate_texts = job.get("generated_texts") or [job.get("generated_text", "")]
            candidate_texts = [strip_solution_prefix(x) for x in candidate_texts if strip_solution_prefix(x)]
            if not candidate_texts:
                errors.append(f"{job['kind'].upper()}:EMPTY_OUTPUT")
                continue

            try:
                for text in candidate_texts:
                    if job["kind"] == "pos":
                        ok = aug._positive_ok(text, origin)
                        dup = text in latest["pos"]
                        if ok and not dup:
                            latest["pos"].append(text)
                            appended_pos += 1
                    else:
                        ok = aug._negative_ok(text, origin)
                        dup = text in latest["neg"]
                        if ok and not dup:
                            latest["neg"].append(text)
                            appended_neg += 1
            except Exception as e:
                errors.append(f"{job['kind'].upper()}:{type(e).__name__}:{e}")

        latest["meta"]["refresh_script"] = True
        latest["pos"] = latest["pos"][-max_bank_size:]
        latest["neg"] = latest["neg"][-max_bank_size:]
        aug._write_cache(qid, latest)

    return {
        "qid": qid,
        "status": "ok",
        "old_pos": prep["old_pos"],
        "old_neg": prep["old_neg"],
        "new_pos": len(latest.get("pos", [])),
        "new_neg": len(latest.get("neg", [])),
        "appended_pos": appended_pos,
        "appended_neg": appended_neg,
        "errors": errors,
    }


def main():
    args = parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # Refresh process should never be read-only
    args.solution_cache_read_only = False
    # Use cache mode semantics; this script itself is the writer
    args.solution_gen_mode = "qwen_vl_cache"

    problems, qids = load_data_std(args)
    use_qids = resolve_use_qids(qids, args.split)

    aug = get_solution_augmentor(args)
    if not hasattr(aug, "_file_lock"):
        from utils_solution_augment_qwenvl import _file_lock as module_file_lock
        setattr(aug, "_file_lock", module_file_lock)

    print(
        f"[Refresh] split={args.split} num_qids={len(use_qids)} cache_dir={args.solution_cache_dir} "
        f"mode={args.mode} gen_batch_size={args.gen_batch_size} task_bucketed=True"
    )
    os.makedirs(args.solution_cache_dir, exist_ok=True)

    round_idx = 0
    while True:
        round_idx += 1
        qid_list = list(use_qids)
        if args.shuffle_each_round:
            random.shuffle(qid_list)

        if args.start_from_cache_only:
            qid_list = [qid for qid in qid_list if os.path.exists(os.path.join(args.solution_cache_dir, f"{qid}.json"))]

        if args.refresh_new_qid_limit > 0 and not args.start_from_cache_only:
            cached = []
            missing = []
            for qid in qid_list:
                if os.path.exists(os.path.join(args.solution_cache_dir, f"{qid}.json")):
                    cached.append(qid)
                else:
                    missing.append(qid)
            qid_list = cached + missing[: args.refresh_new_qid_limit]

        qid_batch = qid_list[: args.qid_batch_size] if args.qid_batch_size > 0 else qid_list
        print(f"[Refresh] round={round_idx} refreshing {len(qid_batch)} qids")

        prepared_items = []
        immediate_results = {}
        all_jobs = []

        for qid in qid_batch:
            problem = dict(problems[qid])
            problem["qid"] = qid
            prep = prepare_refresh_for_qid(aug, problem, qid, args)
            if "result" in prep:
                immediate_results[qid] = prep["result"]
            else:
                prepared_items.append(prep)
                all_jobs.extend(prep.get("jobs", []))

        all_jobs = run_batch_generation(aug, all_jobs, args)
        jobs_by_qid = defaultdict(list)
        for job in all_jobs:
            jobs_by_qid[job["qid"]].append(job)

        results_map = dict(immediate_results)
        for prep in prepared_items:
            prep["jobs"] = jobs_by_qid.get(prep["qid"], prep.get("jobs", []))
            results_map[prep["qid"]] = apply_generated_results_for_qid(aug, prep, args)

        appended_pos_total = 0
        appended_neg_total = 0
        error_count = 0

        for qid in qid_batch:
            result = results_map[qid]
            appended_pos_total += int(result.get("appended_pos", 0))
            appended_neg_total += int(result.get("appended_neg", 0))
            if result.get("errors"):
                error_count += 1
                print(f"[Refresh][{qid}] errors={result['errors'][:2]}")
            print(
                f"[Refresh][{qid}] {result['status']} "
                f"pos {result.get('old_pos', '-')}->{result.get('new_pos', '-')} (+{result.get('appended_pos', 0)}) "
                f"neg {result.get('old_neg', '-')}->{result.get('new_neg', '-')} (+{result.get('appended_neg', 0)})"
            )

        print(
            f"[Refresh] round={round_idx} done; appended_pos_total={appended_pos_total}, "
            f"appended_neg_total={appended_neg_total}, error_qids={error_count}"
        )

        if args.mode == "once":
            break
        if args.num_rounds > 0 and round_idx >= args.num_rounds:
            break
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()

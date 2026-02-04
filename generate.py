# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import random
import time
import json
from collections import Counter

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {"prompt": "视频中的人在做动作", "video": "", "pose": "", "mask": ""},
    "s2v-14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "image": "examples/i2v_input.JPG",
        "audio": "examples/talk.wav",
        "tts_prompt_audio": "examples/zero_shot_prompt.wav",
        "tts_prompt_text": "希望你以后能够做的比我还好呦。",
        "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    },
}


def _load_prompts_from_file(path: str):
    """Read prompts line-by-line; ignore blank lines and lines starting with #."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            prompts.append(s)
    return prompts


def _get_gpu_inventory_string():
    """Return e.g. 'NVIDIA RTX A6000x2' if possible, else None."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        c = Counter([x.strip() for x in out if x.strip()])
        if not c:
            return None
        parts = [f"{k}x{v}" for k, v in sorted(c.items(), key=lambda kv: kv[0])]
        return ", ".join(parts)
    except Exception:
        return None


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]

    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]

    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    if "s2v" not in args.task:
        assert args.size in SUPPORTED_SIZES[args.task], (
            f"Unsupport size {args.size} for task {args.task}, "
            f"supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
        )


def _parse_args():
    p = argparse.ArgumentParser(description="Generate an image or video using Wan2.2")
    p.add_argument("--task", type=str, default="t2v-A14B", choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--frame_num", type=int, default=None, help="How many frames (4n+1).")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--offload_model", type=str2bool, default=None)
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--t5_fsdp", action="store_true", default=False)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--dit_fsdp", action="store_true", default=False)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--use_prompt_extend", action="store_true", default=False)
    p.add_argument("--prompt_extend_method", type=str, default="local_qwen", choices=["dashscope", "local_qwen"])
    p.add_argument("--prompt_extend_model", type=str, default=None)
    p.add_argument("--prompt_extend_target_lang", type=str, default="zh", choices=["zh", "en"])
    p.add_argument("--base_seed", type=int, default=-1)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", type=float, default=None)
    p.add_argument("--convert_model_dtype", action="store_true", default=False)

    # animate
    p.add_argument("--src_root_path", type=str, default=None)
    p.add_argument("--refert_num", type=int, default=77)
    p.add_argument("--replace_flag", action="store_true", default=False)
    p.add_argument("--use_relighting_lora", action="store_true", default=False)

    # s2v
    p.add_argument("--num_clip", type=int, default=None)
    p.add_argument("--audio", type=str, default=None)
    p.add_argument("--enable_tts", action="store_true", default=False)
    p.add_argument("--tts_prompt_audio", type=str, default=None)
    p.add_argument("--tts_prompt_text", type=str, default=None)
    p.add_argument("--tts_text", type=str, default=None)
    p.add_argument("--pose_video", type=str, default=None)
    p.add_argument("--start_from_ref", action="store_true", default=False)
    p.add_argument("--infer_frames", type=int, default=80)

    # ===== scy: batch prompt file + outputs =====
    p.add_argument("--prompt_file", type=str, default=None, help="Text file with one prompt per line.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory to save videos/metrics.")
    p.add_argument("--save_videos", type=str2bool, default=True, help="If False, save metrics only (no mp4).")

    args = p.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank: int):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


# ===== scy modi: always barrier with explicit device_ids =====
def _dist_barrier(local_rank: int):
    if dist.is_initialized():
        dist.barrier()


# ===== scy modi: allreduce on the correct local device =====
def _allreduce_max_float(x: float, local_rank: int):
    if not dist.is_initialized():
        return x
    dev = torch.device("cuda", local_rank)
    t = torch.tensor([x], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


# ===== scy modi: seed broadcast via CUDA tensor (avoid broadcast_object_list) =====
def _broadcast_seed_int64(seed: int, rank: int, local_rank: int) -> int:
    if not dist.is_initialized():
        return seed
    dev = torch.device("cuda", local_rank)
    t = torch.tensor([seed if rank == 0 else 0], device=dev, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return int(t.item())


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_id = local_rank  # keep original semantics

    _init_logging(rank)

    # default offload_model behavior
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        if rank == 0:
            logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    # distributed init
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        _ = torch.cuda.current_device()  # scy modi (force context)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        print(f"[rank{rank}] after init_process_group", flush=True)  # scy modi
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp/dit_fsdp need torchrun multi-process."
        assert not (args.ulysses_size > 1), "ulysses_size > 1 needs torchrun multi-process."

    # ulysses group
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal WORLD_SIZE."
        print(f"[rank{rank}] before init_distributed_group", flush=True)  # scy modi
        init_distributed_group()
        print(f"[rank{rank}] after init_distributed_group", flush=True)  # scy modi (moved here)

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` not divisible by `{args.ulysses_size=}`."

    # ===== scy modi: barrier + seed broadcast using CUDA tensor =====
    if dist.is_initialized():
        _dist_barrier(local_rank)  # scy modi
        args.base_seed = _broadcast_seed_int64(args.base_seed, rank, local_rank)  # scy modi
        _dist_barrier(local_rank)  # scy modi

    # prepare optional prompt expander
    prompt_expander = None
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=(args.image is not None),
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=(args.image is not None),
                device=rank,
            )
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    # load image once if used
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        if rank == 0:
            logging.info(f"Input image: {args.image}")

    # output dir
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    # GPU inventory string (rank0 only)
    gpu_inv = _get_gpu_inventory_string() if rank == 0 else None

    # ===== scy: prompt list =====
    if args.prompt_file is not None:
        prompts = _load_prompts_from_file(args.prompt_file)
        if not prompts:
            raise ValueError(f"No valid prompts found in: {args.prompt_file}")
    else:
        prompts = [args.prompt]

    # ===== Build pipeline ONCE =====
    pipeline = None
    if "t2v" in args.task:
        if rank == 0:
            logging.info("Creating WanT2V pipeline.")
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
    elif "ti2v" in args.task:
        if rank == 0:
            logging.info("Creating WanTI2V pipeline.")
        pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
    elif "animate" in args.task:
        if rank == 0:
            logging.info("Creating Wan-Animate pipeline.")
        pipeline = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            use_relighting_lora=args.use_relighting_lora,
        )
    elif "s2v" in args.task:
        if rank == 0:
            logging.info("Creating WanS2V pipeline.")
        pipeline = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
    else:
        if rank == 0:
            logging.info("Creating WanI2V pipeline.")
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device_id,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

    # metrics aggregation (rank0 only)
    per_prompt_results = []

    # ===== MAIN LOOP =====
    for idx, raw_prompt in enumerate(prompts):
        prompt_to_use = raw_prompt

        if args.use_prompt_extend and prompt_expander is not None:
            if rank == 0:
                logging.info("Extending prompt ...")
                prompt_output = prompt_expander(
                    raw_prompt,
                    image=img,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed,
                )
                if prompt_output.status is False:
                    logging.info(f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    prompt_to_use = raw_prompt
                else:
                    prompt_to_use = prompt_output.prompt
            # (kept disabled) prompt broadcast could be added later if needed

        if rank == 0:
            logging.info(f"[{idx+1}/{len(prompts)}] Input prompt: {prompt_to_use}")

        # measure start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(local_rank)  # scy modi
        torch.cuda.synchronize(local_rank)  # scy modi
        start_time = time.perf_counter()

        # generate
        if "t2v" in args.task:
            video = pipeline.generate(
                prompt_to_use,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
        elif "ti2v" in args.task:
            video = pipeline.generate(
                prompt_to_use,
                img=img,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
        elif "animate" in args.task:
            video = pipeline.generate(
                src_root_path=args.src_root_path,
                replace_flag=args.replace_flag,
                refert_num=args.refert_num,
                clip_len=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
        elif "s2v" in args.task:
            video = pipeline.generate(
                input_prompt=prompt_to_use,
                ref_image_path=args.image,
                audio_path=args.audio,
                enable_tts=args.enable_tts,
                tts_prompt_audio=args.tts_prompt_audio,
                tts_prompt_text=args.tts_prompt_text,
                tts_text=args.tts_text,
                num_repeat=args.num_clip,
                pose_video=args.pose_video,
                max_area=MAX_AREA_CONFIGS[args.size],
                infer_frames=args.infer_frames,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                init_first_frame=args.start_from_ref,
            )
        else:
            video = pipeline.generate(
                prompt_to_use,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        # measure end
        torch.cuda.synchronize(local_rank)  # scy modi
        elapsed = time.perf_counter() - start_time
        peak_alloc = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 3)  # scy modi
        peak_reserved = torch.cuda.max_memory_reserved(local_rank) / (1024 ** 3)  # scy modi

        # multi-gpu: take MAX across ranks
        elapsed_max = _allreduce_max_float(elapsed, local_rank)  # scy modi
        peak_alloc_max = _allreduce_max_float(peak_alloc, local_rank)  # scy modi
        peak_reserved_max = _allreduce_max_float(peak_reserved, local_rank)  # scy modi

        # save (rank0 only)
        if rank == 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_slug = prompt_to_use.replace(" ", "_").replace("/", "_")[:50]
            base_name = (
                f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_"
                f"{args.ulysses_size}_p{idx:04d}_{prompt_slug}_{ts}"
            )
            video_path = os.path.join(args.out_dir, base_name + ".mp4")
            metrics_path = os.path.join(args.out_dir, base_name + ".metrics.json")

            result = {
                "index": idx,
                "task": args.task,
                "size": args.size,
                "frame_num": args.frame_num,
                "sample_steps": args.sample_steps,
                "sample_solver": args.sample_solver,
                "sample_shift": args.sample_shift,
                "guide_scale": args.sample_guide_scale,
                "offload_model": args.offload_model,
                "convert_model_dtype": args.convert_model_dtype,
                "world_size": world_size,
                "ulysses_size": args.ulysses_size,
                "gpu": gpu_inv,
                "elapsed_sec": elapsed_max,
                "peak_alloc_gb": peak_alloc_max,
                "peak_reserved_gb": peak_reserved_max,
                "prompt": prompt_to_use,
                "video_path": video_path if args.save_videos else None,
            }

            logging.info("RUN_METRICS_JSON=" + json.dumps(result, ensure_ascii=False))

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            jsonl_path = os.path.join(args.out_dir, "all_metrics.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if args.save_videos:
                logging.info(f"Saving generated video to {video_path}")
                save_video(
                    tensor=video[None],
                    save_file=video_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if "s2v" in args.task:
                    if args.enable_tts is False:
                        merge_video_audio(video_path=video_path, audio_path=args.audio)
                    else:
                        merge_video_audio(video_path=video_path, audio_path="tts.wav")

            per_prompt_results.append(result)

        # cleanup per-iter
        del video
        if dist.is_initialized():
            _dist_barrier(local_rank)  # scy modi

    # ===== summary (rank0 only) =====
    if rank == 0:
        def _avg(key):
            vals = [r[key] for r in per_prompt_results if r.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        summary = {
            "task": args.task,
            "size": args.size,
            "frame_num": args.frame_num,
            "num_prompts": len(per_prompt_results),
            "gpu": gpu_inv,
            "world_size": world_size,
            "ulysses_size": args.ulysses_size,
            "avg_elapsed_sec": _avg("elapsed_sec"),
            "avg_peak_alloc_gb": _avg("peak_alloc_gb"),
            "avg_peak_reserved_gb": _avg("peak_reserved_gb"),
        }
        summary_path = os.path.join(args.out_dir, "summary_avg_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info("SUMMARY_AVG_JSON=" + json.dumps(summary, ensure_ascii=False))

    # teardown
    torch.cuda.synchronize(local_rank)  # scy modi
    if dist.is_initialized():
        _dist_barrier(local_rank)  # scy modi
        dist.destroy_process_group()

    if rank == 0:
        logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)

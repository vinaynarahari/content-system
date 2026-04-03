import argparse
import functools
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import warnings
from collections import Counter

# --- CONFIG ---
MIN_CLIP_DURATION = 2     # Minimum seconds per individual clip
MAX_CLIP_DURATION = 3     # Maximum seconds per individual clip
MIN_TAIL_CLIP_DURATION = 0.75
SCENE_THRESHOLD   = 0.25  # 0.0–1.0 — lower = more sensitive to scene changes
EXTENSIONS        = ('.mov', '.mp4', '.m4v', '.avi', '.mkv')
DEFAULT_SOURCE_DIRS = ["/Users/vinaynarahari/B-Roll"]
COMPLETED_DIR     = "/Users/vinaynarahari/B-Roll/Completed"
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_PATH = os.path.join(SCRIPT_DIR, ".broll_index.json")
DEFAULT_MODEL_CACHE_DIR = os.path.join(SCRIPT_DIR, ".model_cache")
INDEX_VERSION = 5
DEFAULT_MIN_SOURCE_VIDEOS = 10
DEFAULT_MAX_SOURCE_VIDEOS = 15
DEFAULT_ANALYSIS_SAMPLE_VIDEOS = 24
DEFAULT_STYLE = "builder"
DEFAULT_SCENIC_SHARE = 0.0
DEFAULT_TAGGER = "model"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CLIP_CATEGORY_PROMPTS = {
    "builder_realworld": [
        "hands typing on a keyboard at a desk",
        "a person working on a laptop in a real workspace",
        "a person writing notes or planning work at a desk",
        "a person setting up camera gear or a microphone in a workspace",
        "a close-up of desk work and tools",
    ],
    "proof_screen": [
        "a screen recording of analytics or a dashboard",
        "code or charts on a computer screen",
        "a social media app or feed on a phone screen",
        "an app interface or dashboard filling the frame",
        "a screen with comments, metrics, or UI",
    ],
    "scenic": [
        "a scenic outdoor city b-roll shot",
        "a beautiful travel or landscape scene",
        "an atmospheric street or skyline shot",
        "an outdoor road or nature view",
    ],
    "talking_head": [
        "a person talking directly to the camera indoors",
        "a portrait video of a person speaking to camera",
        "a selfie style talking head video",
    ],
    "people_lifestyle": [
        "a person walking or moving through a space",
        "people hanging out or doing everyday lifestyle activities",
        "a casual real-life moment with people",
    ],
}
GROUNDING_LABELS = [
    "keyboard",
    "laptop",
    "monitor",
    "mouse",
    "camera",
    "microphone",
    "tripod",
    "whiteboard",
    "notebook",
    "phone",
    "desk",
    "person",
    "tree",
    "road",
    "sky",
    "building",
    "car",
    "water",
    "mountain",
]
BUILDER_LABELS = {
    "keyboard",
    "laptop",
    "mouse",
    "camera",
    "microphone",
    "tripod",
    "whiteboard",
    "notebook",
    "desk",
}
PROOF_LABELS = {
    "monitor",
    "phone",
}
SCENIC_LABELS = {
    "tree",
    "road",
    "sky",
    "building",
    "car",
    "water",
    "mountain",
}
RNG = random.SystemRandom()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message=r"The key `labels` is will return integer ids.*",
    category=FutureWarning,
)


# ─────────────────────────────────────────────
# AUTO-NUMBER OUTPUT FILE
# ─────────────────────────────────────────────
def get_next_output_path():
    existing = [
        f for f in os.listdir(COMPLETED_DIR)
        if f.startswith("broll_") and f.endswith(".mov")
    ]
    numbers = []
    for f in existing:
        try:
            numbers.append(int(f.replace("broll_", "").replace(".mov", "")))
        except ValueError:
            pass
    next_num = max(numbers, default=0) + 1
    return os.path.join(COMPLETED_DIR, f"broll_{next_num:03d}.mov")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a B-roll montage from one or more source folders.",
    )
    parser.add_argument(
        "target_seconds",
        nargs="?",
        type=float,
        help="Target duration for the final montage in seconds.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help=(
            "Source directory to scan for footage. "
            "Pass multiple times to combine folders, for example the local B-Roll folder and an SD card."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan source directories recursively.",
    )
    parser.add_argument(
        "--style",
        choices=["builder", "proof", "balanced", "scenic", "random"],
        default=DEFAULT_STYLE,
        help=(
            "How to rank source videos before clip extraction. "
            "builder favors real-world work footage, proof favors screens/dashboards, "
            "scenic favors beauty shots, balanced mixes them, and random ignores ranking."
        ),
    )
    parser.add_argument(
        "--tagger",
        choices=["model", "heuristic"],
        default=DEFAULT_TAGGER,
        help=(
            "Use the model-backed classifier or the old heuristic fallback. "
            "Defaults to model."
        ),
    )
    parser.add_argument(
        "--scenic-share",
        type=float,
        default=DEFAULT_SCENIC_SHARE,
        help=(
            "In builder mode, reserve this share of source videos for scenic footage. "
            f"Defaults to {DEFAULT_SCENIC_SHARE:.2f}."
        ),
    )
    parser.add_argument(
        "--analysis-sample-videos",
        type=int,
        default=DEFAULT_ANALYSIS_SAMPLE_VIDEOS,
        help=(
            "How many uncached videos to analyze per run before ranking. "
            f"Defaults to {DEFAULT_ANALYSIS_SAMPLE_VIDEOS}."
        ),
    )
    parser.add_argument(
        "--index-all",
        action="store_true",
        help=(
            "Analyze every uncached source video before selection so styled runs "
            "can rank against the full library."
        ),
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Build or refresh the local analysis cache and exit without exporting a montage."
        ),
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Path for the local B-roll analysis cache.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=DEFAULT_MODEL_CACHE_DIR,
        help="Directory that stores the local open-source tagging models.",
    )
    parser.add_argument(
        "--min-source-videos",
        type=int,
        default=DEFAULT_MIN_SOURCE_VIDEOS,
        help=(
            "Minimum number of random source videos to sample per run. "
            f"Defaults to {DEFAULT_MIN_SOURCE_VIDEOS}."
        ),
    )
    parser.add_argument(
        "--max-source-videos",
        type=int,
        default=DEFAULT_MAX_SOURCE_VIDEOS,
        help=(
            "Maximum number of random source videos to sample per run. "
            f"Defaults to {DEFAULT_MAX_SOURCE_VIDEOS}."
        ),
    )
    return parser.parse_args()


def list_source_videos(source_dirs, recursive):
    videos = []
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"  ⚠️  Source folder not found, skipping: {source_dir}")
            continue

        if recursive:
            walker = os.walk(source_dir)
            for root, _, files in walker:
                for filename in files:
                    lower = filename.lower()
                    if filename.startswith(".") or filename.startswith("._"):
                        continue
                    if lower.endswith(EXTENSIONS):
                        videos.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(source_dir):
                lower = filename.lower()
                if filename.startswith(".") or filename.startswith("._"):
                    continue
                if lower.endswith(EXTENSIONS):
                    videos.append(os.path.join(source_dir, filename))

    # Preserve order while removing duplicates.
    unique_videos = []
    seen = set()
    for path in videos:
        normalized = os.path.realpath(path)
        if normalized in seen:
            continue
        unique_videos.append(path)
        seen.add(normalized)
    return unique_videos


def choose_random_source_videos(videos, min_count, max_count):
    if not videos:
        return []

    if min_count < 1 or max_count < 1:
        raise ValueError("Source video sample size must be at least 1.")
    if min_count > max_count:
        raise ValueError("--min-source-videos cannot be greater than --max-source-videos.")

    upper_bound = min(max_count, len(videos))
    lower_bound = min(min_count, upper_bound)

    if lower_bound == upper_bound:
        sample_size = upper_bound
    else:
        sample_size = RNG.randint(lower_bound, upper_bound)

    return RNG.sample(videos, sample_size)


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def qualitative_level(score):
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def load_analysis_cache(cache_path):
    if not os.path.exists(cache_path):
        return {"version": INDEX_VERSION, "videos": {}}

    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {"version": INDEX_VERSION, "videos": {}}

    if payload.get("version") != INDEX_VERSION or not isinstance(payload.get("videos"), dict):
        return {"version": INDEX_VERSION, "videos": {}}
    return payload


def save_analysis_cache(cache_path, cache):
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    temp_path = f"{cache_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    os.replace(temp_path, cache_path)


def get_video_signature(path):
    stat = os.stat(path)
    return {
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "path": os.path.realpath(path),
    }


def get_cached_analysis(cache, path):
    signature = get_video_signature(path)
    cached = cache["videos"].get(signature["path"])
    if not cached:
        return None
    if cached.get("size") != signature["size"] or cached.get("mtime") != signature["mtime"]:
        return None
    analysis = cached.get("analysis")
    if not isinstance(analysis, dict):
        return None
    return analysis


def cache_video_analysis(cache, path, analysis):
    signature = get_video_signature(path)
    cache["videos"][signature["path"]] = {
        "size": signature["size"],
        "mtime": signature["mtime"],
        "analysis": analysis,
    }


def compute_colorfulness(frame, cv2):
    (blue, green, red) = cv2.split(frame.astype("float32"))
    rg = red - green
    yb = (0.5 * (red + green)) - blue
    rg_std, rg_mean = rg.std(), abs(rg.mean())
    yb_std, yb_mean = yb.std(), abs(yb.mean())
    return math.sqrt((rg_std ** 2) + (yb_std ** 2)) + (0.3 * math.sqrt((rg_mean ** 2) + (yb_mean ** 2)))


def compute_nature_fraction(hsv_frame):
    hue = hsv_frame[:, :, 0]
    sat = hsv_frame[:, :, 1]
    val = hsv_frame[:, :, 2]
    mask = (hue >= 18) & (hue <= 70) & (sat >= 60) & (val >= 50)
    return float(mask.mean())


def load_face_cascade(cv2):
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {cascade_path}")
    return cascade


def classify_primary_type(analysis):
    if analysis["proof_screen_score"] >= max(analysis["builder_realworld_score"] + 0.10, 0.58):
        return "proof_screen"
    if (
        analysis["builder_realworld_score"] >= 0.44
        and analysis_score(analysis, "builder_label_score") >= 0.50
        and analysis["proof_screen_score"] <= 0.40
        and analysis["scenic_score"] <= 0.55
    ):
        return "builder_realworld"
    if analysis["builder_realworld_score"] >= max(
        analysis["proof_screen_score"] + 0.08,
        analysis["scenic_score"] + 0.05,
        0.50,
    ):
        return "builder_realworld"
    if (
        analysis["scenic_score"] >= 0.46
        and analysis_score(analysis, "scenic_label_score") >= 0.34
        and analysis["proof_screen_score"] <= 0.38
    ):
        return "scenic"
    if analysis["scenic_score"] >= max(analysis["builder_realworld_score"] + 0.08, 0.52):
        return "scenic"
    if analysis["talking_head_score"] >= 0.55:
        return "talking_head"
    if analysis["people_lifestyle_score"] >= 0.45:
        return "people"
    return "mixed"


def resolve_torch_device(torch):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_hf_snapshot_path(cache_dir, model_id):
    repo_name = model_id.replace("/", "--")
    repo_dir = os.path.join(cache_dir, "hub", f"models--{repo_name}")
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    refs_main = os.path.join(repo_dir, "refs", "main")

    if os.path.exists(refs_main):
        with open(refs_main, "r", encoding="utf-8") as handle:
            revision = handle.read().strip()
        snapshot_path = os.path.join(snapshots_dir, revision)
        if os.path.isdir(snapshot_path):
            return snapshot_path

    if os.path.isdir(snapshots_dir):
        candidates = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if candidates:
            return sorted(candidates)[-1]

    raise FileNotFoundError(
        f"Missing local model snapshot for {model_id} in {cache_dir}. "
        "Download the model first or point --model-cache-dir at the populated cache."
    )


@functools.lru_cache(maxsize=4)
def load_clip_bundle(cache_dir):
    import torch
    from transformers import AutoProcessor, CLIPModel

    device = resolve_torch_device(torch)
    snapshot_path = resolve_hf_snapshot_path(cache_dir, CLIP_MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        snapshot_path,
        local_files_only=True,
    )
    model = CLIPModel.from_pretrained(
        snapshot_path,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model, device


@functools.lru_cache(maxsize=4)
def load_grounding_bundle(cache_dir):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cpu"
    snapshot_path = resolve_hf_snapshot_path(cache_dir, GROUNDING_MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        snapshot_path,
        local_files_only=True,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        snapshot_path,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model, device


def sample_video_frames(filepath, total_dur, sample_count=4):
    try:
        import cv2
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("OpenCV and PIL are required for video sampling.") from exc

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError("could not open video")

    edge_padding = min(max(total_dur * 0.08, 0.35), 1.5)
    start = edge_padding
    end = max(start + 0.01, total_dur - edge_padding)
    if sample_count == 1:
        sample_times = [total_dur * 0.5]
    else:
        sample_times = [
            start + ((end - start) * index / (sample_count - 1))
            for index in range(sample_count)
        ]

    brightness_values = []
    saturation_values = []
    edge_values = []
    colorfulness_values = []
    nature_values = []
    motion_values = []
    previous_gray = None
    processed_samples = 0
    pil_images = []

    try:
        for sample_time in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            max_dimension = max(height, width)
            if max_dimension > 480:
                scale = 480.0 / max_dimension
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 90, 180)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            brightness_values.append(float(gray.mean() / 255.0))
            saturation_values.append(float(hsv[:, :, 1].mean() / 255.0))
            edge_values.append(float((edges > 0).mean()))
            colorfulness_values.append(float(compute_colorfulness(frame, cv2)))
            nature_values.append(compute_nature_fraction(hsv))
            pil_images.append(Image.fromarray(rgb_frame))

            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                motion_values.append(float(diff.mean() / 255.0))
            previous_gray = gray
            processed_samples += 1
    finally:
        cap.release()

    if processed_samples == 0:
        raise RuntimeError("no frames could be sampled")

    metrics = {
        "brightness_score": round(clamp(statistics.median(brightness_values)), 4),
        "saturation_score": round(clamp(statistics.median(saturation_values)), 4),
        "edge_score": round(clamp(statistics.median(edge_values) * 5.0), 4),
        "colorfulness_score": round(clamp(statistics.median(colorfulness_values) / 80.0), 4),
        "nature_score": round(clamp(statistics.median(nature_values) * 2.4), 4),
        "movement_score": round(
            clamp((statistics.median(motion_values) if motion_values else 0.0) * 14.0),
            4,
        ),
        "sample_count": int(processed_samples),
    }
    metrics["movement"] = qualitative_level(metrics["movement_score"])
    metrics["energy_score"] = round(
        clamp(
            (metrics["movement_score"] * 0.72)
            + (metrics["edge_score"] * 0.18)
            + (metrics["colorfulness_score"] * 0.10)
        ),
        4,
    )
    metrics["energy"] = qualitative_level(metrics["energy_score"])
    return pil_images, metrics


def compute_clip_category_scores(images, cache_dir):
    import torch

    processor, model, device = load_clip_bundle(cache_dir)
    category_names = list(CLIP_CATEGORY_PROMPTS.keys())
    prompt_texts = []
    prompt_slices = {}
    cursor = 0
    for name in category_names:
        prompts = CLIP_CATEGORY_PROMPTS[name]
        prompt_texts.extend(prompts)
        prompt_slices[name] = slice(cursor, cursor + len(prompts))
        cursor += len(prompts)

    inputs = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=-1).detach().cpu()

    category_scores = {}
    for name in category_names:
        category_probabilities = probabilities[:, prompt_slices[name]]
        per_frame = category_probabilities.max(dim=1).values
        category_scores[name] = round(float(per_frame.mean().item()), 4)
    return category_scores


def run_grounding_dino(images, cache_dir, max_images=2):
    import torch

    images = images[:max_images]
    processor, model, device = load_grounding_bundle(cache_dir)
    prompt_text = " . ".join(GROUNDING_LABELS) + " ."
    inputs = processor(
        images=images,
        text=[prompt_text] * len(images),
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        threshold=0.25,
        text_threshold=0.22,
        target_sizes=[image.size[::-1] for image in images],
    )

    label_scores = {label: 0.0 for label in GROUNDING_LABELS}
    label_hits = Counter()

    for result in results:
        output_labels = result.get("text_labels", result["labels"])
        for label, score in zip(output_labels, result["scores"]):
            label_name = str(label).strip().lower()
            if label_name not in label_scores:
                continue
            confidence = float(score)
            label_scores[label_name] = max(label_scores[label_name], confidence)
            label_hits[label_name] += 1

    return label_scores, label_hits


def analyze_video_model_backed(filepath, model_cache_dir):
    total_dur = get_duration(filepath)
    images, metrics = sample_video_frames(filepath, total_dur, sample_count=4)
    clip_scores = compute_clip_category_scores(images, model_cache_dir)
    object_scores, object_hits = run_grounding_dino(images, model_cache_dir, max_images=2)

    filename = os.path.basename(filepath).lower()
    filename_screen_bonus = 0.0
    if "screenrecording" in filename or filename.startswith("screen") or "recording" in filename:
        filename_screen_bonus = 0.85

    builder_object_score = clamp(
        (object_scores["keyboard"] * 0.30)
        + (object_scores["laptop"] * 0.22)
        + (object_scores["mouse"] * 0.08)
        + (object_scores["camera"] * 0.10)
        + (object_scores["microphone"] * 0.10)
        + (object_scores["tripod"] * 0.10)
        + (object_scores["whiteboard"] * 0.10)
        + (object_scores["notebook"] * 0.08)
        + (object_scores["desk"] * 0.05)
    )
    scenic_object_score = clamp(
        (object_scores["tree"] * 0.20)
        + (object_scores["road"] * 0.14)
        + (object_scores["sky"] * 0.18)
        + (object_scores["building"] * 0.14)
        + (object_scores["car"] * 0.08)
        + (object_scores["water"] * 0.18)
        + (object_scores["mountain"] * 0.20)
    )
    person_score = clamp(
        (object_scores["person"] * 0.78)
        + (0.12 if object_hits["person"] >= 2 else 0.0)
    )
    detected_labels = [
        label for label, score in object_scores.items() if score >= 0.28
    ]
    builder_label_score = clamp(
        sum(1 for label in detected_labels if label in BUILDER_LABELS) / 2.0
    )
    proof_label_score = clamp(
        sum(1 for label in detected_labels if label in PROOF_LABELS) / 2.0
    )
    scenic_label_score = clamp(
        sum(1 for label in detected_labels if label in SCENIC_LABELS) / 2.0
    )

    if object_hits["person"] >= 2 or object_scores["person"] >= 0.75:
        people_mode = "group"
    elif object_scores["person"] >= 0.25:
        people_mode = "solo"
    else:
        people_mode = "none"

    proof_screen_score = clamp(
        filename_screen_bonus
        + (clip_scores["proof_screen"] * 0.72)
        + (object_scores["monitor"] * 0.15)
        + (object_scores["phone"] * 0.15)
        + (proof_label_score * 0.14)
        + (metrics["edge_score"] * 0.06)
    )
    if proof_screen_score >= 0.75 and filename_screen_bonus > 0.0:
        people_mode = "none"
        person_score = 0.0

    builder_realworld_score = clamp(
        (clip_scores["builder_realworld"] * 0.42)
        + (builder_object_score * 0.24)
        + (builder_label_score * 0.24)
        + (person_score * 0.07)
        + (metrics["movement_score"] * 0.05)
        + (metrics["edge_score"] * 0.04)
        - (proof_screen_score * 0.20)
        - (clip_scores["scenic"] * 0.06)
    )
    scenic_score = clamp(
        (clip_scores["scenic"] * 0.46)
        + (scenic_object_score * 0.18)
        + (scenic_label_score * 0.18)
        + (metrics["colorfulness_score"] * 0.10)
        + (metrics["nature_score"] * 0.10)
        + (0.04 if people_mode == "none" else 0.0)
        - (proof_screen_score * 0.18)
        - (builder_object_score * 0.12)
    )
    talking_head_score = clamp(
        (clip_scores["talking_head"] * 0.70)
        + (person_score * 0.18)
        - (builder_object_score * 0.14)
        - (proof_screen_score * 0.12)
    )
    people_lifestyle_score = clamp(
        (clip_scores["people_lifestyle"] * 0.62)
        + (person_score * 0.18)
        + (metrics["movement_score"] * 0.06)
        - (proof_screen_score * 0.10)
    )

    analysis = {
        "builder_realworld_score": round(builder_realworld_score, 4),
        "proof_screen_score": round(proof_screen_score, 4),
        "scenic_score": round(scenic_score, 4),
        "talking_head_score": round(talking_head_score, 4),
        "people_lifestyle_score": round(people_lifestyle_score, 4),
        "builder_object_score": round(builder_object_score, 4),
        "builder_label_score": round(builder_label_score, 4),
        "proof_label_score": round(proof_label_score, 4),
        "scenic_object_score": round(scenic_object_score, 4),
        "scenic_label_score": round(scenic_label_score, 4),
        "person_score": round(person_score, 4),
        "people_mode": people_mode,
        "movement_score": metrics["movement_score"],
        "movement": metrics["movement"],
        "energy_score": metrics["energy_score"],
        "energy": metrics["energy"],
        "brightness_score": metrics["brightness_score"],
        "colorfulness_score": metrics["colorfulness_score"],
        "edge_score": metrics["edge_score"],
        "nature_score": metrics["nature_score"],
        "sample_count": metrics["sample_count"],
        "screen_score": round(proof_screen_score, 4),
        "work_score": round(builder_realworld_score, 4),
        "clip_scores": clip_scores,
        "detected_labels": detected_labels,
        "model_stack": "clip+grounding_dino",
    }
    analysis["primary_type"] = classify_primary_type(analysis)
    return analysis


def analyze_video_heuristic(filepath):
    try:
        import cv2
        import numpy as np  # noqa: F401
    except Exception as exc:
        raise RuntimeError("OpenCV and NumPy are required for heuristic B-roll analysis.") from exc

    total_dur = get_duration(filepath)
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError("could not open video")

    face_cascade = load_face_cascade(cv2)
    sample_count = 6 if total_dur <= 30 else 8
    edge_padding = min(max(total_dur * 0.08, 0.35), 1.5)
    start = edge_padding
    end = max(start + 0.01, total_dur - edge_padding)
    if sample_count == 1:
        sample_times = [total_dur * 0.5]
    else:
        sample_times = [
            start + ((end - start) * index / (sample_count - 1))
            for index in range(sample_count)
        ]

    brightness_values = []
    saturation_values = []
    edge_values = []
    colorfulness_values = []
    nature_values = []
    face_counts = []
    motion_values = []
    previous_gray = None
    processed_samples = 0

    try:
        for sample_time in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            max_dimension = max(height, width)
            if max_dimension > 480:
                scale = 480.0 / max_dimension
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 90, 180)
            faces = face_cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(36, frame.shape[1] // 8), max(36, frame.shape[1] // 8)),
            )

            brightness_values.append(float(gray.mean() / 255.0))
            saturation_values.append(float(hsv[:, :, 1].mean() / 255.0))
            edge_values.append(float((edges > 0).mean()))
            colorfulness_values.append(float(compute_colorfulness(frame, cv2)))
            nature_values.append(compute_nature_fraction(hsv))
            face_counts.append(int(len(faces)))

            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                motion_values.append(float(diff.mean() / 255.0))
            previous_gray = gray
            processed_samples += 1
    finally:
        cap.release()

    if processed_samples == 0:
        raise RuntimeError("no frames could be sampled")

    filename = os.path.basename(filepath).lower()
    filename_screen_bonus = 0.0
    if "screenrecording" in filename or filename.startswith("screen") or "recording" in filename:
        filename_screen_bonus = 0.7

    brightness_score = clamp(statistics.median(brightness_values))
    saturation_score = clamp(statistics.median(saturation_values))
    edge_score = clamp(statistics.median(edge_values) * 5.0)
    colorfulness_score = clamp(statistics.median(colorfulness_values) / 80.0)
    nature_score = clamp(statistics.median(nature_values) * 2.4)
    movement_score = clamp((statistics.median(motion_values) if motion_values else 0.0) * 14.0)
    face_presence = clamp(sum(1 for count in face_counts if count > 0) / processed_samples)
    max_faces = max(face_counts) if face_counts else 0

    if max_faces >= 2:
        people_mode = "group"
    elif max_faces == 1:
        people_mode = "solo"
    else:
        people_mode = "none"

    people_score = clamp(face_presence * 0.75 + (0.25 if people_mode != "none" else 0.0))
    screen_score = clamp(
        filename_screen_bonus
        + (edge_score * 0.45)
        + (0.18 if face_presence < 0.15 else 0.0)
        + (0.12 if saturation_score < 0.25 else 0.0)
        + (0.08 if movement_score < 0.35 else 0.0)
    )
    if screen_score >= 0.75 and face_presence <= 0.5:
        people_mode = "none"
        people_score = 0.0
        max_faces = 0
        face_presence = 0.0
    scenic_score = clamp(
        (nature_score * 0.40)
        + (colorfulness_score * 0.25)
        + (movement_score * 0.10)
        + (0.18 if people_mode == "none" else 0.0)
        + (0.10 if brightness_score > 0.48 else 0.0)
        - (screen_score * 0.35)
    )
    work_score = clamp(
        (screen_score * 0.40)
        + (people_score * 0.20)
        + (movement_score * 0.16)
        + (edge_score * 0.14)
        + (0.08 if brightness_score < 0.72 else 0.0)
        - (scenic_score * 0.10)
    )
    energy_score = clamp(
        (movement_score * 0.62)
        + (edge_score * 0.14)
        + (screen_score * 0.12)
        + (0.08 if people_mode == "group" else 0.0)
    )

    analysis = {
        "movement_score": round(movement_score, 4),
        "movement": qualitative_level(movement_score),
        "energy_score": round(energy_score, 4),
        "energy": qualitative_level(energy_score),
        "brightness_score": round(brightness_score, 4),
        "colorfulness_score": round(colorfulness_score, 4),
        "edge_score": round(edge_score, 4),
        "nature_score": round(nature_score, 4),
        "face_presence": round(face_presence, 4),
        "max_faces": int(max_faces),
        "people_mode": people_mode,
        "screen_score": round(screen_score, 4),
        "work_score": round(work_score, 4),
        "scenic_score": round(scenic_score, 4),
        "sample_count": int(processed_samples),
    }
    analysis["builder_realworld_score"] = round(work_score, 4)
    analysis["proof_screen_score"] = round(screen_score, 4)
    analysis["talking_head_score"] = round(people_score * 0.55, 4)
    analysis["people_lifestyle_score"] = round(people_score, 4)
    analysis["builder_object_score"] = 0.0
    analysis["builder_label_score"] = 0.0
    analysis["proof_label_score"] = 0.0
    analysis["scenic_object_score"] = 0.0
    analysis["scenic_label_score"] = 0.0
    analysis["person_score"] = round(people_score, 4)
    analysis["detected_labels"] = []
    analysis["model_stack"] = "heuristic"
    analysis["primary_type"] = classify_primary_type(analysis)
    return analysis


def analyze_video(filepath, tagger, model_cache_dir):
    if tagger == "heuristic":
        return analyze_video_heuristic(filepath)
    return analyze_video_model_backed(filepath, model_cache_dir)


def analysis_score(analysis, key):
    return float(analysis.get(key, 0.0))


def is_builder_candidate(analysis, relaxed=False):
    builder = analysis_score(analysis, "builder_realworld_score")
    proof = analysis_score(analysis, "proof_screen_score")
    scenic = analysis_score(analysis, "scenic_score")
    builder_objects = analysis_score(analysis, "builder_object_score")
    builder_labels = analysis_score(analysis, "builder_label_score")
    proof_labels = analysis_score(analysis, "proof_label_score")

    if analysis.get("primary_type") == "builder_realworld":
        return True
    if relaxed:
        return (
            builder >= 0.16
            and (builder_objects >= 0.05 or builder_labels >= 0.50)
            and proof <= 0.22
            and proof_labels <= 0.15
            and scenic <= 0.42
        )
    return (
        builder >= 0.28
        and (builder_objects >= 0.08 or builder_labels >= 0.50)
        and proof <= 0.18
        and proof_labels <= 0.10
        and scenic <= 0.36
    )


def is_proof_candidate(analysis, relaxed=False):
    proof = analysis_score(analysis, "proof_screen_score")
    builder = analysis_score(analysis, "builder_realworld_score")
    scenic = analysis_score(analysis, "scenic_score")
    proof_labels = analysis_score(analysis, "proof_label_score")

    if analysis.get("primary_type") == "proof_screen":
        return True
    if relaxed:
        return proof >= 0.42 and (proof_labels >= 0.20 or proof >= 0.58) and scenic <= 0.45 and builder <= 0.60
    return proof >= 0.54 and (proof_labels >= 0.34 or proof >= 0.70) and scenic <= 0.34 and builder <= 0.48


def is_scenic_candidate(analysis, relaxed=False):
    scenic = analysis_score(analysis, "scenic_score")
    proof = analysis_score(analysis, "proof_screen_score")
    builder = analysis_score(analysis, "builder_realworld_score")
    scenic_objects = analysis_score(analysis, "scenic_object_score")
    scenic_labels = analysis_score(analysis, "scenic_label_score")

    if analysis.get("primary_type") == "scenic":
        return True
    if relaxed:
        return scenic >= 0.36 and (scenic_objects >= 0.10 or scenic_labels >= 0.34) and proof <= 0.46 and builder <= 0.54
    return (
        scenic >= 0.50
        and (scenic_objects >= 0.10 or scenic_labels >= 0.50)
        and proof <= 0.34
        and builder <= 0.46
    )


def format_analysis_summary(analysis):
    labels = ", ".join(analysis.get("detected_labels", [])[:4]) or "none"
    return (
        f"{analysis.get('primary_type', 'mixed')} | "
        f"builder={analysis_score(analysis, 'builder_realworld_score'):.2f} "
        f"proof={analysis_score(analysis, 'proof_screen_score'):.2f} "
        f"scenic={analysis_score(analysis, 'scenic_score'):.2f} "
        f"move={analysis.get('movement', 'low')} "
        f"people={analysis.get('people_mode', 'none')} "
        f"labels={labels}"
    )


def builder_priority_score(analysis):
    solo_bonus = 0.05 if analysis.get("people_mode") == "solo" else 0.0
    primary_bonus = 0.10 if analysis.get("primary_type") == "builder_realworld" else 0.0
    quiet_work_bonus = 0.07 if (
        analysis_score(analysis, "builder_label_score") >= 0.50
        and analysis_score(analysis, "proof_screen_score") <= 0.14
        and analysis_score(analysis, "scenic_score") <= 0.28
    ) else 0.0
    return clamp(
        (analysis_score(analysis, "builder_realworld_score") * 0.50)
        + (analysis_score(analysis, "builder_object_score") * 0.18)
        + (analysis_score(analysis, "builder_label_score") * 0.16)
        + (analysis_score(analysis, "movement_score") * 0.08)
        + (analysis_score(analysis, "energy_score") * 0.05)
        + (analysis_score(analysis, "person_score") * 0.03)
        + solo_bonus
        + primary_bonus
        + quiet_work_bonus
        - (analysis_score(analysis, "proof_screen_score") * 0.26)
        - (analysis_score(analysis, "scenic_score") * 0.14)
        - (analysis_score(analysis, "talking_head_score") * 0.08)
    )


def proof_priority_score(analysis):
    primary_bonus = 0.10 if analysis.get("primary_type") == "proof_screen" else 0.0
    no_people_bonus = 0.05 if analysis.get("people_mode") == "none" else 0.0
    return clamp(
        (analysis_score(analysis, "proof_screen_score") * 0.64)
        + (analysis_score(analysis, "proof_label_score") * 0.12)
        + (analysis_score(analysis, "edge_score") * 0.08)
        + (analysis_score(analysis, "energy_score") * 0.05)
        + primary_bonus
        + no_people_bonus
        - (analysis_score(analysis, "builder_realworld_score") * 0.16)
        - (analysis_score(analysis, "scenic_score") * 0.18)
    )


def scenic_priority_score(analysis):
    scenic_bonus = 0.12 if analysis.get("primary_type") == "scenic" else 0.0
    no_people_bonus = 0.06 if analysis.get("people_mode") == "none" else 0.0
    return clamp(
        (analysis_score(analysis, "scenic_score") * 0.54)
        + (analysis_score(analysis, "scenic_object_score") * 0.12)
        + (analysis_score(analysis, "scenic_label_score") * 0.12)
        + (analysis_score(analysis, "colorfulness_score") * 0.08)
        + (analysis_score(analysis, "nature_score") * 0.06)
        + (analysis_score(analysis, "movement_score") * 0.04)
        + scenic_bonus
        + no_people_bonus
        - (analysis_score(analysis, "proof_screen_score") * 0.22)
        - (analysis_score(analysis, "builder_realworld_score") * 0.12)
    )


def balanced_priority_score(analysis):
    return clamp(
        (builder_priority_score(analysis) * 0.52)
        + (proof_priority_score(analysis) * 0.24)
        + (scenic_priority_score(analysis) * 0.24)
    )


def rank_entries(entries, score_fn):
    ranked = []
    for entry in entries:
        scored_entry = dict(entry)
        scored_entry["rank_score"] = round(score_fn(entry["analysis"]), 4)
        scored_entry["rank_jitter"] = RNG.uniform(0.0, 0.04)
        ranked.append(scored_entry)
    ranked.sort(key=lambda item: item["rank_score"] + item["rank_jitter"], reverse=True)
    return ranked


def take_ranked_entries(ranked_entries, count, selected_paths=None, type_limits=None):
    selected = []
    selected_paths = selected_paths or set()
    type_counts = Counter()
    if type_limits is None:
        type_limits = {}

    for entry in ranked_entries:
        if len(selected) >= count:
            break
        if entry["path"] in selected_paths:
            continue
        primary_type = entry["analysis"]["primary_type"]
        if type_counts[primary_type] >= type_limits.get(primary_type, count):
            continue
        selected.append(entry)
        selected_paths.add(entry["path"])
        type_counts[primary_type] += 1
    return selected


def filter_entries(entries, predicate):
    return [entry for entry in entries if predicate(entry["analysis"])]


def select_ranked_source_videos(entries, style, min_count, max_count, scenic_share):
    if not entries:
        return []

    if style == "random":
        videos = [entry["path"] for entry in entries]
        return choose_random_source_videos(videos, min_count, max_count)

    desired_count = len(choose_random_source_videos([entry["path"] for entry in entries], min_count, max_count))
    desired_count = min(desired_count, len(entries))
    if desired_count <= 0:
        return []

    selected_entries = []
    selected_paths = set()

    if style == "scenic":
        ranked = rank_entries(filter_entries(entries, lambda analysis: is_scenic_candidate(analysis)), scenic_priority_score)
        selected_entries = take_ranked_entries(ranked, desired_count, selected_paths=selected_paths)
        if len(selected_entries) < desired_count:
            relaxed_ranked = rank_entries(
                filter_entries(entries, lambda analysis: is_scenic_candidate(analysis, relaxed=True)),
                scenic_priority_score,
            )
            selected_entries.extend(
                take_ranked_entries(
                    relaxed_ranked,
                    desired_count - len(selected_entries),
                    selected_paths=selected_paths,
                )
            )
    elif style == "proof":
        ranked = rank_entries(filter_entries(entries, lambda analysis: is_proof_candidate(analysis)), proof_priority_score)
        selected_entries = take_ranked_entries(ranked, desired_count, selected_paths=selected_paths)
        if len(selected_entries) < desired_count:
            relaxed_ranked = rank_entries(
                filter_entries(entries, lambda analysis: is_proof_candidate(analysis, relaxed=True)),
                proof_priority_score,
            )
            selected_entries.extend(
                take_ranked_entries(
                    relaxed_ranked,
                    desired_count - len(selected_entries),
                    selected_paths=selected_paths,
                )
            )
    elif style == "balanced":
        builder_count = max(1, round(desired_count * 0.55))
        proof_count = max(1, round(desired_count * 0.20))
        scenic_count = max(1, desired_count - builder_count - proof_count)

        selected_entries.extend(
            take_ranked_entries(
                rank_entries(
                    filter_entries(entries, lambda analysis: is_builder_candidate(analysis)),
                    builder_priority_score,
                ),
                builder_count,
                selected_paths=selected_paths,
            )
        )
        selected_entries.extend(
            take_ranked_entries(
                rank_entries(
                    filter_entries(entries, lambda analysis: is_proof_candidate(analysis)),
                    proof_priority_score,
                ),
                proof_count,
                selected_paths=selected_paths,
            )
        )
        selected_entries.extend(
            take_ranked_entries(
                rank_entries(
                    filter_entries(entries, lambda analysis: is_scenic_candidate(analysis)),
                    scenic_priority_score,
                ),
                scenic_count,
                selected_paths=selected_paths,
            )
        )
        if len(selected_entries) < desired_count:
            mixed_ranked = rank_entries(
                filter_entries(
                    entries,
                    lambda analysis: (
                        is_builder_candidate(analysis, relaxed=True)
                        or is_proof_candidate(analysis, relaxed=True)
                        or is_scenic_candidate(analysis, relaxed=True)
                    ),
                ),
                balanced_priority_score,
            )
            selected_entries.extend(
                take_ranked_entries(
                    mixed_ranked,
                    desired_count - len(selected_entries),
                    selected_paths=selected_paths,
                )
            )
    else:
        scenic_count = 0
        if desired_count > 1 and scenic_share > 0.0:
            scenic_count = min(
                max(1, round(desired_count * clamp(scenic_share, 0.0, 0.5))),
                desired_count - 1,
            )
        builder_ranked = rank_entries(
            filter_entries(entries, lambda analysis: is_builder_candidate(analysis)),
            builder_priority_score,
        )
        if scenic_count:
            scenic_ranked = rank_entries(
                filter_entries(entries, lambda analysis: is_scenic_candidate(analysis)),
                scenic_priority_score,
            )
            selected_entries.extend(
                take_ranked_entries(
                    scenic_ranked,
                    scenic_count,
                    selected_paths=selected_paths,
                )
            )
        selected_entries.extend(
            take_ranked_entries(
                builder_ranked,
                desired_count - len(selected_entries),
                selected_paths=selected_paths,
            )
        )
        if len(selected_entries) < desired_count:
            relaxed_builder_ranked = rank_entries(
                filter_entries(entries, lambda analysis: is_builder_candidate(analysis, relaxed=True)),
                builder_priority_score,
            )
            selected_entries.extend(
                take_ranked_entries(
                    relaxed_builder_ranked,
                    desired_count - len(selected_entries),
                    selected_paths=selected_paths,
                )
            )

    return [entry["path"] for entry in selected_entries[:desired_count]]


# ─────────────────────────────────────────────
# GET VIDEO DURATION
# ─────────────────────────────────────────────
def get_duration(filepath):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return float(result.strip())


@functools.lru_cache(maxsize=1024)
def probe_video_stream(filepath):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries",
        "-of",
        "json",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {filepath}")
    stream = streams[0]
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "pix_fmt": stream.get("pix_fmt") or "unknown",
        "color_range": stream.get("color_range") or "unknown",
        "color_space": stream.get("color_space") or "unknown",
        "color_transfer": stream.get("color_transfer") or "unknown",
        "color_primaries": stream.get("color_primaries") or "unknown",
    }


def build_extraction_filter(source):
    metadata = probe_video_stream(source)
    filters = []

    # iPhone HDR clips are commonly stored as BT.2020 HLG. Normalize them to BT.709
    # before scaling so FCP does not see washed-out or gray footage in the montage.
    if (
        metadata["color_space"] == "bt2020nc"
        and metadata["color_transfer"] == "arib-std-b67"
        and metadata["color_primaries"] == "bt2020"
    ):
        filters.append(
            "scale="
            "in_color_matrix=bt2020:"
            "out_color_matrix=bt709:"
            "in_primaries=bt2020:"
            "out_primaries=bt709:"
            "in_transfer=arib-std-b67:"
            "out_transfer=bt709"
        )

    filters.extend(
        [
            "scale=1920:1080:force_original_aspect_ratio=decrease",
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
            "fps=25",
            "format=yuv422p10le",
        ]
    )
    return ",".join(filters)


# ─────────────────────────────────────────────
# SMART: FIND VISUALLY ACTIVE TIMESTAMPS
# Uses ffmpeg scene detection to find moments where
# something is actually happening in the footage.
# Falls back to evenly-spaced segments if no scenes detected.
# ─────────────────────────────────────────────
def get_interesting_timestamps(filepath, total_dur):
    try:
        cmd = [
            "ffmpeg", "-i", filepath,
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
            "-vsync", "vfr",
            "-f", "null", "-"
        ]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            timeout=30
        )

        timestamps = []
        for line in result.stderr.split("\n"):
            if "pts_time:" in line:
                try:
                    pts = float(line.split("pts_time:")[1].split()[0])
                    # Skip the first and last 5% of the video (fades, dead frames)
                    if 0.05 * total_dur < pts < 0.95 * total_dur:
                        timestamps.append(pts)
                except (IndexError, ValueError):
                    pass

        if len(timestamps) >= 2:
            return timestamps

    except Exception:
        pass

    # Fallback: divide into 10 segments, pick midpoint of each interior segment
    segments = 10
    step = total_dur / segments
    return [step * i + step * 0.5 for i in range(1, segments - 1)]


# ─────────────────────────────────────────────
# EXTRACT A SINGLE CLIP
# - No audio (-an)
# - Normalize HDR iPhone clips to bt709 only when needed
# - ProRes 422 HQ for clean FCP import
# ─────────────────────────────────────────────
def extract_clip(source, start, duration, output_path):
    filter_chain = build_extraction_filter(source)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", source,

        "-vf", filter_chain,

        # ProRes 422 HQ — FCP native
        "-c:v", "prores_ks",
        "-profile:v", "3",
        "-pix_fmt", "yuv422p10le",

        # Tag the output so FCP reads colorspace correctly
        "-colorspace", "1",
        "-color_primaries", "1",
        "-color_trc", "1",

        # No audio
        "-an",

        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    OUTPUT_PATH = get_next_output_path()
    source_dirs = args.source_dirs or DEFAULT_SOURCE_DIRS

    if args.analysis_sample_videos < 1:
        print("❌ --analysis-sample-videos must be at least 1.")
        sys.exit(1)
    if not 0.0 <= args.scenic_share <= 0.5:
        print("❌ --scenic-share must be between 0.0 and 0.5.")
        sys.exit(1)

    # Get target duration from command line or prompt
    if args.target_seconds is not None:
        target_seconds = args.target_seconds
    else:
        try:
            target_seconds = float(input("⏱  How many seconds should the final clip be? "))
        except ValueError:
            print("❌ Please enter a number.")
            sys.exit(1)

    print(f"\n🎯 Target length: {target_seconds}s  |  Each clip: {MIN_CLIP_DURATION}–{MAX_CLIP_DURATION}s")
    print("─" * 55)
    print("📁 Source folders:")
    for source_dir in source_dirs:
        print(f"  - {source_dir}")
    print(f"🧭 Style: {args.style}")

    # Find all source videos
    all_videos = list_source_videos(source_dirs, recursive=args.recursive)

    if not all_videos:
        print("❌ No video files found in the selected source folders.")
        sys.exit(1)

    cache_path = os.path.expanduser(args.cache_path)
    model_cache_dir = os.path.expanduser(args.model_cache_dir)
    cache = load_analysis_cache(cache_path)
    cached_entries = []
    uncached_videos = []

    for video in all_videos:
        analysis = get_cached_analysis(cache, video)
        if analysis is None:
            uncached_videos.append(video)
            continue
        cached_entries.append({"path": video, "analysis": analysis})

    print(f"📂 Found {len(all_videos)} video(s).")
    print(f"🧠 Cache coverage: {len(cached_entries)}/{len(all_videos)} video(s).")

    newly_analyzed_entries = []
    ranked_entries = list(cached_entries)
    selected_videos = select_ranked_source_videos(
        ranked_entries,
        style=args.style,
        min_count=args.min_source_videos,
        max_count=args.max_source_videos,
        scenic_share=args.scenic_share,
    ) if ranked_entries else []

    remaining_uncached = list(uncached_videos)
    RNG.shuffle(remaining_uncached)
    analyzed_total = 0
    full_library_index = args.index_all or args.index_only or args.style != "random"
    if full_library_index:
        analysis_budget = len(remaining_uncached)
        if remaining_uncached:
            print(
                f"🗂️  Full-library indexing enabled for {args.style}. "
                f"Analyzing {len(remaining_uncached)} uncached video(s) in batches of {args.analysis_sample_videos}.\n"
            )
    else:
        analysis_budget = max(args.analysis_sample_videos, args.max_source_videos * 4)

    while (
        remaining_uncached
        and analyzed_total < analysis_budget
        and (
            full_library_index
            or not selected_videos
            or len(selected_videos) < min(args.min_source_videos, len(ranked_entries))
        )
    ):
        batch_size = min(
            args.analysis_sample_videos,
            analysis_budget - analyzed_total,
            len(remaining_uncached),
        )
        analysis_batch = remaining_uncached[:batch_size]
        remaining_uncached = remaining_uncached[batch_size:]
        analyzed_total += len(analysis_batch)

        print(f"🔬 Analyzing {len(analysis_batch)} new video(s) to grow the local library...\n")
        for video in analysis_batch:
            try:
                analysis = analyze_video(video, args.tagger, model_cache_dir)
                cache_video_analysis(cache, video, analysis)
                entry = {"path": video, "analysis": analysis}
                newly_analyzed_entries.append(entry)
                print(f"  🏷️  {os.path.basename(video)} — {format_analysis_summary(analysis)}")
            except Exception as exc:
                print(f"  ❌ {os.path.basename(video)} — analysis failed: {exc}")
        print()
        save_analysis_cache(cache_path, cache)

        ranked_entries = cached_entries + newly_analyzed_entries
        if not ranked_entries:
            continue
        selected_videos = select_ranked_source_videos(
            ranked_entries,
            style=args.style,
            min_count=args.min_source_videos,
            max_count=args.max_source_videos,
            scenic_share=args.scenic_share,
        )
        if (
            not full_library_index
            and not selected_videos
            and remaining_uncached
            and analyzed_total < analysis_budget
        ):
            print(f"↻ No strong {args.style} candidates yet. Expanding analysis pool...\n")

    if not ranked_entries:
        print("❌ No usable videos could be analyzed.")
        sys.exit(1)

    if args.index_only:
        print(
            f"✅ Indexed {len(ranked_entries)}/{len(all_videos)} video(s). "
            f"Cache saved to {cache_path}"
        )
        sys.exit(0)

    if not selected_videos:
        print("❌ Could not choose any source videos for this run.")
        sys.exit(1)

    print(f"🎯 Selected {len(selected_videos)} ranked source video(s) for this run.")
    for video in selected_videos:
        analysis = get_cached_analysis(cache, video)
        if not analysis:
            continue
        print(f"  ✅ {os.path.basename(video)} — {format_analysis_summary(analysis)}")
    print("\n🔎 Running scene analysis on the selected videos...\n")

    # Build a per-video timestamp pool from the random sample only.
    source_candidates = []
    for video in selected_videos:
        try:
            total_dur = get_duration(video)
            if total_dur < MIN_CLIP_DURATION:
                continue
            timestamps = get_interesting_timestamps(video, total_dur)
            name = os.path.basename(video)
            valid = [t for t in timestamps if t + MAX_CLIP_DURATION <= total_dur]
            if valid:
                print(f"  🔍 {name} — {len(valid)} interesting moment(s) found")
                source_candidates.append((video, valid, total_dur))
            else:
                print(f"  ⚠️  {name} — no valid moments, skipping")
        except Exception as e:
            print(f"  ❌ {os.path.basename(video)} — error: {e}")

    if not source_candidates:
        print("\n❌ No usable footage found.")
        sys.exit(1)

    print(f"\n✂️  Extracting clips from {len(source_candidates)} ranked source video(s)...\n")

    # Cycle through a shuffled source order for better variety.
    RNG.shuffle(source_candidates)

    temp_dir = tempfile.mkdtemp(prefix="broll_tmp_")
    clips = []
    total_so_far = 0.0
    pool_index = 0
    attempts = 0
    max_attempts = max(60, len(source_candidates) * 8)

    while total_so_far < target_seconds and attempts < max_attempts:
        attempts += 1

        if not source_candidates:
            break

        if pool_index > 0 and pool_index % len(source_candidates) == 0:
            RNG.shuffle(source_candidates)

        source, timestamps, total_dur = source_candidates[pool_index % len(source_candidates)]
        pool_index += 1

        remaining = target_seconds - total_so_far
        if clips and remaining < MIN_TAIL_CLIP_DURATION:
            break

        clip_len = round(RNG.uniform(MIN_CLIP_DURATION, MAX_CLIP_DURATION), 2)
        clip_len = min(clip_len, remaining)

        valid_timestamps = [t for t in timestamps if t + clip_len <= total_dur]
        if not valid_timestamps:
            continue

        base_time = RNG.choice(valid_timestamps)

        # Slight jitter around the detected scene timestamp for natural feel
        jitter = RNG.uniform(-0.5, 0.5)
        start_time = max(0.0, min(base_time + jitter, total_dur - clip_len))
        start_time = round(start_time, 2)

        out_clip = os.path.join(temp_dir, f"clip_{len(clips):04d}.mov")
        extract_clip(source, start_time, clip_len, out_clip)

        if os.path.exists(out_clip) and os.path.getsize(out_clip) > 0:
            clips.append(out_clip)
            total_so_far += clip_len
            src_name = os.path.basename(source)
            print(f"  ✅ Clip {len(clips):02d} — {clip_len}s from '{src_name}' @ {start_time:.1f}s  [{total_so_far:.1f}/{target_seconds}s]")
        else:
            print(f"  ⚠️  Failed to extract from '{os.path.basename(source)}', skipping.")

    if not clips:
        print("\n❌ Could not extract any clips. Check your video files.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    print(f"\n🔗 Joining {len(clips)} clips into final video...")

    # Build concat list
    list_file = os.path.join(temp_dir, "concat_list.txt")
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    # Every extracted clip is normalized to the same frame rate and codec, so
    # a fast concat copy keeps the final montage timing stable.
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        OUTPUT_PATH
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    shutil.rmtree(temp_dir)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        out_name = os.path.basename(OUTPUT_PATH)
        print(f"\n🎬 Done!")
        print(f"   📁 {OUTPUT_PATH}")
        print(f"   🎞  {out_name}  |  {total_so_far:.1f}s  |  {size_mb:.1f} MB")
    else:
        print("\n❌ Something went wrong during export. Check that ffmpeg is installed.")


if __name__ == "__main__":
    main()

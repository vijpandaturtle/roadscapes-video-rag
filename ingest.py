"""
ingest.py — Build the LanceDB index from videos + CSV labels.
Run once (or re-run when you add new clips).

Usage:
    python ingest.py --split train
    python ingest.py --split test
"""

import argparse
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
VIDEO_DIR  = BASE_DIR / "videos"
LABEL_DIR  = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "index"

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Load CLIP ─────────────────────────────────────────────────────────────────
def load_clip(device):
    print(f"Loading CLIP on {device}...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


# ── Extract frames ────────────────────────────────────────────────────────────
def extract_frames(video_path: Path) -> list[np.ndarray]:
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ── Embed frames ──────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_frames(frames, model, processor, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(frames), batch_size):
        batch  = [Image.fromarray(f) for f in frames[i : i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        feats  = model.get_image_features(**inputs)
        # handle newer transformers returning a model output object
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().float().numpy())
    return np.vstack(all_embeddings)


# ── Embed texts ───────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_texts(texts, model, processor, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats  = model.get_text_features(**inputs)
    # handle newer transformers returning a model output object
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


# ── Fuse embeddings ───────────────────────────────────────────────────────────
def fuse(visual_emb, text_emb, alpha=0.7):
    fused = alpha * visual_emb + (1 - alpha) * text_emb
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return fused / np.clip(norms, 1e-8, None)


# ── Schema ────────────────────────────────────────────────────────────────────
def make_schema(dim=512):
    return pa.schema([
        pa.field("vector",        pa.list_(pa.float32(), dim)),
        pa.field("clip_name",     pa.string()),
        pa.field("clip_path",     pa.string()),
        pa.field("frame_idx",     pa.int32()),
        pa.field("total_frames",  pa.int32()),
        pa.field("label_text",    pa.string()),
        pa.field("split",         pa.string()),
        pa.field("action",        pa.string()),
        pa.field("justification", pa.string()),
        pa.field("instruction",   pa.string()),
        pa.field("surroundings",  pa.string()),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
def ingest(split: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device           = get_device()
    model, processor = load_clip(device)

    # Load labels
    label_csv = LABEL_DIR / f"roadscapes_x_{split}.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")
    labels_df = pd.read_csv(label_csv)
    labels_df.columns = labels_df.columns.str.strip()
    filename_col = "Video File Name"
    labels_df[filename_col] = labels_df[filename_col].str.strip()
    label_lookup = {row[filename_col]: row.to_dict() for _, row in labels_df.iterrows()}

    # Find videos — rglob handles nested Sequence_Day_* subdirs
    video_dir   = VIDEO_DIR / split
    video_paths = sorted(video_dir.rglob("*.mp4"))

    print(f"Video dir : {video_dir.resolve()}")
    print(f"Exists    : {video_dir.exists()}")
    print(f"Found {len(video_paths)} videos in '{split}' split.")

    if not video_paths:
        print("No videos found — check VIDEO_DIR path above.")
        return

    # LanceDB setup
    db         = lancedb.connect(str(OUTPUT_DIR / "roadscapes.lancedb"))
    table_name = f"frames_{split}"
    if table_name in db.list_tables():
        db.drop_table(table_name)
    table = db.create_table(table_name, schema=make_schema())

    WRITE_BATCH = 500

    for video_path in tqdm(video_paths, desc="Embedding clips"):
        clip_name = video_path.name
        label     = label_lookup.get(clip_name)

        if label is None:
            print(f"  [WARN] No label for {clip_name}, skipping.")
            continue

        action        = str(label.get("What is the action being performed by the ego vehicle? Answer in a single sentence.", ""))
        justification = str(label.get("What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.", ""))
        instruction   = str(label.get("What should be driver be doing now ? Provide me a definite action.", ""))
        surroundings  = str(label.get("Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.", ""))
        label_text    = " | ".join([action, justification, instruction, surroundings])

        frames = extract_frames(video_path)
        if not frames:
            print(f"  [WARN] No frames in {clip_name}, skipping.")
            continue

        visual_embs = embed_frames(frames, model, processor, device)
        text_emb    = embed_texts([label_text], model, processor, device)
        text_embs   = np.repeat(text_emb, len(frames), axis=0)
        fused_embs  = fuse(visual_embs, text_embs)

        rows = []
        for frame_idx in range(len(frames)):
            rows.append({
                "vector":        fused_embs[frame_idx].tolist(),
                "clip_name":     clip_name,
                "clip_path":     str(video_path),
                "frame_idx":     frame_idx,
                "total_frames":  len(frames),
                "label_text":    label_text,
                "split":         split,
                "action":        action,
                "justification": justification,
                "instruction":   instruction,
                "surroundings":  surroundings,
            })
            if len(rows) >= WRITE_BATCH:
                table.add(rows)
                rows = []

        if rows:
            table.add(rows)

    total = table.count_rows()
    print(f"\nTotal frame vectors in LanceDB: {total}")

    if total > 50_000:
        print("Building ANN index...")
        table.create_index(metric="cosine")

    print(f"Saved → {OUTPUT_DIR}/roadscapes.lancedb  (table: {table_name})")
    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()
    ingest(args.split)
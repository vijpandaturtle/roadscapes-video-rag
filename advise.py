"""
advise.py — Generate structured driving advice for a test scenario using RAG.

Retrieves the top-3 most similar training clips from LanceDB, uses their
verified labels as context, and generates:
  - ACTION   : What the driver should do right now
  - REASONING: Why, grounded in the retrieved similar scenarios
  - WARNING  : What to watch out for

Usage:
    # From a test video clip
    python advise.py --clip /path/to/videos/test/Sequence_Day_1/clip1.mp4

    # From a text description
    python advise.py --text "I am driving on a two lane highway at night following a truck"

    # From a test clip + its label CSV (richer query — fuses visual + text)
    python advise.py --clip /path/to/clip.mp4 --label_csv /path/to/roadscapes_x_test.csv
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import anthropic
import cv2
import lancedb
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from dotenv import load_dotenv
load_dotenv()

BASE_DIR  = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "index"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(device):
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


@torch.no_grad()
def embed_text(text: str, model, processor, device) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    feats  = model.get_text_features(**inputs)
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    feats  = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


@torch.no_grad()
def embed_frames_clip(frames: list[np.ndarray], model, processor, device, batch_size=32) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided to embed_frames_clip.")
    all_embs = []
    for i in range(0, len(frames), batch_size):
        batch  = [Image.fromarray(f) for f in frames[i:i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        feats  = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().float().numpy())
    return np.vstack(all_embs)


def extract_all_frames(video_path: str) -> list[np.ndarray]:
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_frames(video_path: str, n: int = 5) -> list[np.ndarray]:
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def frame_to_b64(frame: np.ndarray, max_size: int = 512) -> str:
    img = Image.fromarray(frame)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def load_label_for_clip(label_csv: str, clip_name: str) -> dict | None:
    import pandas as pd
    df = pd.read_csv(label_csv)
    df.columns            = df.columns.str.strip()
    df["Video File Name"] = df["Video File Name"].str.strip()
    row = df[df["Video File Name"] == clip_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def label_to_text(label: dict) -> str:
    return " | ".join([
        str(label.get("What is the action being performed by the ego vehicle? Answer in a single sentence.", "")),
        str(label.get("What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.", "")),
        str(label.get("What should be driver be doing now ? Provide me a definite action.", "")),
        str(label.get("Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.", "")),
    ])


def retrieve_top_k_clips(query_vec: np.ndarray, table, top_k: int = 3) -> list[dict]:
    results = (
        table.search(query_vec.flatten().tolist())
             .metric("cosine")
             .limit(top_k * 50)
             .to_list()
    )
    seen = {}
    for row in results:
        clip  = row["clip_name"]
        score = 1.0 - float(row.get("_distance", 1.0))
        if clip not in seen:
            seen[clip] = {
                "score":     score,
                "frame_idx": row["frame_idx"],
                "meta": {
                    "clip_name":    row["clip_name"],
                    "clip_path":    row["clip_path"],
                    "frame_idx":    row["frame_idx"],
                    "total_frames": row["total_frames"],
                    "label_text":   row["label_text"],
                    "label": {
                        "What is the action being performed by the ego vehicle? Answer in a single sentence.":                                row["action"],
                        "What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.": row["justification"],
                        "What should be driver be doing now ? Provide me a definite action.":                                               row["instruction"],
                        "Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.":    row["surroundings"],
                    },
                },
            }
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def build_query_vector(clip_path, text, label, model, processor, device) -> np.ndarray:
    if clip_path:
        frames = extract_all_frames(clip_path)
        if not frames:
            raise ValueError(f"Could not extract frames from: {clip_path}")
        visual_embs = embed_frames_clip(frames, model, processor, device)
        visual_mean = visual_embs.mean(axis=0, keepdims=True)
        visual_mean = visual_mean / np.linalg.norm(visual_mean)
        if label:
            text_emb = embed_text(label_to_text(label), model, processor, device)
            fused    = 0.7 * visual_mean + 0.3 * text_emb
            fused    = fused / np.linalg.norm(fused)
            return fused.astype("float32")
        return visual_mean.astype("float32")
    if text:
        return embed_text(text, model, processor, device).astype("float32")
    raise ValueError("Provide --clip and/or --text.")


ADVICE_SYSTEM = """You are an expert autonomous driving advisor.
You are given a description of a current driving scenario (and optionally video frames),
along with the top-3 most similar scenarios retrieved from a training database,
each with verified labels describing what the ego vehicle was doing and what it should do.

Your job is to synthesise this information and produce structured driving advice.

Respond ONLY in valid JSON with exactly this structure:
{
  "action": "A single clear sentence: what the driver should do RIGHT NOW.",
  "reasoning": "2-3 sentences explaining why, referencing patterns from the similar scenarios.",
  "warning": "A single sentence flagging the most important hazard or risk to watch for."
}

Be specific and concrete. Do not hedge with 'maybe' or 'possibly'.
Ground your reasoning in the retrieved scenarios."""


def build_advice_prompt(query_description, retrieved, query_frames) -> list[dict]:
    context_lines = []
    for i, r in enumerate(retrieved, 1):
        m     = r["meta"]
        label = m["label"]
        context_lines.append(f"--- Similar Scenario {i} (similarity: {r['score']:.3f}) ---")
        context_lines.append(f"Clip: {m['clip_name']}")
        context_lines.append(f"Action: {label.get('What is the action being performed by the ego vehicle? Answer in a single sentence.', '')}")
        context_lines.append(f"Justification: {label.get('What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.', '')}")
        context_lines.append(f"Recommended action: {label.get('What should be driver be doing now ? Provide me a definite action.', '')}")
        context_lines.append(f"Scene: {label.get('Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.', '')}")
        context_lines.append("")

    content = []
    if query_frames:
        for i, frame in enumerate(query_frames):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_to_b64(frame)},
            })
            content.append({"type": "text", "text": f"Query frame {i+1} of {len(query_frames)}"})

    user_text = ""
    if query_description:
        user_text += f"CURRENT SCENARIO:\n{query_description}\n\n"
    elif query_frames:
        user_text += "CURRENT SCENARIO:\n(See the video frames above)\n\n"

    user_text += f"RETRIEVED SIMILAR SCENARIOS FROM TRAINING DATABASE:\n{chr(10).join(context_lines)}"
    user_text += "\nBased on the current scenario and these retrieved examples, provide structured driving advice."
    content.append({"type": "text", "text": user_text})

    return [{"role": "user", "content": content}]


def generate_advice(query_description, retrieved, query_frames, client) -> dict:
    messages = build_advice_prompt(query_description, retrieved, query_frames)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=ADVICE_SYSTEM,
        messages=messages,
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def print_advice(advice: dict, retrieved: list[dict], query_source: str):
    print("\n" + "═" * 70)
    print(f"  DRIVING ADVICE")
    print(f"  Query: {query_source}")
    print("═" * 70)
    print(f"\n  ACTION\n  ▶  {advice.get('action', '')}")
    print(f"\n  REASONING\n  {advice.get('reasoning', '')}")
    print(f"\n  WARNING\n  ⚠  {advice.get('warning', '')}")
    print("\n" + "─" * 70)
    print("  Retrieved context:")
    for i, r in enumerate(retrieved, 1):
        print(f"  [{i}] {r['meta']['clip_name']}  (score: {r['score']:.3f})")
    print("═" * 70 + "\n")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--clip",        type=str, default=None)
#     parser.add_argument("--text",        type=str, default=None)
#     parser.add_argument("--label_csv",   type=str, default=None)
#     parser.add_argument("--top_k",       type=int, default=3)
#     parser.add_argument("--output_json", type=str, default=None)
#     args = parser.parse_args()

#     if not args.clip and not args.text:
#         parser.error("Provide --clip and/or --text.")

#     if not os.environ.get("ANTHROPIC_API_KEY"):
#         raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable.")

#     db_path = INDEX_DIR / "roadscapes.lancedb"
#     if not db_path.exists():
#         raise FileNotFoundError(f"LanceDB not found at {db_path}. Run ingest.py first.")

#     db    = lancedb.connect(str(db_path))
#     table = db.open_table("frames_train")

#     device           = get_device()
#     model, processor = load_clip(device)
#     client           = anthropic.Anthropic()

#     label = None
#     if args.clip and args.label_csv:
#         label = load_label_for_clip(args.label_csv, Path(args.clip).name)

#     query_vec = build_query_vector(
#         clip_path=args.clip,
#         text=args.text,
#         label=label,
#         model=model,
#         processor=processor,
#         device=device,
#     )

#     retrieved    = retrieve_top_k_clips(query_vec, table, top_k=args.top_k)
#     query_frames = sample_frames(args.clip, n=5) if args.clip else None
#     query_source = args.clip or args.text

#     advice = generate_advice(
#         query_description=args.text,
#         retrieved=retrieved,
#         query_frames=query_frames,
#         client=client,
#     )

#     print_advice(advice, retrieved, query_source)

#     if args.output_json:
#         with open(args.output_json, "w") as f:
#             json.dump({"advice": advice, "retrieved": [r["meta"] for r in retrieved]}, f, indent=2)
#         print(f"Saved to {args.output_json}")


# if __name__ == "__main__":
#     main()
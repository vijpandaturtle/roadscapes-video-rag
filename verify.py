"""
verify_labels.py — Use a VLM (Claude) to verify or regenerate labels for clips.

Takes the output of query.py and re-scores each label against its video frames.
Useful for auditing the partially-verified synthetic labels in your dataset.

Usage:
    python verify_labels.py \
        --clip_path /path/to/Sequence_Night_2_clip88.mp4 \
        --label_csv /path/to/VIDEO_DATA_ANON/labels/roadscapes_x_train.csv

Requirements:
    pip install anthropic opencv-python pillow
    export ANTHROPIC_API_KEY=your_key
"""

import argparse
import base64
import json
import os
from pathlib import Path

import anthropic
import cv2
import numpy as np
from PIL import Image


# ── Sample N evenly-spaced frames from a video ───────────────────────────────
def sample_frames(video_path: str, n_frames: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ── Encode a frame to base64 JPEG ────────────────────────────────────────────
def frame_to_b64(frame: np.ndarray, max_size: int = 512) -> str:
    img = Image.fromarray(frame)
    img.thumbnail((max_size, max_size))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ── Load label for a specific clip ───────────────────────────────────────────
def load_label(label_csv: str, clip_name: str) -> dict | None:
    import pandas as pd
    df = pd.read_csv(label_csv)
    df.columns = df.columns.str.strip()
    df["Video File Name"] = df["Video File Name"].str.strip()
    row = df[df["Video File Name"] == clip_name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ── Build the VLM prompt ──────────────────────────────────────────────────────
VERIFY_PROMPT = """You are a driving scene analyst verifying synthetic labels for an autonomous driving dataset.

You are given {n} evenly-spaced frames from a 5-second dashcam video clip, followed by its existing synthetic label.

Your task:
1. Carefully examine the frames.
2. For each label field, score it as CORRECT, PARTIALLY_CORRECT, or INCORRECT, and briefly explain why.
3. If a field is incorrect or partially correct, provide a corrected version.
4. Give an overall confidence score (0.0–1.0) for the label.

Existing label:
- Action: {action}
- Justification: {justification}
- Should do: {should_do}
- Scene description: {scene}

Respond ONLY in valid JSON with this structure:
{{
  "overall_confidence": 0.0,
  "fields": {{
    "action":        {{"verdict": "CORRECT|PARTIALLY_CORRECT|INCORRECT", "reason": "...", "correction": "...or null"}},
    "justification": {{"verdict": "...", "reason": "...", "correction": "...or null"}},
    "should_do":     {{"verdict": "...", "reason": "...", "correction": "...or null"}},
    "scene":         {{"verdict": "...", "reason": "...", "correction": "...or null"}}
  }}
}}"""


# ── Call Claude vision ────────────────────────────────────────────────────────
def verify_with_claude(frames: list[np.ndarray], label: dict, client: anthropic.Anthropic) -> dict:
    # Build image content blocks (max 5 frames to stay within token limits)
    content = []
    for i, frame in enumerate(frames[:5]):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_to_b64(frame),
            },
        })
        content.append({
            "type": "text",
            "text": f"Frame {i+1} of {len(frames)}"
        })

    # Add the verification prompt
    content.append({
        "type": "text",
        "text": VERIFY_PROMPT.format(
            n=len(frames),
            action=label.get("What is the action being performed by the ego vehicle? Answer in a single sentence.", ""),
            justification=label.get("What is the justification for the current action being performed by the ego vehicle? Answer in a single sentence.", ""),
            should_do=label.get("What should be driver be doing now ? Provide me a definite action.", ""),
            scene=label.get("Tell me about the surroundings such as the weather type, road type, time of day and scenario in two sentences.", ""),
        )
    })

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Pretty print verification result ─────────────────────────────────────────
def print_verification(clip_name: str, result: dict):
    conf = result.get("overall_confidence", "?")
    print(f"\n{'═'*70}")
    print(f"  Clip: {clip_name}")
    print(f"  Overall confidence: {conf:.2f}" if isinstance(conf, float) else f"  Overall confidence: {conf}")
    print(f"{'─'*70}")
    for field, info in result.get("fields", {}).items():
        verdict = info.get("verdict", "?")
        reason = info.get("reason", "")
        correction = info.get("correction")
        icon = {"CORRECT": "✓", "PARTIALLY_CORRECT": "~", "INCORRECT": "✗"}.get(verdict, "?")
        print(f"  {icon} {field:<16} {verdict}")
        print(f"    Reason: {reason}")
        if correction:
            print(f"    Correction: {correction}")
    print(f"{'═'*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", required=True, help="Path to .mp4 clip")
    parser.add_argument("--label_csv", required=True, help="Path to roadscapes_x_train.csv or test.csv")
    parser.add_argument("--n_frames", type=int, default=5, help="Number of frames to send to VLM")
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save result to JSON")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)
    clip_name = Path(args.clip_path).name
    label = load_label(args.label_csv, clip_name)

    if label is None:
        print(f"No label found for {clip_name} in {args.label_csv}")
        return

    print(f"Sampling {args.n_frames} frames from {clip_name}...")
    frames = sample_frames(args.clip_path, args.n_frames)

    print("Sending to Claude for verification...")
    result = verify_with_claude(frames, label, client)

    print_verification(clip_name, result)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"clip": clip_name, "label": label, "verification": result}, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
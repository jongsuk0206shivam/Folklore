"""
Folklore Channel — Free Pipeline
Produces ONE video (script -> voice -> visuals -> captions -> render) from the
next "idea" entry in queue.json, using entirely free tools.

Environment variables required (set as GitHub Actions secrets):
  GEMINI_API_KEY   - https://aistudio.google.com/apikey  (free, no card needed)
  PEXELS_API_KEY   - https://www.pexels.com/api/          (free, no card needed)

System requirement: ffmpeg (installed by the GitHub Actions workflow)
Python requirements: see requirements.txt
"""

import json
import os
import subprocess
from pathlib import Path

import requests

QUEUE_PATH = Path("queue.json")
OUTPUT_DIR = Path("output")
WORK_DIR = Path("work")

VOICE = "en_US-lessac-medium"   # any piper voice name — auto-downloads on first use
FPS = 25
WIDTH, HEIGHT = 1080, 1920      # vertical: Shorts / Reels / TikTok

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

SCRIPT_PROMPT = """You are a folklore narrator for a short-form video series. Write a script for a short vertical video about:

Story: {story}
Culture/Region: {culture}
Format: {format_note}
Target length: {length_note}

Rules:
- Open with a hook in the first 2 sentences pulled from the story's own imagery, not a generic "did you know" line
- Ground it with one real, specific detail of where/when it comes from
- Plain spoken language, short sentences, no stage directions, no bullet points, no headers
- {ending_rule}
- Return ONLY the narration text, nothing else — no title, no labels, no quotation marks around it
"""


def load_queue():
    return json.loads(QUEUE_PATH.read_text())


def save_queue(queue):
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


def next_idea(queue):
    for item in queue:
        if item["status"] == "idea":
            return item
    return None


def build_prompt(item):
    is_part = item.get("part") and item.get("part_total")
    if is_part:
        format_note = f"Part {item['part']} of {item['part_total']}"
        length_note = "45-75 seconds spoken"
        if item["part"] < item["part_total"]:
            ending_rule = 'End on a genuine cliffhanger tied to the next real plot beat, never "stay tuned"'
        else:
            ending_rule = "End with the story's actual cultural meaning, not a canned moral"
    else:
        format_note = "standalone"
        length_note = "30-60 seconds spoken"
        ending_rule = "End with the story's actual cultural meaning, not a canned moral"

    return SCRIPT_PROMPT.format(
        story=item["story"],
        culture=item["culture"],
        format_note=format_note,
        length_note=length_note,
        ending_rule=ending_rule,
    )


def pick_model():
    """Ask Google which models this key can actually use right now, instead of
    trusting a hardcoded name — Gemini model names get renamed/retired often
    enough that a hardcoded string is the most likely thing to break here."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    usable = [
        m["name"] for m in models
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    if not usable:
        raise RuntimeError(
            "No models supporting generateContent are available to this API key. "
            "Double-check GEMINI_API_KEY was created at aistudio.google.com/apikey."
        )
    flash_models = [m for m in usable if "flash" in m.lower()]
    return (flash_models or usable)[0]


def generate_script(item):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    prompt = build_prompt(item)
    model = pick_model()
    print(f"Using Gemini model: {model}")
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={GEMINI_API_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def generate_voice(script_text, out_wav):
    text_path = WORK_DIR / "script.txt"
    text_path.write_text(script_text)
    with open(text_path, "r") as f_in:
        subprocess.run(
            ["piper", "--model", VOICE, "--output_file", str(out_wav.resolve())],
            stdin=f_in,
            check=True,
        )


def fetch_visuals(query, n, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY is not set")
    headers = {"Authorization": PEXELS_API_KEY}
    resp = requests.get(
        "https://api.pexels.com/v1/search",
        headers=headers,
        params={"query": query, "per_page": n, "orientation": "portrait"},
        timeout=30,
    )
    resp.raise_for_status()
    photos = resp.json().get("photos", [])
    paths = []
    for i, photo in enumerate(photos[:n]):
        img_path = out_dir / f"img{i}.jpg"
        img_data = requests.get(photo["src"]["portrait"], timeout=30).content
        img_path.write_bytes(img_data)
        paths.append(img_path)
    return paths


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def generate_captions(wav_path, out_srt):
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path))

    def fmt_ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    out_srt.write_text("\n".join(lines))


def assemble_video(images, audio_wav, srt_path, out_mp4):
    audio_duration = get_duration(audio_wav)
    seg_dur = max(3, audio_duration / len(images))
    frames = int(seg_dur * FPS)

    segment_paths = []
    for i, img in enumerate(images):
        seg_path = WORK_DIR / f"seg{i}.mp4"
        vf = (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},"
            f"zoompan=z='min(zoom+0.0015,1.2)':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"fade=t=in:st=0:d=0.4,fade=t=out:st={seg_dur - 0.4}:d=0.4"
        )
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(img.resolve()),
            "-vf", vf, "-t", str(seg_dur), "-r", str(FPS),
            "-pix_fmt", "yuv420p", str(seg_path.resolve()), "-loglevel", "error",
        ], check=True)
        segment_paths.append(seg_path)

    list_path = WORK_DIR / "list.txt"
    list_path.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_paths))
    visuals_path = WORK_DIR / "visuals.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path.resolve()),
        "-c", "copy", str(visuals_path.resolve()), "-loglevel", "error",
    ], check=True)

    with_audio_path = WORK_DIR / "with_audio.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(visuals_path.resolve()), "-i", str(audio_wav.resolve()),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        str(with_audio_path.resolve()), "-loglevel", "error",
    ], check=True)

    subtitle_style = (
        "FontName=DejaVu Sans,FontSize=26,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=3,Outline=1.5,MarginV=90,Alignment=2"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", str(with_audio_path.resolve()),
        "-vf", f"subtitles={srt_path.resolve()}:force_style='{subtitle_style}'",
        "-c:a", "copy", str(out_mp4.resolve()), "-loglevel", "error",
    ], check=True)


def main():
    WORK_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    queue = load_queue()
    item = next_idea(queue)
    if not item:
        print("No ideas waiting in queue.json — add some before running again.")
        return

    print(f"Producing: {item['story']} ({item['culture']})")

    script_text = generate_script(item)
    (WORK_DIR / "script.txt").write_text(script_text)
    print("Script done:\n" + script_text)

    audio_wav = WORK_DIR / "narration.wav"
    generate_voice(script_text, audio_wav)
    print("Voice done.")

    images = fetch_visuals(f"{item['culture']} forest folklore", 4, WORK_DIR / "images")
    if not images:
        images = fetch_visuals(item["culture"], 4, WORK_DIR / "images")
    print(f"Fetched {len(images)} background images.")

    srt_path = WORK_DIR / "captions.srt"
    generate_captions(audio_wav, srt_path)
    print("Captions done.")

    out_mp4 = OUTPUT_DIR / f"{item['id']}.mp4"
    assemble_video(images, audio_wav, srt_path, out_mp4)
    print(f"Video rendered: {out_mp4}")

    item["status"] = "rendered"
    item["output"] = str(out_mp4)
    save_queue(queue)


if __name__ == "__main__":
    main()

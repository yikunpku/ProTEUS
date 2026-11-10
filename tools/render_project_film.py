#!/usr/bin/env python3
"""Compose the 140-second ProTEUS film from newly rendered paper-source scenes.

Requirements: Python 3.10+, Pillow, FFmpeg/libx264, and Georgia fonts.
Run molecular, landscape, chart and schematic scripts in tools/video first,
then run this script. See tools/video/README.md for source provenance.

The molecular footage is rendered from native structure coordinates and the
paper's ChimeraX/Blender scenes. Charts preserve the manuscript data. Diagram
animations are conceptual. No PDF crops or old dark MP4 footage are used.
SCENES is the edit list; captions and chapters are generated from the same list.
Original source files in Dropbox are read-only and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1920, 1080, 24
INK, BLUE, MUTED, RULE = "#2c3e50", "#5290d3", "#7f8c8d", "#e4e9ed"
MARGIN = 100
VIEWPORT = (100, 176, 1820, 946)
SELF_TITLED = {
    'world-model-intro.mp4', 'world-model-loop.mp4', 'world-model-outro.mp4', 'architecture.mp4', 'attention-comparison.mp4',
    'guided-exploration.mp4', 'structure-projection.mp4', 'scaling-architecture.mp4',
    'wetlab-loop.mp4', 'sacb-sequence.mp4', 'aldh-sequence.mp4',
    'sacb-recursive.mp4', 'aldh-recursive.mp4',
}



@dataclass(frozen=True)
class Scene:
    title: str
    group: str
    seconds: float
    asset: str
    source_start: float
    source_length: float
    caption: str
    reference: str = ""
    end_hold: float = 0


SCENES = (
    Scene("A mutation-resolved world model", "ProTEUS", 8, "world-model-intro.mp4", 0, 8, ""),
    Scene("A mutation-resolved world model", "Model", 9, "world-model-loop.mp4", 0, 12, ""),
    Scene("Three modalities, one joint embedding space", "Model", 11, "architecture.mp4", 0, 12, ""),
    Scene("Attention to mutations", "Model", 9, "attention-comparison.mp4", 0, 14, ""),
    Scene("Mutation-guided exploration", "Model", 15, "guided-exploration.mp4", 0, 15, ""),
    Scene("Fast structures with physical constraints", "Structure", 21, "structure-projection.mp4", 0, 21, ""),
    Scene("7.8–10.8× faster than AlphaFold 3", "Structure", 3, "structure-speed.mp4", 0, 6,
          "Five CASP15 atom-count bins · NVIDIA RTX A6000"),
    Scene("Pretraining corpus: 1.32M variant measurements", "Pretraining", 9, "scaling-architecture.mp4", 0, 9, ""),
    Scene("A scaling law for functional data", "Pretraining", 5, "scaling.mp4", 0, 6,
          "Held-out variants within the pretraining assays · fixed model architecture"),
    Scene("Zero-shot prediction and variant selection", "Prediction", 8, "zero-shot.mp4", 0, 6,
          "Mean ± SEM · 19 prediction assays / 13 selection assays · Figure 2E–F"),
    Scene("Protein evolution at a matched measurement budget", "In silico benchmark", 5, "closed-loop.mp4", 0, 6,
          "17 datasets · 30 initial variants + five guided rounds of 30",
          "EVOLVEpro · Science 2025, adr6006   |   MULTI-evolve · Science 2026, aea1820"),
    Scene("Levansucrase · SacB", "Wet-lab closed loop", 5, "sacb-sequence.mp4", 0, 8, ""),
    Scene("SacB · Cumulative top 10 through each round", "Wet-lab closed loop", 6, "sacb-recursive.mp4", 0, 10, ""),
    Scene("Aldehyde dehydrogenase · ALDH", "Wet-lab closed loop", 5, "aldh-sequence.mp4", 0, 8, ""),
    Scene("ALDH · Cumulative top 10 through each round", "Wet-lab closed loop", 8, "aldh-recursive.mp4", 0, 10, ""),
    Scene("Closed-loop protein evolution", "ProTEUS", 13, "world-model-outro.mp4", 0, 16, ""),
)
DURATION = sum(scene.seconds for scene in SCENES)
assert DURATION <= 140



def find_font(explicit: str | None, bold: bool) -> Path:
    filename = "Georgia Bold.ttf" if bold else "Georgia.ttf"
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path("/System/Library/Fonts/Supplemental") / filename,
        Path("/Library/Fonts") / filename,
        Path.home() / ".fonts" / filename,
        Path("C:/Windows/Fonts") / ("georgiab.ttf" if bold else "georgia.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Georgia not found; pass --font-{'bold' if bold else 'regular'}.")


def timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def probe(path: Path, ffprobe: str) -> dict:
    result = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0",
                                      "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)])
    return json.loads(result)


def geometry(scene: Scene, info: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    width, height = info["streams"][0]["width"], info["streams"][0]["height"]
    x, y, w, h = 0, 0, width, height
    left, top, right, bottom = (0, 0, 1920, 1080) if scene.asset in SELF_TITLED else VIEWPORT
    scale = min((right - left) / w, (bottom - top) / h)
    dw, dh = round(w * scale / 2) * 2, round(h * scale / 2) * 2
    dx, dy = (WIDTH - dw) // 2, top + (bottom - top - dh) // 2
    return (x, y, w, h), (dx, dy, dw, dh)


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def make_overlay(scene: Scene, regular: Path, bold: Path) -> Image.Image:
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = lambda size, strong=False: ImageFont.truetype(str(bold if strong else regular), size)
    if scene.asset not in SELF_TITLED:
        if scene.title != 'ProTEUS':
            draw.text((MARGIN, 33), "ProTEUS", font=font(25, True), fill=BLUE)
        draw.text((WIDTH - MARGIN, 33), scene.group, font=font(23), fill=MUTED, anchor="rt")
        heading = font(47, True)
        if draw.textlength(scene.title, font=heading) > WIDTH - MARGIN * 2:
            heading = font(42, True)
        draw.text((MARGIN, 82), scene.title, font=heading, fill=INK)
        draw.line((MARGIN, 150, WIDTH - MARGIN, 150), fill=RULE, width=2)
    if scene.caption:
        draw.line((MARGIN, 968, WIDTH - MARGIN, 968), fill=RULE, width=2)
    if scene.asset in ('sacb-structure.mp4', 'aldh-structure.mp4'):
        sacb = scene.asset.startswith('sacb')
        draw.text((130, 350), 'WT reference', font=font(25, True), fill=INK)
        draw.text((130, 389), '1OYG' if sacb else 'AF-P46368-F1', font=font(23), fill=MUTED)
        labels = [('Selected sites', '#C85F6B'), ('Catalytic residues', '#6B2233'),
                  ('Sucrose' if sacb else 'NAD-binding region', '#F2B632' if sacb else '#D9962F')]
        for index, (label, color) in enumerate(labels):
            y = 470 + index * 57
            draw.ellipse((132, y+7, 146, y+21), fill=color)
            draw.text((163, y), label, font=font(23), fill=INK)
    elif scene.asset == 'attention-structure.mp4':
        draw.text((1450, 400), 'Mean attention', font=font(25), fill=INK)
        stops = [(253,241,240), (236,140,147), (199,52,70), (142,23,48)]
        for x in range(250):
            scaled = x / 249 * 3
            index = min(int(scaled), 2)
            portion = scaled - index
            color = tuple(round(a*(1-portion)+b*portion) for a,b in zip(stops[index],stops[index+1]))
            draw.line((1450+x,452,1450+x,474),fill=color,width=1)
        draw.text((1450,487), '0', font=font(22), fill=MUTED, anchor='lt')
        draw.text((1700,487), '0.22', font=font(22), fill=MUTED, anchor='rt')
    caption_font = font(27)
    lines = wrap(draw, scene.caption, caption_font, WIDTH - MARGIN * 2)
    if len(lines) > 2:
        raise ValueError(f"Caption too long: {scene.title}")
    for index, line in enumerate(lines):
        draw.text((MARGIN, 988 + index * 34), line, font=caption_font, fill=INK)
    if scene.reference:
        if len(lines) != 1:
            raise ValueError(f"Caption with reference must fit one line: {scene.title}")
        draw.text((MARGIN, 1029), scene.reference, font=font(22), fill=MUTED)
    return overlay


def make_preview(scene: Scene, assets: Path, overlay: Image.Image, crop: tuple,
                 destination: tuple, ffmpeg: str) -> Image.Image:
    time = scene.source_start + scene.source_length * (0.80 if scene.asset == "world-model-intro.mp4" else 0.68)
    raw = subprocess.check_output([ffmpeg, "-v", "error", "-ss", str(time), "-i", str(assets / scene.asset),
                                   "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
    source = Image.open(io.BytesIO(raw)).convert("RGB")
    x, y, w, h = crop
    dx, dy, dw, dh = destination
    source = source.crop((x, y, x + w, y + h)).resize((dw, dh), Image.Resampling.LANCZOS)
    frame = Image.new("RGB", (WIDTH, HEIGHT), "white")
    frame.paste(source, (dx, dy))
    frame.paste(overlay, (0, 0), overlay)
    return frame


def write_text_tracks(captions: Path, chapters: Path) -> None:
    vtt = ["WEBVTT", "", "NOTE Newly rendered native molecular scenes and data-driven diagrams. No narration.", ""]
    metadata = [";FFMETADATA1", "title=ProTEUS: A mutation-resolved world model for closed-loop protein evolution",
                "comment=Paper-source molecular rendering; edit recipe: tools/render_project_film.py"]
    start = 0.0
    for index, scene in enumerate(SCENES, 1):
        end = start + scene.seconds
        vtt += [str(index), f"{timestamp(start)} --> {timestamp(end)}", scene.caption or scene.title, ""]
        metadata += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(start * 1000)}", f"END={round(end * 1000)}",
                     f"title={scene.title}"]
        start = end
    assert start == DURATION
    captions.write_text("\n".join(vtt), encoding="utf-8")
    chapters.write_text("\n".join(metadata) + "\n", encoding="utf-8")


def encode_scene(scene: Scene, source: Path, overlay: Path, crop: tuple, destination: tuple,
                 output: Path, ffmpeg: str, first: bool, last: bool) -> None:
    x, y, w, h = crop
    dx, dy, dw, dh = destination
    speed = (scene.seconds - scene.end_hold) / scene.source_length
    filters = (f"[0:v]trim=duration={scene.source_length},setpts=(PTS-STARTPTS)*{speed:.12f},fps={FPS},"
               f"crop={w}:{h}:{x}:{y},scale={dw}:{dh}:flags=lanczos,"
               f"pad={WIDTH}:{HEIGHT}:{dx}:{dy}:color=white,tpad=stop_mode=clone:stop_duration={scene.end_hold + 1},"
               f"trim=duration={scene.seconds}[footage];"
               "[footage][1:v]overlay=0:0:format=auto,format=yuv420p,setsar=1")
    # Brief fades keep the white composition continuous across scene changes.
    if not first:
        filters += ",fade=t=in:st=0:d=0.16:color=white"
    if not last:
        filters += f",fade=t=out:st={scene.seconds - 0.16:.3f}:d=0.16:color=white"
    filters += "[out]"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", str(scene.source_start), "-i", str(source),
                    "-loop", "1", "-i", str(overlay), "-filter_complex", filters, "-map", "[out]",
                    "-frames:v", str(round(scene.seconds * FPS)), "-an", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "4", str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets", type=Path, default=ROOT / "docs/assets/video")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/proteus-overview.mp4")
    parser.add_argument("--poster", type=Path, default=ROOT / "docs/assets/proteus-overview-poster.webp")
    parser.add_argument("--captions", type=Path, default=ROOT / "docs/assets/proteus-overview.vtt")
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--font-regular")
    parser.add_argument("--font-bold")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and ffprobe are required")
    regular, bold = find_font(args.font_regular, False), find_font(args.font_bold, True)
    sources = {s.asset for s in SCENES}
    missing = [s for s in sources if not (args.assets / s).is_file()]
    if missing:
        parser.error("Missing rendered scenes: " + ", ".join(sorted(missing)))
    information = {s: probe(args.assets / s, ffprobe) for s in sources}
    for path in (args.output, args.poster, args.captions):
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="proteus-paper-style-") as temporary:
        work = Path(temporary)
        chapters = work / "chapters.ffmeta"
        write_text_tracks(args.captions, chapters)
        segments = []
        start = 0.0
        for index, scene in enumerate(SCENES):
            source = args.assets / scene.asset
            if scene.source_start + scene.source_length > float(information[scene.asset]["format"]["duration"]) + 0.05:
                raise ValueError(f"Source interval exceeds clip duration: {scene.title}")
            crop, destination = geometry(scene, information[scene.asset])
            overlay = make_overlay(scene, regular, bold)
            overlay_path = work / f"overlay-{index:02d}.png"
            overlay.save(overlay_path)
            if index == 0 or args.preview_only or args.frames_dir:
                frame = make_preview(scene, args.assets, overlay, crop, destination, ffmpeg)
                if index == 0:
                    frame.save(args.poster, "WEBP", quality=90, method=6)
                if args.frames_dir:
                    frame.save(args.frames_dir / f"scene-{index + 1:02d}.png")
            print(f"{timestamp(start)}–{timestamp(start + scene.seconds)} {scene.title} | {scene.asset} @ {scene.source_start}s", flush=True)
            start += scene.seconds
            if not args.preview_only:
                segment = work / f"segment-{index:02d}.mp4"
                encode_scene(scene, source, overlay_path, crop, destination, segment, ffmpeg,
                             first=index == 0, last=index == len(SCENES) - 1)
                segments.append(segment)
        if not args.preview_only:
            listing = work / "segments.txt"
            listing.write_text("".join(f"file '{p.name}'\n" for p in segments), encoding="utf-8")
            subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(listing),
                            "-i", str(chapters), "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1",
                            "-c:v", "copy", "-an", "-movflags", "+faststart", str(work / 'complete.mp4')], check=True)
            final_info = probe(work / 'complete.mp4', ffprobe)
            if abs(float(final_info['format']['duration']) - DURATION) > .05:
                raise ValueError(f'Encoded film must be exactly {DURATION} seconds.')
            shutil.move(work / 'complete.mp4', args.output)
            print(f"Saved {args.output} ({args.output.stat().st_size / 1048576:.2f} MiB)", flush=True)
    for asset in sorted(sources):
        print(f"Source sha256 {asset}: {hashlib.sha256((args.assets / asset).read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

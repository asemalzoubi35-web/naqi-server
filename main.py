from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


# ═══════════════════════════════════════════════════════════
# NAQI PRO 3.0 FAST
# ═══════════════════════════════════════════════════════════

APP_DIR = Path(__file__).resolve().parent

WORK_DIR = Path(
    os.getenv(
        "NAQI_WORK_DIR",
        str(APP_DIR / "data"),
    )
)

WORK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ═══════════════════════════════════════════════════════════
# UPLOAD / API
# ═══════════════════════════════════════════════════════════

MAX_UPLOAD = int(
    os.getenv(
        "NAQI_MAX_UPLOAD_BYTES",
        str(500 * 1024 * 1024),
    )
)

API_KEY = os.getenv(
    "NAQI_API_KEY",
    "",
)


# ═══════════════════════════════════════════════════════════
# FAST DEMUCS CONFIG
# ═══════════════════════════════════════════════════════════

# Fast default:
# htdemucs is substantially more practical than htdemucs_ft
# when running separation on CPU.

MODEL = os.getenv(
    "NAQI_DEMUCS_MODEL",
    "htdemucs",
)

DEVICE = os.getenv(
    "NAQI_DEMUCS_DEVICE",
    "cpu",
)

DEMUCS_SHIFTS = int(
    os.getenv(
        "NAQI_DEMUCS_SHIFTS",
        "0",
    )
)

# Lower overlap reduces repeated processing between segments.
# 0.25 is the normal Demucs default.
# 0.10 is intentionally selected for FAST mode.
DEMUCS_OVERLAP = float(
    os.getenv(
        "NAQI_DEMUCS_OVERLAP",
        "0.10",
    )
)

# CPU threads.
CPU_COUNT = os.cpu_count() or 4

DEFAULT_TORCH_THREADS = max(
    1,
    CPU_COUNT - 1,
)

TORCH_THREADS = int(
    os.getenv(
        "NAQI_TORCH_THREADS",
        str(DEFAULT_TORCH_THREADS),
    )
)

TORCH_INTEROP_THREADS = int(
    os.getenv(
        "NAQI_TORCH_INTEROP_THREADS",
        str(max(1, min(4, TORCH_THREADS))),
    )
)

# Help PyTorch / BLAS use the available CPU efficiently.
os.environ.setdefault(
    "OMP_NUM_THREADS",
    str(TORCH_THREADS),
)

os.environ.setdefault(
    "MKL_NUM_THREADS",
    str(TORCH_THREADS),
)

os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    str(TORCH_THREADS),
)

os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    str(TORCH_THREADS),
)


# ═══════════════════════════════════════════════════════════
# FFMPEG
# ═══════════════════════════════════════════════════════════

FFMPEG = os.getenv(
    "NAQI_FFMPEG",
    r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe",
)

FFPROBE = os.getenv(
    "NAQI_FFPROBE",
    str(Path(FFMPEG).with_name("ffprobe.exe")),
)

if not Path(FFMPEG).exists():
    raise RuntimeError(
        f"ffmpeg not found at {FFMPEG}"
    )

if not Path(FFPROBE).exists():
    raise RuntimeError(
        f"ffprobe not found at {FFPROBE}"
    )


print(
    "═══════════════════════════════════════════════════════",
    flush=True,
)

print(
    "🚀 NAQI PRO 3.0 FAST",
    flush=True,
)

print(
    f"FFmpeg: {FFMPEG}",
    flush=True,
)

print(
    f"FFprobe: {FFPROBE}",
    flush=True,
)

print(
    f"Python: {sys.executable}",
    flush=True,
)

print(
    f"CPU cores detected: {CPU_COUNT}",
    flush=True,
)

print(
    f"PyTorch threads: {TORCH_THREADS}",
    flush=True,
)

print(
    f"PyTorch interop threads: {TORCH_INTEROP_THREADS}",
    flush=True,
)

print(
    f"Demucs model: {MODEL}",
    flush=True,
)

print(
    f"Demucs device: {DEVICE}",
    flush=True,
)

print(
    f"Demucs shifts: {DEMUCS_SHIFTS}",
    flush=True,
)

print(
    f"Demucs overlap: {DEMUCS_OVERLAP}",
    flush=True,
)

print(
    "═══════════════════════════════════════════════════════",
    flush=True,
)


# ═══════════════════════════════════════════════════════════
# GENERAL SETTINGS
# ═══════════════════════════════════════════════════════════

KEEP_FILES = True

STREAM_SAMPLE_RATE = 44100
STREAM_CHANNELS = 2
STREAM_CHUNK_SECONDS = 3

STREAM_CHUNK_BYTES = (
    STREAM_SAMPLE_RATE
    * STREAM_CHANNELS
    * 2
    * STREAM_CHUNK_SECONDS
)


# ═══════════════════════════════════════════════════════════
# URL RULES
# ═══════════════════════════════════════════════════════════

BLOCKED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "tiktok.com",
    "www.tiktok.com",
    "instagram.com",
    "www.instagram.com",
    "facebook.com",
    "www.facebook.com",
    "fb.watch",
    "twitter.com",
    "x.com",
    "vimeo.com",
    "www.vimeo.com",
    "dailymotion.com",
    "www.dailymotion.com",
    "netflix.com",
    "spotify.com",
    "soundcloud.com",
    "music.apple.com",
    "podcasts.apple.com",
}

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".mkv",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
}


# ═══════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="NAQI PRO 3.0 FAST Separation API",
    version="3.0.0-fast",
)


# ═══════════════════════════════════════════════════════════
# BASIC HELPERS
# ═══════════════════════════════════════════════════════════

def _clamp(value: float) -> float:
    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def _timer() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


def _run(
    command: list[str],
    cwd: Path | None = None,
) -> None:

    print(
        "\n▶ Running:",
        " ".join(command),
        flush=True,
    )

    started = _timer()

    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.stdout:
        print(
            proc.stdout[-15000:],
            flush=True,
        )

    if proc.returncode != 0:
        tail = (
            proc.stdout[-15000:]
            if proc.stdout
            else "(no output)"
        )

        raise RuntimeError(
            "Command failed:\n\n"
            + " ".join(command)
            + "\n\n"
            + tail
        )

    print(
        f"✅ Command finished in {_elapsed(started)}",
        flush=True,
    )


def _validate_format(
    format: str,
) -> str:

    format = format.lower().strip()

    if format not in {
        "flac",
        "wav",
        "mp4",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "format must be flac, wav, or mp4"
            ),
        )

    return format


def _cleanup_task(
    job: Path,
) -> BackgroundTask | None:

    if KEEP_FILES:
        return None

    return BackgroundTask(
        shutil.rmtree,
        job,
        ignore_errors=True,
    )


# ═══════════════════════════════════════════════════════════
# FFMPEG / MEDIA HELPERS
# ═══════════════════════════════════════════════════════════

def _has_audio_stream(
    file_path: Path,
) -> bool:

    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return bool(
        proc.stdout.strip()
    )


def _has_video_stream(
    file_path: Path,
) -> bool:

    command = [
        FFPROBE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return bool(
        proc.stdout.strip()
    )


def _get_duration(
    file_path: Path,
) -> float:

    command = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        return float(
            proc.stdout.strip()
        )
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
# VOICE PROCESSING
# ═══════════════════════════════════════════════════════════

def _build_voice_filter(
    noise: float,
) -> str:

    noise = _clamp(noise)

    nr = 3.0 + (
        18.0 * noise
    )

    if noise <= 0.001:
        noise_filter = "anull"
    else:
        noise_filter = (
            f"afftdn="
            f"nr={nr:.2f}:"
            f"nf=-25:"
            f"tn=1"
        )

    return (
        f"{noise_filter},"
        "highpass=f=70,"
        "equalizer="
        "f=250:"
        "width_type=h:"
        "width=180:"
        "g=-1.2,"
        "equalizer="
        "f=1800:"
        "width_type=h:"
        "width=900:"
        "g=1.8,"
        "equalizer="
        "f=3500:"
        "width_type=h:"
        "width=1400:"
        "g=1.2,"
        "acompressor="
        "threshold=-20dB:"
        "ratio=2.2:"
        "attack=8:"
        "release=120:"
        "makeup=1.5,"
        "alimiter="
        "limit=0.95"
    )


def _build_mix_filter(
    voice: float,
    music: float,
    noise: float,
    other: float,
) -> str:

    voice = _clamp(voice)
    music = _clamp(music)
    noise = _clamp(noise)
    other = _clamp(other)

    vocal_chain = _build_voice_filter(
        noise
    )

    music_gain = min(
        1.0,
        (
            0.75 * music
            + 0.25 * other
        ),
    )

    return (
        "[0:a]"
        f"{vocal_chain},"
        f"volume={voice:.4f}"
        "[v];"
        "[1:a]"
        f"volume={music_gain:.4f}"
        "[m];"
        "[v][m]"
        "amix="
        "inputs=2:"
        "duration=longest:"
        "dropout_transition=0:"
        "normalize=0,"
        "loudnorm="
        "I=-16:"
        "TP=-1.5:"
        "LRA=11"
    )


def _build_fast_voice_only_filter(
    voice: float,
    noise: float,
) -> str:

    voice = _clamp(voice)

    return (
        _build_voice_filter(noise)
        + ","
        + f"volume={voice:.4f},"
        "loudnorm="
        "I=-16:"
        "TP=-1.5:"
        "LRA=11"
    )


# ═══════════════════════════════════════════════════════════
# DEMUCS
# ═══════════════════════════════════════════════════════════

def _find_stem(
    root: Path,
    name: str,
) -> Path:

    matches = list(
        root.rglob(name)
    )

    if not matches:
        raise RuntimeError(
            f"Demucs did not produce {name}"
        )

    return matches[0]


def _run_demucs(
    input_wav: Path,
    separated: Path,
) -> tuple[Path, Path]:

    separated.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "🤖 FAST Demucs separation starting...",
        flush=True,
    )

    print(
        f"   Model   : {MODEL}",
        flush=True,
    )

    print(
        f"   Device  : {DEVICE}",
        flush=True,
    )

    print(
        f"   Shifts  : {DEMUCS_SHIFTS}",
        flush=True,
    )

    print(
        f"   Overlap : {DEMUCS_OVERLAP}",
        flush=True,
    )

    print(
        f"   Threads : {TORCH_THREADS}",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "demucs",
        "-d",
        DEVICE,
        "-n",
        MODEL,
        "--two-stems=vocals",
        "--shifts",
        str(DEMUCS_SHIFTS),
        "--overlap",
        str(DEMUCS_OVERLAP),
        "-o",
        str(separated),
        str(input_wav),
    ]

    started = _timer()

    _run(
        command
    )

    print(
        f"🤖 Demucs finished in {_elapsed(started)}",
        flush=True,
    )

    vocals = _find_stem(
        separated,
        "vocals.wav",
    )

    accompaniment = _find_stem(
        separated,
        "no_vocals.wav",
    )

    return (
        vocals,
        accompaniment,
    )


# ═══════════════════════════════════════════════════════════
# AUDIO EXTRACTION
# ═══════════════════════════════════════════════════════════

def _extract_audio(
    input_path: Path,
    output_wav: Path,
) -> None:

    print(
        "🎧 Extracting audio...",
        flush=True,
    )

    if not _has_audio_stream(
        input_path
    ):
        raise RuntimeError(
            "The input video/audio file "
            "does not contain an audio stream."
        )

    started = _timer()

    _run(
        [
            FFMPEG,
            "-y",
            "-threads",
            "0",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )

    print(
        f"🎧 Audio extraction finished in {_elapsed(started)}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════
# COMPLETE AUDIO SEPARATION
# ═══════════════════════════════════════════════════════════

def _run_separation_pipeline(
    job: Path,
    input_path: Path,
    voice: float,
    music: float,
    noise: float,
    other: float,
    format: str,
    prefer_voice_only: bool = False,
) -> Path:

    total_started = _timer()

    input_wav = (
        job / "input_44k.wav"
    )

    _extract_audio(
        input_path,
        input_wav,
    )

    separated = (
        job / "separated"
    )

    vocals, accompaniment = _run_demucs(
        input_wav,
        separated,
    )

    print(
        f"🎤 Vocals: {vocals}",
        flush=True,
    )

    print(
        f"🎵 Accompaniment: {accompaniment}",
        flush=True,
    )

    # ═══════════════════════════════════════════════════════
    # FAST PATH
    #
    # When music and other are zero, there is no reason
    # to mix the accompaniment back in.
    # ═══════════════════════════════════════════════════════

    voice_only = (
        prefer_voice_only
        and music <= 0.001
        and other <= 0.001
    )

    if voice_only:

        print(
            "⚡ FAST VOICE-ONLY PATH",
            flush=True,
        )

        output = (
            job / "naqi_separated.flac"
        )

        voice_filter = (
            _build_fast_voice_only_filter(
                voice,
                noise,
            )
        )

        print(
            "✨ Enhancing vocals directly...",
            flush=True,
        )

        _run(
            [
                FFMPEG,
                "-y",
                "-threads",
                "0",
                "-i",
                str(vocals),
                "-filter:a",
                voice_filter,
                "-c:a",
                "flac",
                "-compression_level",
                "1",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(output),
            ]
        )

        print(
            f"✅ Fast clean audio created: {output}",
            flush=True,
        )

        print(
            f"⏱ Total separation pipeline: "
            f"{_elapsed(total_started)}",
            flush=True,
        )

        return output

    # ═══════════════════════════════════════════════════════
    # NORMAL MIX PATH
    # ═══════════════════════════════════════════════════════

    output_ext = (
        "flac"
        if format == "flac"
        else "wav"
    )

    output = (
        job
        / f"naqi_separated.{output_ext}"
    )

    mix_filter = _build_mix_filter(
        voice,
        music,
        noise,
        other,
    )

    if format == "flac":
        codec = [
            "-c:a",
            "flac",
            "-compression_level",
            "1",
        ]
    else:
        codec = [
            "-c:a",
            "pcm_s16le",
        ]

    print(
        "✨ Enhancing separated voice...",
        flush=True,
    )

    _run(
        [
            FFMPEG,
            "-y",
            "-threads",
            "0",
            "-i",
            str(vocals),
            "-i",
            str(accompaniment),
            "-filter_complex",
            mix_filter,
            *codec,
            "-ar",
            "48000",
            "-ac",
            "2",
            str(output),
        ]
    )

    print(
        f"✅ Clean audio created: {output}",
        flush=True,
    )

    print(
        f"⏱ Total separation pipeline: "
        f"{_elapsed(total_started)}",
        flush=True,
    )

    return output


# ═══════════════════════════════════════════════════════════
# YOUTUBE DOWNLOAD
# ═══════════════════════════════════════════════════════════

def _download_youtube_media(
    job: Path,
    url: str,
) -> tuple[Path, Path]:

    video_template = str(
        job / "video.%(ext)s"
    )

    audio_template = str(
        job / "audio.%(ext)s"
    )

    video_format = (
        "bestvideo[ext=mp4][vcodec^=avc1]/"
        "bestvideo[ext=mp4]/"
        "bestvideo"
    )

    audio_format = (
        "bestaudio[ext=m4a]/"
        "bestaudio"
    )

    # ═══════════════════════════════════════════════════════
    # VIDEO
    # ═══════════════════════════════════════════════════════

    print(
        "⬇ Downloading YouTube VIDEO...",
        flush=True,
    )

    _run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            video_format,
            "-o",
            video_template,
            "--no-playlist",
            "--no-warnings",
            "--no-part",
            "--ffmpeg-location",
            str(Path(FFMPEG).parent),
            url,
        ]
    )

    # ═══════════════════════════════════════════════════════
    # AUDIO
    # ═══════════════════════════════════════════════════════

    print(
        "⬇ Downloading YouTube AUDIO...",
        flush=True,
    )

    _run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            audio_format,
            "-o",
            audio_template,
            "--no-playlist",
            "--no-warnings",
            "--no-part",
            "--ffmpeg-location",
            str(Path(FFMPEG).parent),
            url,
        ]
    )

    video_files = [
        f
        for f in job.iterdir()
        if (
            f.is_file()
            and f.name.startswith("video.")
        )
    ]

    audio_files = [
        f
        for f in job.iterdir()
        if (
            f.is_file()
            and f.name.startswith("audio.")
        )
    ]

    if not video_files:
        raise RuntimeError(
            "yt-dlp did not produce a video file."
        )

    if not audio_files:
        raise RuntimeError(
            "yt-dlp did not produce an audio file."
        )

    video_file = video_files[0]
    audio_file = audio_files[0]

    print(
        f"✅ YouTube video: "
        f"{video_file.name} "
        f"({video_file.stat().st_size} bytes)",
        flush=True,
    )

    print(
        f"✅ YouTube audio: "
        f"{audio_file.name} "
        f"({audio_file.stat().st_size} bytes)",
        flush=True,
    )

    if not _has_video_stream(
        video_file
    ):
        raise RuntimeError(
            "Downloaded YouTube video "
            "does not contain a video stream."
        )

    if not _has_audio_stream(
        audio_file
    ):
        raise RuntimeError(
            "Downloaded YouTube audio "
            "does not contain an audio stream."
        )

    # ═══════════════════════════════════════════════════════
    # MERGE
    # ═══════════════════════════════════════════════════════

    source_video = (
        job / "source.mp4"
    )

    print(
        "🔗 Merging YouTube video + audio...",
        flush=True,
    )

    _run(
        [
            FFMPEG,
            "-y",
            "-threads",
            "0",
            "-i",
            str(video_file),
            "-i",
            str(audio_file),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(source_video),
        ]
    )

    if not source_video.exists():
        raise RuntimeError(
            "FFmpeg did not create source.mp4."
        )

    if not _has_video_stream(
        source_video
    ):
        raise RuntimeError(
            "Merged source.mp4 has no video stream."
        )

    if not _has_audio_stream(
        source_video
    ):
        raise RuntimeError(
            "Merged source.mp4 has no audio stream."
        )

    print(
        f"✅ Merged source video: "
        f"{source_video}",
        flush=True,
    )

    return (
        source_video,
        audio_file,
    )


# ═══════════════════════════════════════════════════════════
# FINAL CLEAN VIDEO
# ═══════════════════════════════════════════════════════════

def _build_clean_video(
    job: Path,
    original_video: Path,
    cleaned_audio: Path,
) -> Path:

    output = (
        job / "naqi_cleaned.mp4"
    )

    print(
        "🎬 Replacing original audio "
        "with cleaned voice...",
        flush=True,
    )

    _run(
        [
            FFMPEG,
            "-y",
            "-threads",
            "0",
            "-i",
            str(original_video),
            "-i",
            str(cleaned_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )

    if not output.exists():
        raise RuntimeError(
            "FFmpeg did not create "
            "naqi_cleaned.mp4."
        )

    if not _has_video_stream(
        output
    ):
        raise RuntimeError(
            "Final MP4 has no video stream."
        )

    if not _has_audio_stream(
        output
    ):
        raise RuntimeError(
            "Final MP4 has no audio stream."
        )

    print(
        f"🎉 FINAL VIDEO: {output}",
        flush=True,
    )

    return output


# ═══════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:

    return {
        "ok": True,
        "model": MODEL,
        "device": DEVICE,
        "demucs_shifts": DEMUCS_SHIFTS,
        "demucs_overlap": DEMUCS_OVERLAP,
        "torch_threads": TORCH_THREADS,
        "cpu_cores": CPU_COUNT,
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "keep_files": KEEP_FILES,
        "stream_chunk_seconds": STREAM_CHUNK_SECONDS,
        "fast_mode": True,
        "version": "3.0.0-fast",
    }


# ═══════════════════════════════════════════════════════════
# POST /v1/separate
# LOCAL FILE
# ═══════════════════════════════════════════════════════════

@app.post("/v1/separate")
async def separate(
    file: UploadFile = File(...),
    voice: float = Form(1.0),
    music: float = Form(0.0),
    noise: float = Form(0.6),
    other: float = Form(0.0),
    format: str = Form("flac"),
    x_naqi_api_key: str | None = Header(
        default=None,
    ),
):

    if (
        API_KEY
        and x_naqi_api_key != API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    format = _validate_format(
        format
    )

    if format == "mp4":
        raise HTTPException(
            status_code=400,
            detail=(
                "MP4 output is only supported "
                "by the YouTube video endpoint."
            ),
        )

    voice = _clamp(voice)
    music = _clamp(music)
    noise = _clamp(noise)
    other = _clamp(other)

    job = (
        WORK_DIR
        / uuid.uuid4().hex
    )

    job.mkdir(
        parents=True,
        exist_ok=False,
    )

    input_path = (
        job
        / Path(
            file.filename or "input"
        ).name
    )

    try:

        size = 0

        with input_path.open(
            "wb"
        ) as out:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File is too large"
                        ),
                    )

                out.write(chunk)

        output = (
            _run_separation_pipeline(
                job,
                input_path,
                voice,
                music,
                noise,
                other,
                format,
                prefer_voice_only=True,
            )
        )

        return FileResponse(
            output,
            media_type=(
                "audio/flac"
                if format == "flac"
                else "audio/wav"
            ),
            filename=output.name,
            background=_cleanup_task(
                job
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            traceback.format_exc(),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Separation failed: "
                f"{exc}"
            ),
        ) from exc


# ═══════════════════════════════════════════════════════════
# POST /v1/separate-url
# DIRECT MEDIA URL
# ═══════════════════════════════════════════════════════════

@app.post("/v1/separate-url")
async def separate_url(
    url: str = Form(...),
    voice: float = Form(1.0),
    music: float = Form(0.0),
    noise: float = Form(0.6),
    other: float = Form(0.0),
    format: str = Form("flac"),
    x_naqi_api_key: str | None = Header(
        default=None,
    ),
):

    if (
        API_KEY
        and x_naqi_api_key != API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    parsed = urllib.parse.urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only HTTP and HTTPS URLs "
                "are supported."
            ),
        )

    host = (
        parsed.hostname or ""
    ).lower()

    if (
        host in BLOCKED_HOSTS
        or any(
            host.endswith(
                f".{b}"
            )
            for b in BLOCKED_HOSTS
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "This platform is not supported "
                "on the direct URL endpoint. "
                "Use the YouTube endpoint or "
                "upload the video directly."
            ),
        )

    ext = Path(
        parsed.path
    ).suffix.lower()

    if (
        ext
        and ext not in ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension "
                f"'{ext}' is not supported."
            ),
        )

    format = _validate_format(
        format
    )

    if format == "mp4":
        raise HTTPException(
            status_code=400,
            detail=(
                "MP4 output is not enabled "
                "for direct URLs."
            ),
        )

    job = (
        WORK_DIR
        / uuid.uuid4().hex
    )

    job.mkdir(
        parents=True,
        exist_ok=False,
    )

    input_path = (
        job
        / f"downloaded"
        f"{ext or '.mp4'}"
    )

    try:

        print(
            f"⬇ Downloading: {url}",
            flush=True,
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NAQI/3.0",
            },
        )

        with urllib.request.urlopen(
            req,
            timeout=60,
        ) as response:

            total = 0

            with input_path.open(
                "wb"
            ) as out:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    total += len(chunk)

                    if total > MAX_UPLOAD:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "File too large"
                            ),
                        )

                    out.write(chunk)

        print(
            f"✅ Downloaded "
            f"{total} bytes",
            flush=True,
        )

        output = (
            _run_separation_pipeline(
                job,
                input_path,
                _clamp(voice),
                _clamp(music),
                _clamp(noise),
                _clamp(other),
                format,
                prefer_voice_only=True,
            )
        )

        return FileResponse(
            output,
            media_type=(
                "audio/flac"
                if format == "flac"
                else "audio/wav"
            ),
            filename=output.name,
            background=_cleanup_task(
                job
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            traceback.format_exc(),
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"URL separation failed: "
                f"{exc}"
            ),
        ) from exc


# ═══════════════════════════════════════════════════════════
# POST /v1/separate-youtube
# YOUTUBE → CLEAN MP4
# ═══════════════════════════════════════════════════════════

@app.post("/v1/separate-youtube")
async def separate_youtube(
    url: str = Form(...),
    voice: float = Form(1.0),
    music: float = Form(0.0),
    noise: float = Form(0.6),
    other: float = Form(0.0),
    format: str = Form("mp4"),
    x_naqi_api_key: str | None = Header(
        default=None,
    ),
):

    total_started = _timer()

    if (
        API_KEY
        and x_naqi_api_key != API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    parsed = urllib.parse.urlparse(
        url
    )

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only HTTP and HTTPS URLs "
                "are supported."
            ),
        )

    host = (
        parsed.hostname or ""
    ).lower()

    if host not in YOUTUBE_HOSTS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only YouTube links are "
                "supported on this endpoint."
            ),
        )

    format = _validate_format(
        format
    )

    voice = _clamp(voice)
    music = _clamp(music)
    noise = _clamp(noise)
    other = _clamp(other)

    job = (
        WORK_DIR
        / uuid.uuid4().hex
    )

    job.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:

        # ═══════════════════════════════════════════════════
        # 1 — DOWNLOAD
        # ═══════════════════════════════════════════════════

        download_started = _timer()

        source_video, _ = (
            _download_youtube_media(
                job,
                url,
            )
        )

        print(
            f"⬇ YouTube download + merge: "
            f"{_elapsed(download_started)}",
            flush=True,
        )

        if (
            source_video.stat().st_size
            > MAX_UPLOAD
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "Downloaded YouTube video "
                    "is too large."
                ),
            )

        duration = _get_duration(
            source_video
        )

        print(
            f"🎬 Source duration: "
            f"{duration:.2f} seconds",
            flush=True,
        )

        # ═══════════════════════════════════════════════════
        # 2 — DEMUCS
        # ═══════════════════════════════════════════════════

        print(
            "⚡ FAST MODE: "
            "voice isolation enabled",
            flush=True,
        )

        cleaned_audio = (
            _run_separation_pipeline(
                job,
                source_video,
                voice,
                music,
                noise,
                other,
                "flac",
                prefer_voice_only=True,
            )
        )

        # ═══════════════════════════════════════════════════
        # 3 — FINAL MP4
        # ═══════════════════════════════════════════════════

        if format == "mp4":

            final_started = _timer()

            cleaned_video = (
                _build_clean_video(
                    job,
                    source_video,
                    cleaned_audio,
                )
            )

            print(
                f"🎬 Final video mux: "
                f"{_elapsed(final_started)}",
                flush=True,
            )

            print(
                "════════════════════════════════════════════",
                flush=True,
            )

            print(
                f"🎉 NAQI COMPLETE",
                flush=True,
            )

            print(
                f"⏱ TOTAL TIME: "
                f"{_elapsed(total_started)}",
                flush=True,
            )

            print(
                f"📁 OUTPUT: {cleaned_video}",
                flush=True,
            )

            print(
                "════════════════════════════════════════════",
                flush=True,
            )

            return FileResponse(
                cleaned_video,
                media_type="video/mp4",
                filename=cleaned_video.name,
                background=_cleanup_task(
                    job
                ),
            )

        # ═══════════════════════════════════════════════════
        # AUDIO ONLY
        # ═══════════════════════════════════════════════════

        if format == "flac":

            return FileResponse(
                cleaned_audio,
                media_type="audio/flac",
                filename=cleaned_audio.name,
                background=_cleanup_task(
                    job
                ),
            )

        wav_output = (
            job
            / "naqi_separated.wav"
        )

        _run(
            [
                FFMPEG,
                "-y",
                "-threads",
                "0",
                "-i",
                str(cleaned_audio),
                "-c:a",
                "pcm_s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(wav_output),
            ]
        )

        return FileResponse(
            wav_output,
            media_type="audio/wav",
            filename=wav_output.name,
            background=_cleanup_task(
                job
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "\n"
            + "=" * 80,
            flush=True,
        )

        print(
            f"❌ YOUTUBE ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

        print(
            traceback.format_exc(),
            flush=True,
        )

        print(
            "=" * 80,
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"YouTube separation failed: "
                f"{exc}"
            ),
        ) from exc


# ═══════════════════════════════════════════════════════════
# WEBSOCKET
# ═══════════════════════════════════════════════════════════

@app.websocket(
    "/ws/separate-stream"
)
async def separate_stream(
    websocket: WebSocket,
):

    await websocket.accept()

    print(
        "🔌 WebSocket connected",
        flush=True,
    )

    buffer = bytearray()
    chunk_index = 0

    try:

        while True:

            data = (
                await websocket.receive_bytes()
            )

            buffer.extend(data)

            while (
                len(buffer)
                >= STREAM_CHUNK_BYTES
            ):

                chunk = bytes(
                    buffer[
                        :STREAM_CHUNK_BYTES
                    ]
                )

                buffer = buffer[
                    STREAM_CHUNK_BYTES:
                ]

                chunk_index += 1

                print(
                    f"🎵 Processing "
                    f"chunk #{chunk_index}",
                    flush=True,
                )

                try:

                    processed = (
                        await _process_stream_chunk(
                            chunk
                        )
                    )

                    await websocket.send_bytes(
                        processed
                    )

                    print(
                        f"✅ Sent chunk "
                        f"#{chunk_index}",
                        flush=True,
                    )

                except Exception as exc:

                    print(
                        f"❌ Chunk "
                        f"#{chunk_index} "
                        f"failed: {exc}",
                        flush=True,
                    )

                    try:

                        await websocket.send_bytes(
                            chunk
                        )

                    except Exception:
                        break

    except WebSocketDisconnect:

        print(
            "🔌 WebSocket disconnected",
            flush=True,
        )

    except Exception as exc:

        print(
            f"❌ WebSocket error: {exc}",
            flush=True,
        )


# ═══════════════════════════════════════════════════════════
# WEBSOCKET CHUNK PROCESSING
# ═══════════════════════════════════════════════════════════

async def _process_stream_chunk(
    pcm_bytes: bytes,
) -> bytes:

    loop = asyncio.get_running_loop()

    def _process() -> bytes:

        audio = np.frombuffer(
            pcm_bytes,
            dtype=np.int16,
        )

        audio = audio.reshape(
            -1,
            STREAM_CHANNELS,
        )

        audio_float = (
            audio.astype(
                np.float32
            )
            / 32768.0
        )

        temp_dir = tempfile.mkdtemp(
            prefix="naqi_stream_"
        )

        try:

            input_wav = (
                Path(temp_dir)
                / "input.wav"
            )

            sf.write(
                str(input_wav),
                audio_float,
                STREAM_SAMPLE_RATE,
            )

            separated = (
                Path(temp_dir)
                / "separated"
            )

            separated.mkdir()

            _run(
                [
                    sys.executable,
                    "-m",
                    "demucs",
                    "-d",
                    DEVICE,
                    "-n",
                    MODEL,
                    "--two-stems=vocals",
                    "--shifts",
                    str(DEMUCS_SHIFTS),
                    "--overlap",
                    str(DEMUCS_OVERLAP),
                    "-o",
                    str(separated),
                    str(input_wav),
                ]
            )

            vocals_path = None

            for root, dirs, files in os.walk(
                str(separated)
            ):

                if (
                    "vocals.wav"
                    in files
                ):

                    vocals_path = (
                        Path(root)
                        / "vocals.wav"
                    )

                    break

            if vocals_path is None:
                return pcm_bytes

            processed, _ = sf.read(
                str(vocals_path),
                dtype="float32",
            )

            if processed.ndim == 1:

                processed = np.stack(
                    [
                        processed,
                        processed,
                    ],
                    axis=1,
                )

            processed = np.clip(
                processed,
                -1.0,
                1.0,
            )

            processed_int16 = (
                processed * 32767
            ).astype(
                np.int16
            )

            return (
                processed_int16
                .tobytes()
            )

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    return await loop.run_in_executor(
        None,
        _process,
    )
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


@dataclass(slots=True)
class Segment:
    """One transcribed audio segment."""

    start: float
    end: float
    text: str


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to an SRT timestamp (HH:MM:SS,mmm)."""
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments: Iterable[Segment], output_path: str | Path) -> None:
    """Write transcript segments to an SRT subtitle file."""
    lines: list[str] = []
    subtitle_index = 1
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.extend(
            [
                str(subtitle_index),
                f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}",
                text,
                "",
            ]
        )
        subtitle_index += 1
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def convert_to_wav(input_path: str | Path, output_path: str | Path, sample_rate: int = 16_000) -> None:
    """Convert any FFmpeg-readable audio/video file to mono 16 kHz WAV."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required. Install it from https://ffmpeg.org/.")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def chunk_wav(input_wav: str | Path, chunk_seconds: int = 600) -> list[Path]:
    """Split a WAV file into fixed-size chunks without loading the whole file in memory."""
    input_wav = Path(input_wav)
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    output_dir = input_wav.parent / f"{input_wav.stem}_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Path] = []
    with wave.open(str(input_wav), "rb") as reader:
        params = reader.getparams()
        frames_per_chunk = int(params.framerate * chunk_seconds)
        index = 1
        while True:
            frames = reader.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = output_dir / f"chunk_{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(frames)
            chunks.append(chunk_path)
            index += 1
    return chunks


def transcribe_with_openai(
    audio_path: str | Path,
    *,
    model: str = "gpt-4o-transcribe",
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """Transcribe audio using OpenAI speech-to-text models."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK first: pip install openai") from exc

    kwargs: dict[str, object] = {"model": model}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    client = OpenAI()
    with Path(audio_path).open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(file=audio_file, **kwargs)
    return transcript.text


def transcribe_large_file_with_openai(
    input_path: str | Path,
    *,
    model: str = "gpt-4o-transcribe",
    language: str | None = None,
    prompt: str | None = None,
    chunk_seconds: int = 600,
) -> str:
    """Convert, chunk, and transcribe a long file with OpenAI's API."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        wav_path = temp_dir_path / "audio.wav"
        convert_to_wav(input_path, wav_path)
        chunks = chunk_wav(wav_path, chunk_seconds=chunk_seconds)
        parts = [
            transcribe_with_openai(chunk, model=model, language=language, prompt=prompt)
            for chunk in chunks
        ]
    return "\n".join(part.strip() for part in parts if part.strip())


def transcribe_with_groq(
    audio_path: str | Path,
    *,
    model: str = "whisper-large-v3-turbo",
    language: str | None = None,
    prompt: str | None = None,
) -> str:
    """Transcribe audio with Groq's OpenAI-compatible Whisper endpoint."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("Install the Groq SDK first: pip install groq") from exc

    kwargs: dict[str, object] = {"model": model, "temperature": 0.0}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    with Path(audio_path).open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(file=audio_file, **kwargs)
    return transcript.text


def transcribe_with_faster_whisper(
    audio_path: str | Path,
    *,
    model_size: str = "large-v3",
    device: Literal["auto", "cpu", "cuda"] = "auto",
    compute_type: str = "auto",
    language: str | None = None,
) -> tuple[str, list[Segment]]:
    """Transcribe audio locally with Faster-Whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install Faster-Whisper first: pip install faster-whisper") from exc

    if device == "auto":
        device = "cuda" if _cuda_is_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    kwargs: dict[str, object] = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        kwargs["language"] = language

    raw_segments, _info = model.transcribe(str(audio_path), **kwargs)
    segments = [Segment(start=s.start, end=s.end, text=s.text) for s in raw_segments]
    return "".join(s.text for s in segments).strip(), segments


def record_microphone(output_path: str | Path = "microphone.wav", seconds: int = 8, sample_rate: int = 16_000) -> Path:
    """Record microphone audio to a WAV file."""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise RuntimeError("Install microphone dependencies: pip install sounddevice scipy") from exc

    output_path = Path(output_path)
    print(f"Recording for {seconds} seconds...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    write(output_path, sample_rate, audio)
    print(f"Saved recording to {output_path}")
    return output_path


def _cuda_is_available() -> bool:
    """Return True when PyTorch sees a CUDA GPU, without requiring torch at install time."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio to text in Python.")
    parser.add_argument("audio", nargs="?", help="Path to an audio/video file")
    parser.add_argument("--engine", choices=("openai", "groq", "faster-whisper"), default="faster-whisper")
    parser.add_argument("--model", default=None, help="Model name. Defaults depend on the engine.")
    parser.add_argument("--language", default=None, help="Optional ISO-639-1 language hint, e.g. en, fr, es")
    parser.add_argument("--prompt", default=None, help="Optional context prompt for API transcription")
    parser.add_argument("--srt", default=None, help="Optional .srt output path (Faster-Whisper engine)")
    parser.add_argument("--long", action="store_true", help="Convert/chunk long files before OpenAI transcription")
    parser.add_argument("--chunk-seconds", type=int, default=600, help="Chunk size for --long, default: 600")
    parser.add_argument("--record", type=int, metavar="SECONDS", help="Record from microphone first")
    args = parser.parse_args(argv)

    audio_path: Path
    if args.record:
        audio_path = record_microphone(seconds=args.record)
    else:
        if not args.audio:
            parser.error("provide an audio file or use --record SECONDS")
        audio_path = Path(args.audio)
        if not audio_path.exists():
            parser.error(f"File not found: {audio_path}")

    if args.engine == "openai":
        if args.long:
            print(transcribe_large_file_with_openai(
                audio_path,
                model=args.model or "gpt-4o-transcribe",
                language=args.language,
                prompt=args.prompt,
                chunk_seconds=args.chunk_seconds,
            ))
        else:
            print(transcribe_with_openai(
                audio_path,
                model=args.model or "gpt-4o-transcribe",
                language=args.language,
                prompt=args.prompt,
            ))
        return 0

    if args.engine == "groq":
        print(transcribe_with_groq(
            audio_path,
            model=args.model or "whisper-large-v3-turbo",
            language=args.language,
            prompt=args.prompt,
        ))
        return 0

    text, segments = transcribe_with_faster_whisper(
        audio_path,
        model_size=args.model or "large-v3",
        language=args.language,
    )
    print(text)
    if args.srt:
        write_srt(segments, args.srt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

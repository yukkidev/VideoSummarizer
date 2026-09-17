from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline.models import Segment, Transcript


@pytest.fixture()
def tmp_data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def sample_segments() -> list[Segment]:
    return [
        Segment(0.0, 2.0, "Hello and welcome to the show."),
        Segment(2.0, 5.5, "Today we talk about local language models."),
        Segment(5.5, 9.0, "First, a word from our sponsor."),
        Segment(9.0, 12.0, "Then we dive into the details."),
        Segment(12.0, 16.0, "Finally we answer your questions live."),
    ]


@pytest.fixture()
def sample_transcript(sample_segments: list[Segment]) -> Transcript:
    text = " ".join(s.text for s in sample_segments)
    return Transcript(
        text=text,
        segments=sample_segments,
        source="test",
        language="en",
        duration=sample_segments[-1].end,
    )


def completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["yt-dlp"], returncode=returncode, stdout=stdout, stderr=stderr,
    )

"""
Unit tests for resumable sitemap search checkpoints.
"""

import json
import os

import pytest

from trafilatura.checkpoint import (
    CheckpointFormatError,
    CheckpointMismatchError,
    SitemapCheckpoint,
)


def _valid_state(**overrides):
    state = {
        "version": 1,
        "url": "https://example.org/sitemap.xml",
        "target_lang": None,
        "external": False,
        "seen": ["https://example.org/sitemap.xml"],
        "queue": ["https://example.org/part-2.xml"],
        "guesses": [],
        "robots_checked": True,
        "urls": ["https://example.org/a"],
        "completed": False,
    }
    state.update(overrides)
    return state


def test_open_creates_a_new_state_file(tmp_path):
    path = tmp_path / "state.json"
    checkpoint = SitemapCheckpoint.open(path, "https://example.org/sitemap.xml")
    # the file is materialized on the first save, not on open
    assert not path.exists()
    assert checkpoint.seen == set()
    assert checkpoint.completed is False
    checkpoint.save()
    assert path.exists()
    # the file is valid JSON with the expected structure
    reloaded = SitemapCheckpoint.load(path)
    assert reloaded.url == "https://example.org/sitemap.xml"


def test_open_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    checkpoint = SitemapCheckpoint.open(path, "https://example.org/sitemap.xml", target_lang="de", external=True)
    checkpoint.record(
        "https://example.org/sitemap.xml",
        {"https://example.org/sitemap.xml"},
        ["https://example.org/part-2.xml"],
        ["https://example.org/sitemap_news.xml"],
        True,
        ["https://example.org/de/a"],
    )
    checkpoint.save()

    reloaded = SitemapCheckpoint.open(path, "https://example.org/sitemap.xml", target_lang="de", external=True)
    assert reloaded.seen == {"https://example.org/sitemap.xml"}
    assert reloaded.queue == ["https://example.org/part-2.xml"]
    assert reloaded.guesses == ["https://example.org/sitemap_news.xml"]
    assert reloaded.robots_checked is True
    assert reloaded.urls == ["https://example.org/de/a"]
    assert reloaded.processed_count == 1


def test_open_rejects_a_mismatched_entry(tmp_path):
    path = tmp_path / "state.json"
    SitemapCheckpoint.open(path, "https://example.org/sitemap.xml").save()
    with pytest.raises(CheckpointMismatchError, match="created for"):
        SitemapCheckpoint.open(path, "https://other.example/sitemap.xml")


def test_open_rejects_mismatched_policy(tmp_path):
    path = tmp_path / "state.json"
    SitemapCheckpoint.open(path, "https://example.org/", target_lang="de").save()
    with pytest.raises(CheckpointMismatchError, match="target_lang"):
        SitemapCheckpoint.open(path, "https://example.org/", target_lang="en")

    other = tmp_path / "external.json"
    SitemapCheckpoint.open(other, "https://example.org/", external=False).save()
    with pytest.raises(CheckpointMismatchError, match="external"):
        SitemapCheckpoint.open(other, "https://example.org/", external=True)


@pytest.mark.parametrize(
    "state",
    [
        {"version": 2},
        {"seen": "not-a-list"},
        {"queue": [1, 2]},
        {"urls": None},
        {"completed": "yes"},
        {"external": "no"},
        {"url": 42},
        {"target_lang": 7},
    ],
)
def test_corrupt_and_unsupported_states_are_rejected(tmp_path, state):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(_valid_state(**state)), encoding="utf-8")
    with pytest.raises(CheckpointFormatError):
        SitemapCheckpoint.load(path)
    with pytest.raises(CheckpointFormatError):
        SitemapCheckpoint.open(path, "https://example.org/sitemap.xml")


def test_empty_state_object_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointFormatError):
        SitemapCheckpoint.load(path)


def test_missing_and_non_json_files_are_rejected(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(CheckpointFormatError):
        SitemapCheckpoint.load(missing)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointFormatError):
        SitemapCheckpoint.load(broken)


def test_failed_save_keeps_the_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    checkpoint = SitemapCheckpoint.open(path, "https://example.org/sitemap.xml")
    checkpoint.record("https://example.org/sitemap.xml", {"https://example.org/sitemap.xml"}, [], [], True, ["u1"])
    checkpoint.save()
    good_content = path.read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    checkpoint.urls = ["u1", "u2"]
    with pytest.raises(CheckpointFormatError):
        checkpoint.save()
    # the last valid state is untouched and still readable
    assert path.read_text(encoding="utf-8") == good_content
    assert SitemapCheckpoint.load(path).urls == ["u1"]
    # no temp file lingers
    assert not list(tmp_path.glob("*.tmp"))


def test_mark_completed_clears_the_frontier(tmp_path):
    path = tmp_path / "state.json"
    checkpoint = SitemapCheckpoint.open(path, "https://example.org/sitemap.xml")
    checkpoint.mark_completed(["https://example.org/a", "https://example.org/b"])
    checkpoint.save()
    reloaded = SitemapCheckpoint.load(path)
    assert reloaded.completed is True
    assert reloaded.queue == []
    assert reloaded.guesses == []
    assert reloaded.urls == ["https://example.org/a", "https://example.org/b"]


def test_record_with_unknown_url_still_marks_robots_decision(tmp_path):
    path = tmp_path / "state.json"
    checkpoint = SitemapCheckpoint.open(path, "https://example.org/")
    checkpoint.record("", set(), ["https://example.org/sitemap.xml"], [], True, [])
    checkpoint.save()
    reloaded = SitemapCheckpoint.load(path)
    assert reloaded.seen == set()
    assert reloaded.queue == ["https://example.org/sitemap.xml"]
    assert reloaded.robots_checked is True

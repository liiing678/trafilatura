"""
Persistent checkpoints for resumable sitemap searches across processes.

A checkpoint stores everything needed to resume a :func:`trafilatura.sitemaps.sitemap_search`
run after the process exits: the sitemaps already submitted, the sitemaps still
queued, pending fallback guesses, the links accumulated so far and the discovery
parameters that define the traversal.

State files are written atomically: a failed write or a killed process can at
worst leave the previous valid file in place.
"""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

STATE_VERSION = 1

# keys that every valid checkpoint file must contain
_REQUIRED_KEYS = frozenset(
    {
        "version",
        "url",
        "target_lang",
        "external",
        "seen",
        "queue",
        "guesses",
        "robots_checked",
        "urls",
        "completed",
    }
)


class CheckpointError(Exception):
    """Base class for checkpoint file problems."""


class CheckpointFormatError(CheckpointError):
    """The checkpoint file is missing, unreadable, corrupt or malformed."""


class CheckpointMismatchError(CheckpointError):
    """The checkpoint belongs to another entry or another link policy."""


def _load_raw(path: str | Path) -> dict[str, Any]:
    "Read and parse a checkpoint file without touching the network."
    try:
        with Path(path).open(encoding="utf-8") as statefile:
            data = json.load(statefile)
    except FileNotFoundError as err:
        raise CheckpointFormatError(f"checkpoint file not found: {path}") from err
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise CheckpointFormatError(f"unreadable or corrupt checkpoint file {path}: {err}") from err
    if not isinstance(data, dict) or not _REQUIRED_KEYS.issubset(data):
        raise CheckpointFormatError(f"malformed checkpoint file: {path}")
    return data


class SitemapCheckpoint:
    "Persistent state of one sitemap search, shared between processes."

    __slots__ = [
        "completed",
        "external",
        "guesses",
        "path",
        "queue",
        "robots_checked",
        "seen",
        "target_lang",
        "url",
        "urls",
    ]

    def __init__(
        self,
        path: str | Path,
        url: str,
        target_lang: str | None,
        external: bool,
        *,
        seen: set[str] | None = None,
        queue: list[str] | None = None,
        guesses: list[str] | None = None,
        robots_checked: bool = False,
        urls: list[str] | None = None,
        completed: bool = False,
    ) -> None:
        self.path = str(Path(path))
        self.url = url
        self.target_lang = target_lang
        self.external = external
        self.seen: set[str] = set() if seen is None else seen
        self.queue: list[str] = [] if queue is None else queue
        self.guesses: list[str] = [] if guesses is None else guesses
        self.robots_checked: bool = robots_checked
        self.urls: list[str] = [] if urls is None else urls
        self.completed: bool = completed

    @classmethod
    def load(cls, path: str | Path) -> "SitemapCheckpoint":
        """Read an existing checkpoint without validating it against an entry.

        Raises:
            CheckpointFormatError: The file is missing or malformed.
        """
        data = _load_raw(path)
        if data.get("version") != STATE_VERSION:
            raise CheckpointFormatError(f"unsupported checkpoint version in {path}")
        if not isinstance(data["url"], str) or not (data["target_lang"] is None or isinstance(data["target_lang"], str)):
            raise CheckpointFormatError(f"malformed checkpoint file: {path}")
        if (
            not isinstance(data["external"], bool)
            or not isinstance(data["robots_checked"], bool)
            or not isinstance(data["completed"], bool)
        ):
            raise CheckpointFormatError(f"malformed checkpoint file: {path}")
        try:
            return cls(
                path,
                data["url"],
                data["target_lang"],
                data["external"],
                seen=set(_as_str_list(data["seen"], path)),
                queue=_as_str_list(data["queue"], path),
                guesses=_as_str_list(data["guesses"], path),
                robots_checked=data["robots_checked"],
                urls=_as_str_list(data["urls"], path),
                completed=data["completed"],
            )
        except KeyError as err:  # covered by _REQUIRED_KEYS, kept defensive
            raise CheckpointFormatError(f"malformed checkpoint file: {path}") from err

    @classmethod
    def open(
        cls,
        path: str | Path,
        url: str,
        target_lang: str | None = None,
        external: bool = False,
    ) -> "SitemapCheckpoint":
        """Open an existing checkpoint or create a fresh one for this entry.

        Raises:
            CheckpointFormatError: The file exists but cannot be parsed, has the
                wrong structure, or its directory is not writable.
            CheckpointMismatchError: The file belongs to another entry URL or
                was created with another language or external-link policy.
        """
        state_path = Path(path)
        if not state_path.exists():
            # do not create the file yet: verify the location is writable,
            # the first save() then materializes the state
            directory = state_path.parent
            if not directory.is_dir() or not os.access(directory, os.W_OK):
                raise CheckpointFormatError(f"checkpoint directory not writable: {directory}")
            return cls(path, url, target_lang, external)

        checkpoint = cls.load(path)
        if checkpoint.url != url:
            raise CheckpointMismatchError(f"checkpoint {path} was created for {checkpoint.url!r}, not {url!r}")
        if checkpoint.target_lang != target_lang:
            raise CheckpointMismatchError(
                f"checkpoint {path} uses target_lang={checkpoint.target_lang!r}, the run requests {target_lang!r}"
            )
        if checkpoint.external != external:
            raise CheckpointMismatchError(
                f"checkpoint {path} uses external={checkpoint.external!r}, the run requests external={external!r}"
            )
        return checkpoint

    @property
    def processed_count(self) -> int:
        "Number of sitemaps already submitted in earlier runs."
        return len(self.seen)

    def record(
        self,
        current_url: str,
        seen: set[str],
        queue: list[str],
        guesses: list[str],
        robots_checked: bool,
        urls: list[str],
    ) -> None:
        "Update the in-memory state after one sitemap has been processed."
        self.seen = set(seen)
        if current_url:
            self.seen.add(current_url)
        self.queue = list(queue)
        self.guesses = list(guesses)
        self.robots_checked = robots_checked
        # the checkpoint only stores the filtered, de-duplicated link set
        self.urls = list(urls)

    def mark_completed(self, urls: list[str]) -> None:
        "Flag the traversal as finished and store the final link set."
        self.completed = True
        self.queue = []
        self.guesses = []
        self.urls = list(urls)

    def save(self) -> None:
        """Write the state atomically.

        The data goes to a temporary file in the same directory and is then
        renamed over the checkpoint, so a crash or write error cannot leave a
        truncated state file behind.
        """
        payload = {
            "version": STATE_VERSION,
            "url": self.url,
            "target_lang": self.target_lang,
            "external": self.external,
            "seen": sorted(self.seen),
            "queue": list(self.queue),
            "guesses": list(self.guesses),
            "robots_checked": self.robots_checked,
            "urls": list(self.urls),
            "completed": self.completed,
        }
        state_path = Path(self.path)
        directory = state_path.parent
        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=f".{state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            tmp_path.replace(state_path)
        except OSError as err:
            # drop the half-written temp file; the previous valid state stays in place
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise CheckpointFormatError(f"could not write checkpoint file {self.path}: {err}") from err


def _as_str_list(value: object, path: str | Path) -> list[str]:
    "Validate a checkpoint field as a list of strings."
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CheckpointFormatError(f"malformed checkpoint file: {path}")
    return value

"""Persistent memory for explicit user corrections, with optional private HF Dataset sync.

Local-only by default (stdlib). Set EIM_CORRECTIONS_REPO or EIM_MEMORY_REPO to a
private Hugging Face Dataset repo and HF_TOKEN to a token with write access to keep
corrections across ephemeral Space restarts. Remote sync failures never break chat.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

_EN_MARKERS = (
    "that's wrong", "that is wrong", "wrong answer", "not what i asked", "i said",
    "i told you", "you repeated", "don't repeat", "do not repeat", "you made the same",
    "not correct", "you forgot", "i already said", "stop doing", "instead of",
)
_AR_MARKERS = (
    "غلط", "مو هذا", "مو هيج", "مو هيچ", "قلتلك", "كلتلك", "نفس الخطأ", "نفس الاخطاء",
    "لا تكرر", "لا تعيد", "كررت", "نسيت", "مو اللي طلبته", "ما طلبت", "مو صحيح", "خطأ",
)
_STOP = set("the a an and or to of for in on with this that it is are was were be do did you your i me my we please fix make write code answer about from into using use".split())


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+", (text or "").lower())
    return {w for w in words if len(w) > 1 and w not in _STOP}


class CorrectionMemory:
    MAX_RECORDS = 300
    PUSH_DELAY = 2.0

    def __init__(self, path: str | None = None, repo: str | None = None):
        self.path = os.path.abspath(path or os.environ.get("EIM_CORRECTIONS_PATH", "eim_corrections.jsonl"))
        self.repo = (repo if repo is not None else (
            os.environ.get("EIM_CORRECTIONS_REPO") or os.environ.get("EIM_MEMORY_REPO", "")
        )).strip()
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or None
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self.sync_status = "disabled" if not self.repo else "configured"
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if self.repo:
            self._pull_remote()

    @staticmethod
    def is_correction(text: str) -> bool:
        low = (text or "").lower()
        return any(x in low for x in _EN_MARKERS + _AR_MARKERS)

    def add(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or not self.is_correction(text):
            return False
        stored_text = text[:2000]
        norm = " ".join(stored_text.split()).casefold()
        with self._lock:
            existing = self._load()
            if any(" ".join(r.get("text", "").split()).casefold() == norm for r in existing):
                return False
            rows = (existing + [{"ts": int(time.time()), "text": stored_text}])[-self.MAX_RECORDS:]
            temporary = f"{self.path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with open(temporary, "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                os.replace(temporary, self.path)
            finally:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
        self._schedule_push()
        return True

    def _load(self) -> list[dict]:
        rows = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and isinstance(row.get("text"), str):
                            rows.append(row)
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass
        return rows[-self.MAX_RECORDS:]

    def relevant(self, query: str, limit: int = 4) -> list[str]:
        rows = self._load()
        if not rows:
            return []
        q = _tokens(query)
        scored = []
        for i, row in enumerate(rows):
            words = _tokens(row.get("text", ""))
            overlap = len(q & words) / max(1, len(q | words))
            # Recency is a tie-breaker, not a replacement for relevance.
            score = overlap + 0.015 * (i / max(1, len(rows) - 1))
            if overlap > 0 or i >= len(rows) - 3:
                scored.append((score, i, row.get("text", "")))
        scored.sort(reverse=True)
        return [text for _, _, text in scored[:max(1, limit)] if text]

    def prompt(self, query: str, limit: int = 4) -> str:
        lessons = self.relevant(query, limit)
        if not lessons:
            return ""
        bullets = "\n".join(f"- {item}" for item in lessons)
        return ("Persistent user corrections from earlier turns. Treat these as constraints; do not repeat "
                "rejected approaches. If a correction conflicts with the current explicit request, follow the current request.\n"
                + bullets)

    def _pull_remote(self) -> None:
        """Pull corrections from a configured dataset repo; tolerate missing repo/file/offline mode."""
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=self.repo,
                repo_type="dataset",
                filename=os.path.basename(self.path),
                token=self.token,
                local_dir=os.path.dirname(self.path),
            )
            if os.path.abspath(local) != self.path and os.path.isfile(local):
                shutil.copyfile(local, self.path)
            self.sync_status = "pulled"
        except Exception as exc:  # A missing file/new repo is normal on first launch.
            self.sync_status = f"pull-unavailable:{type(exc).__name__}"

    def _schedule_push(self) -> None:
        if not self.repo:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.PUSH_DELAY, self.sync_now)
            self._timer.daemon = True
            self._timer.start()

    def sync_now(self) -> bool:
        """Push the JSONL file to the configured dataset. Returns True only after upload succeeds."""
        if not self.repo:
            self.sync_status = "disabled"
            return False
        if not self.token:
            self.sync_status = "push-unavailable:missing-token"
            return False
        try:
            from huggingface_hub import HfApi
            with self._lock:
                api = HfApi(token=self.token)
                api.create_repo(repo_id=self.repo, repo_type="dataset", private=True, exist_ok=True)
                api.upload_file(
                    path_or_fileobj=self.path,
                    path_in_repo=os.path.basename(self.path),
                    repo_id=self.repo,
                    repo_type="dataset",
                    commit_message="Update EIM user-correction memory",
                )
            self.sync_status = "pushed"
            return True
        except Exception as exc:
            self.sync_status = f"push-unavailable:{type(exc).__name__}"
            return False

#!/usr/bin/env python3
"""Keep EmpirioLabs AI's open models and datasets on ModelScope identical to Hugging Face.

Hugging Face (organization empiriolabsai) is the one source. Every public model and dataset there that needs no
access agreement gets a repository of the same name on ModelScope (account empiriolabs) holding the same files: each
file's SHA-256 on ModelScope equals its SHA-256 on Hugging Face, and a file removed from Hugging Face is removed from
ModelScope. Gated or private repositories are never copied, and neither are Spaces (their in-browser demos load
their models from Hugging Face).

    python3 mirror.py plan [--only NAME ...]   what differs; no credential; writes `repos` and `any` to $GITHUB_OUTPUT
    python3 mirror.py sync KIND/NAME           make one ModelScope repository match; needs MODELSCOPE_TOKEN
    python3 mirror.py verify [--only NAME ...] exit 1 unless every repository matches
    python3 mirror.py record RESULTS_DIR       check everything, rewrite mirrored.json and the README table

Standard library only: the job that holds the ModelScope token installs nothing.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client as httpclient
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

HF = "https://huggingface.co"
MS = "https://www.modelscope.cn"
HF_ORG = "empiriolabsai"
MS_OWNER = "empiriolabs"
KINDS = ("models", "datasets")
USER_AGENT = "empiriolabs-modelscope-mirror/1 (+https://github.com/EmpirioLabs-AI/modelscope-mirror)"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
OID = re.compile(r"^[0-9a-f]{64}$")
# File names ModelScope parses itself (the card's front matter, configuration, git attributes): always committed
# inline, never as a storage pointer, or the server would read the pointer text instead of the file.
INLINE_NAMES = {"readme.md", ".gitattributes", ".gitignore", "configuration.json", "configuration.yaml",
                "configuration.yml", "dataset_infos.json", "config.json", ".msc", ".mdl"}
INLINE_MAX = 4 * 1024 * 1024          # a larger file goes through storage even when Hugging Face keeps it in git
COMMIT_MAX_ACTIONS = 500              # the server takes up to 2,000 actions in one commit
COMMIT_MAX_INLINE = 6 * 1024 * 1024   # base64 bytes in one commit request
BATCH_OBJECTS = 100                   # objects per storage pre-sign request
CHUNK = 8 * 1024 * 1024
UPLOAD_WORKERS = 4
STATE = Path(__file__).resolve().parent / "mirrored.json"
README = Path(__file__).resolve().parent / "README.md"
TABLE_START, TABLE_END = "<!-- mirror:start -->", "<!-- mirror:end -->"


class MirrorError(Exception):
    """A failure the run reports and stops on."""


def log(message: str) -> None:
    print(message, flush=True)


def redact(url: str) -> str:
    """A URL without its query string, which can carry a signature."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


# --------------------------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------------------------

def http(method: str, url: str, *, headers: dict | None = None, body: bytes | None = None, timeout: float = 60,
         attempts: int = 4) -> tuple[int, bytes, dict]:
    """One request with retries on 429, 5xx and network errors. Returns (status, body, headers)."""
    last: Exception | None = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as err:
            data = err.read()
            if err.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep(min(retry_after(err.headers) or 2 ** (attempt + 1), 60))
                continue
            return err.code, data, dict(err.headers or {})
        except (urllib.error.URLError, TimeoutError, ConnectionError, httpclient.HTTPException) as err:
            last = err
            if attempt + 1 < attempts:
                time.sleep(2 ** (attempt + 1))
                continue
    raise MirrorError(f"{method} {redact(url)} failed: {last}")


def retry_after(headers) -> float | None:
    try:
        return float((headers or {}).get("Retry-After"))
    except (TypeError, ValueError):
        return None


def get_json(url: str, *, headers: dict | None = None, ok=(200,)) -> tuple[int, object]:
    status, data, _ = http("GET", url, headers=headers)
    if status not in ok:
        return status, None
    return status, json.loads(data or b"null")


def next_link(link_header: str | None) -> str | None:
    """The rel="next" target of an HTTP Link header (Hugging Face pages its tree listing with one)."""
    for part in (link_header or "").split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part)
        if match:
            return match.group(1)
    return None


# --------------------------------------------------------------------------------------------------------------
# Hugging Face: the source
# --------------------------------------------------------------------------------------------------------------

def hf_repos() -> list[tuple[str, str]]:
    """Every (kind, name) the organization lists; refuses an empty answer, which is never true."""
    found = []
    for kind in KINDS:
        status, items = get_json(f"{HF}/api/{kind}?author={HF_ORG}&limit=1000")
        if status != 200 or not isinstance(items, list):
            raise MirrorError(f"Hugging Face did not list the organization's {kind} (HTTP {status})")
        for item in items:
            repo_id = str(item.get("id") or "")
            owner, _, name = repo_id.partition("/")
            if owner != HF_ORG or not NAME.match(name):
                raise MirrorError(f"unexpected repository id from Hugging Face: {repo_id!r}")
            found.append((kind, name))
    if not any(kind == "models" for kind, _ in found):
        raise MirrorError("Hugging Face listed no models for the organization; refusing to plan from that")
    return sorted(found)


def hf_info(kind: str, name: str) -> dict:
    status, info = get_json(f"{HF}/api/{kind}/{HF_ORG}/{name}")
    if status != 200 or not isinstance(info, dict) or not SHA.match(str(info.get("sha") or "")):
        raise MirrorError(f"Hugging Face did not describe {kind}/{name} (HTTP {status})")
    return info


def skip_reason(info: dict) -> str | None:
    """Why a repository is not copied, or None when it is."""
    if info.get("private") is not False:
        return "private on Hugging Face"
    if info.get("gated"):
        return "gated on Hugging Face: downloads need an accepted agreement"
    if info.get("disabled"):
        return "disabled on Hugging Face"
    # ModelScope gives a repository created without a license its own default (Apache 2.0, seen 2026-11-24), so a
    # card that states none would be published under a license it never gave. It is copied once its card has one.
    card = info.get("cardData") or {}
    if not isinstance(card.get("license"), str) or not card["license"].strip():
        return "no license on its Hugging Face card"
    return None


def card_license(info: dict) -> dict:
    """The create request's license fields, from the Hugging Face card."""
    card = info.get("cardData") or {}
    fields = {"license": str(card["license"]).strip()}
    if fields["license"] == "other":
        for key in ("license_name", "license_link"):
            if isinstance(card.get(key), str) and card[key].strip():
                fields[key] = card[key].strip()
    return fields


def hf_file_url(kind: str, name: str, sha: str, path: str) -> str:
    prefix = "" if kind == "models" else f"{kind}/"
    return f"{HF}/{prefix}{HF_ORG}/{name}/resolve/{sha}/{urllib.parse.quote(path, safe='/')}"


def safe_path(path: str) -> str:
    """A repository-relative file path, refused when it could escape the repository."""
    # Split by hand: PurePosixPath folds "a//b" and "a/./b" into a clean path and would hide them.
    if not path or "\\" in path or any(part in ("", ".", "..") for part in path.split("/")) \
            or any(ord(ch) < 32 for ch in path):
        raise MirrorError(f"refusing an unsafe path: {path!r}")
    return path


def hf_files(kind: str, name: str, sha: str, *, contents: dict | None = None) -> dict[str, dict]:
    """path -> {size, sha256, lfs} at revision sha. Files kept in git are downloaded to hash them (they are small);
    their bytes go into `contents` when it is given."""
    files: dict[str, dict] = {}
    url: str | None = f"{HF}/api/{kind}/{HF_ORG}/{name}/tree/{sha}?recursive=true&expand=false"
    while url:
        status, data, headers = http("GET", url)
        if status != 200:
            raise MirrorError(f"Hugging Face did not list the files of {kind}/{name}@{sha[:12]} (HTTP {status})")
        for entry in json.loads(data):
            if entry.get("type") != "file":
                continue
            path = safe_path(str(entry["path"]))
            size = int(entry.get("size") or 0)
            lfs = entry.get("lfs")
            if lfs:
                oid = str(lfs.get("oid") or "")
                if not OID.match(oid) or int(lfs.get("size") or -1) != size:
                    raise MirrorError(f"{kind}/{name}: {path} has no valid storage digest on Hugging Face")
                files[path] = {"size": size, "sha256": oid, "lfs": True}
            else:
                files[path] = {"size": size, "sha256": None, "lfs": False}
        url = next_link(headers.get("Link") or headers.get("link"))
    if not files:
        raise MirrorError(f"Hugging Face listed no files for {kind}/{name}@{sha[:12]}; refusing to mirror that")
    for path, meta in files.items():
        if meta["lfs"]:
            continue
        status, data, _ = http("GET", hf_file_url(kind, name, sha, path), timeout=120)
        if status != 200 or len(data) != meta["size"]:
            raise MirrorError(f"{kind}/{name}: could not read {path} from Hugging Face (HTTP {status})")
        meta["sha256"] = hashlib.sha256(data).hexdigest()
        if contents is not None:
            contents[path] = data
    return files


# --------------------------------------------------------------------------------------------------------------
# ModelScope: the copy
# --------------------------------------------------------------------------------------------------------------

def ms_headers(token: str | None, *, json_body: bool = False) -> dict:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if token:
        # Newer endpoints read the bearer token, older ones the session cookie; ModelScope's own client sends both.
        headers["Authorization"] = f"Bearer {token}"
        headers["Cookie"] = f"m_session_id={token}"
    return headers


def ms_repo_url(kind: str, name: str) -> str:
    return f"{MS}/{kind}/{MS_OWNER}/{name}"


def ms_exists(kind: str, name: str, token: str | None = None) -> bool:
    status, _, _ = http("GET", f"{MS}/api/v1/{kind}/{MS_OWNER}/{name}", headers=ms_headers(token))
    if status == 200:
        return True
    if status == 404:
        return False
    raise MirrorError(f"ModelScope did not say whether {kind}/{name} exists (HTTP {status})")


def _entries(payload: object) -> list:
    data = payload.get("Data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        data = data.get("Files") or data.get("files") or []
    return data if isinstance(data, list) else []


def ms_files(kind: str, name: str, token: str | None = None, *, new: bool = False) -> dict[str, dict]:
    """path -> {size, sha256, lfs} on the master branch. `new`: the repository was just created, so a listing
    that is not there yet means no files."""
    raw: list = []
    if kind == "models":
        url = f"{MS}/api/v1/models/{MS_OWNER}/{name}/repo/files?Revision=master&Recursive=True"
        status, payload = get_json(url, headers=ms_headers(token))
        if status == 404 and new:
            return {}
        if status != 200:
            raise MirrorError(f"ModelScope did not list the files of {kind}/{name} (HTTP {status})")
        raw = _entries(payload)
        if len(raw) >= 3000:
            raise MirrorError(f"ModelScope cuts a model listing at 3,000 entries; {name} reached it")
    else:
        page = 1
        while True:
            url = (f"{MS}/api/v1/datasets/{MS_OWNER}/{name}/repo/tree?Revision=master&Recursive=True"
                   f"&PageNumber={page}&PageSize=200")
            status, payload = get_json(url, headers=ms_headers(token))
            if status == 404 and new and page == 1:
                return {}
            if status != 200:
                raise MirrorError(f"ModelScope did not list the files of {kind}/{name} (HTTP {status})")
            batch = _entries(payload)
            raw.extend(batch)
            if len(batch) < 200:
                break
            page += 1
    files = {}
    for entry in raw:
        if str(entry.get("Type") or entry.get("type") or "") not in ("blob", "file"):
            continue
        path = str(entry.get("Path") or entry.get("path") or "")
        files[path] = {"size": int(entry.get("Size") or entry.get("size") or 0),
                       "sha256": str(entry.get("Sha256") or entry.get("sha256") or ""),
                       "lfs": bool(entry.get("IsLFS"))}
    return files


def ms_create(kind: str, name: str, license_fields: dict, token: str) -> None:
    """A public repository with the Hugging Face card's license (skip_reason refuses a card without one)."""
    if not license_fields.get("license"):
        raise MirrorError(f"{kind}/{name}: no license to create it with")
    payload = {"owner": MS_OWNER, "repo_name": name, "visibility": "public", **license_fields}
    status, data, _ = http("POST", f"{MS}/openapi/v1/{kind}", headers=ms_headers(token, json_body=True),
                           body=json.dumps(payload).encode(), attempts=2)
    if 200 <= status < 300:
        log(f"created {ms_repo_url(kind, name)}")
        return
    text = data.decode("utf-8", "replace")[:300]
    if status == 409 or "exist" in text.lower():
        log(f"{kind}/{name} already exists on ModelScope")
        return
    raise MirrorError(f"ModelScope refused to create {kind}/{name} (HTTP {status}): {text}")


def presign(kind: str, name: str, objects: list[dict], token: str) -> dict[str, dict | None]:
    """oid -> the upload action for each object ModelScope still needs; None for one it already stores."""
    out: dict[str, dict | None] = {}
    for start in range(0, len(objects), BATCH_OBJECTS):
        chunk = objects[start:start + BATCH_OBJECTS]
        status, data, _ = http("POST", f"{MS}/api/v1/repos/{kind}/{MS_OWNER}/{name}/info/lfs/objects/batch",
                               headers=ms_headers(token, json_body=True),
                               body=json.dumps({"operation": "upload", "objects": chunk}).encode())
        if status != 200:
            raise MirrorError(f"ModelScope refused the storage request for {kind}/{name} (HTTP {status}): "
                              f"{data.decode('utf-8', 'replace')[:300]}")
        payload = json.loads(data)
        answered = payload.get("Data", payload) if isinstance(payload, dict) else {}
        listed = {}
        for obj in (answered.get("objects") or []) if isinstance(answered, dict) else []:
            if obj.get("error"):
                raise MirrorError(f"ModelScope refused object {str(obj.get('oid'))[:12]}: {obj['error']}")
            upload = (obj.get("actions") or {}).get("upload")
            listed[str(obj.get("oid"))] = upload if upload and upload.get("href") else None
        for item in chunk:
            out[item["oid"]] = listed.get(item["oid"])
    return out


class HashingReader:
    """Reads a response in chunks, hashing and counting what passes through."""

    def __init__(self, stream, size: int):
        self.stream, self.size, self.seen = stream, size, 0
        self.digest = hashlib.sha256()

    def read(self, amount: int = CHUNK) -> bytes:
        block = self.stream.read(min(amount, CHUNK))
        if block:
            self.seen += len(block)
            self.digest.update(block)
        return block


def stream_blob(url: str, action: dict, size: int, oid: str, token: str) -> None:
    """Download from Hugging Face and upload to ModelScope storage in one pass, checking the digest."""
    source = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=300)
    try:
        if source.status != 200:
            raise MirrorError(f"Hugging Face answered {source.status} for {redact(url)}")
        reader = HashingReader(source, size)
        target = urllib.parse.urlsplit(action["href"])
        conn_cls = httpclient.HTTPSConnection if target.scheme == "https" else httpclient.HTTPConnection
        conn = conn_cls(target.netloc, timeout=600, blocksize=CHUNK)
        try:
            path = target.path + (f"?{target.query}" if target.query else "")
            conn.putrequest("PUT", path, skip_accept_encoding=True)
            headers = {"Content-Length": str(size), "User-Agent": USER_AGENT, **ms_headers(token)}
            headers.pop("Accept", None)
            for key, value in (action.get("header") or {}).items():
                headers[str(key)] = str(value)
            for key, value in headers.items():
                conn.putheader(key, value)
            conn.endheaders()
            while True:
                block = reader.read(CHUNK)
                if not block:
                    break
                conn.send(block)
            if reader.seen != size:
                raise MirrorError(f"read {reader.seen} of {size} bytes from Hugging Face")
            if reader.digest.hexdigest() != oid:
                raise MirrorError("the bytes read from Hugging Face do not match their digest")
            resp = conn.getresponse()
            body = resp.read()
            if resp.status >= 300:
                raise MirrorError(f"ModelScope storage answered HTTP {resp.status}: "
                                  f"{body.decode('utf-8', 'replace')[:200]}")
            try:
                answer = json.loads(body) if body.strip() else {}
            except ValueError:
                answer = {}
            if isinstance(answer, dict) and answer.get("Code") not in (None, 200, "200"):
                raise MirrorError(f"ModelScope storage refused the upload: {answer.get('Message')}")
        finally:
            conn.close()
    finally:
        source.close()


def commit(kind: str, name: str, actions: list[dict], message: str, token: str) -> None:
    status, data, _ = http("POST", f"{MS}/api/v1/repos/{kind}/{MS_OWNER}/{name}/commit/master",
                           headers=ms_headers(token, json_body=True), timeout=300,
                           body=json.dumps({"commit_message": message, "actions": actions}).encode())
    text = data.decode("utf-8", "replace")
    try:
        answer = json.loads(text) if text.strip() else {}
    except ValueError:
        answer = {}
    if status >= 300 or (isinstance(answer, dict) and (answer.get("Success") is False or answer.get("success") is False)):
        raise MirrorError(f"ModelScope refused the commit to {kind}/{name} (HTTP {status}): {text[:300]}")


# --------------------------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------------------------

def diff(source: dict[str, dict], copy: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(paths to write, paths to delete) that make copy equal source."""
    write = sorted(p for p, meta in source.items()
                   if p not in copy or copy[p]["sha256"] != meta["sha256"] or copy[p]["size"] != meta["size"])
    delete = sorted(p for p in copy if p not in source)
    return write, delete


def goes_to_storage(path: str, meta: dict) -> bool:
    if PurePosixPath(path).name.lower() in INLINE_NAMES:
        return False
    return bool(meta["lfs"]) or meta["size"] > INLINE_MAX


def plan_one(kind: str, name: str) -> dict:
    """The state of one repository: in sync, to sync (and why), or skipped."""
    info = hf_info(kind, name)
    entry = {"repo": f"{kind}/{name}", "hf_sha": info["sha"], "modelscope": ms_repo_url(kind, name)}
    reason = skip_reason(info)
    if reason:
        return {**entry, "state": "skipped", "reason": reason}
    source = hf_files(kind, name, info["sha"])
    if not ms_exists(kind, name):
        return {**entry, "state": "sync", "reason": "not on ModelScope yet", "write": len(source), "delete": 0}
    write, delete = diff(source, ms_files(kind, name))
    if write or delete:
        return {**entry, "state": "sync", "reason": f"{len(write)} to write, {len(delete)} to delete",
                "write": len(write), "delete": len(delete)}
    return {**entry, "state": "in-sync"}


def plan(only: list[str] | None = None) -> list[dict]:
    repos = hf_repos()
    if only:
        wanted = set(only)
        repos = [r for r in repos if r[1] in wanted or f"{r[0]}/{r[1]}" in wanted]
        missing = wanted - {r[1] for r in repos} - {f"{r[0]}/{r[1]}" for r in repos}
        if missing:
            raise MirrorError(f"not on Hugging Face: {', '.join(sorted(missing))}")
    return [plan_one(kind, name) for kind, name in repos]


# --------------------------------------------------------------------------------------------------------------
# Syncing one repository
# --------------------------------------------------------------------------------------------------------------

class Rate:
    """Upload speed of one stream, measured from finished uploads; assumed slow until one finishes, so a large file
    is not started when it could not finish before the run's time limit."""

    ASSUMED = 4e6  # bytes a second

    def __init__(self):
        self.bytes, self.seconds = 0, 0.0
        self.lock = threading.Lock()

    def add(self, size: int, seconds: float) -> None:
        if size >= 50e6:  # small files measure latency, not speed
            with self.lock:
                self.bytes += size
                self.seconds += seconds

    def seconds_for(self, size: int) -> float:
        with self.lock:
            speed = self.bytes / self.seconds if self.seconds else self.ASSUMED
        return 1.25 * size / max(speed, 1e5)


def commit_batches(actions: list[dict]) -> list[list[dict]]:
    """Split actions so each commit stays under the action count and inline byte budget."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    weight = 0
    for action in actions:
        size = len(action.get("content") or "")
        if current and (len(current) >= COMMIT_MAX_ACTIONS or weight + size > COMMIT_MAX_INLINE):
            batches.append(current)
            current, weight = [], 0
        current.append(action)
        weight += size
    if current:
        batches.append(current)
    return batches


def sync(repo: str, token: str, *, deadline: float) -> dict:
    kind, _, name = repo.partition("/")
    if kind not in KINDS or not NAME.match(name):
        raise MirrorError(f"not a repository key: {repo!r}")
    info = hf_info(kind, name)
    reason = skip_reason(info)
    if reason:
        raise MirrorError(f"{repo} is not copied: {reason}")
    sha = info["sha"]
    contents: dict[str, bytes] = {}
    source = hf_files(kind, name, sha, contents=contents)
    started = time.monotonic()
    if ms_exists(kind, name, token):
        copy = ms_files(kind, name, token)
    else:
        ms_create(kind, name, card_license(info), token)
        copy = ms_files(kind, name, token, new=True)
    write, delete = diff(source, copy)
    log(f"{repo}@{sha[:12]}: {len(source)} files on Hugging Face, {len(copy)} on ModelScope; "
        f"{len(write)} to write, {len(delete)} to delete")
    result = {"repo": repo, "hf_sha": sha, "modelscope": ms_repo_url(kind, name)}
    if not write and not delete:
        return {**result, "state": "in-sync", "uploaded_bytes": 0}

    stored = [p for p in write if goes_to_storage(p, source[p])]
    for path in stored:  # a file Hugging Face keeps in git has its digest computed above
        if not OID.match(str(source[path]["sha256"])):
            raise MirrorError(f"{repo}: no digest for {path}")
    objects = {source[p]["sha256"]: {"oid": source[p]["sha256"], "size": source[p]["size"]} for p in stored}
    needed = presign(kind, name, list(objects.values()), token) if objects else {}
    todo = [(oid, action) for oid, action in needed.items() if action]
    first_path = {}
    for path in stored:
        first_path.setdefault(source[path]["sha256"], path)
    log(f"{repo}: {len(objects)} stored objects, {len(todo)} to upload "
        f"({sum(objects[o]['size'] for o, _ in todo) / 1e9:.2f} GB), the rest already in ModelScope storage")

    uploaded = 0
    pending = []
    rate = Rate()

    def one(item):
        oid, action = item
        path = first_path[oid]
        size = objects[oid]["size"]
        if time.time() + rate.seconds_for(size) > deadline:
            return oid, "deferred", 0  # would not finish in this run; stored objects carry over to the next
        url = hf_file_url(kind, name, sha, path)
        for attempt in range(3):
            began = time.monotonic()
            try:
                stream_blob(url, action, size, oid, token)
                seconds = max(time.monotonic() - began, 0.001)
                rate.add(size, seconds)
                log(f"  uploaded {path} ({size / 1e6:.1f} MB, {size / 1e6 / seconds:.1f} MB/s)")
                return oid, "done", size
            except (MirrorError, OSError, httpclient.HTTPException) as err:
                log(f"  {path}: attempt {attempt + 1} failed: {err}")
                if attempt == 2 or time.time() + rate.seconds_for(size) > deadline:
                    return oid, f"failed: {err}", 0
                time.sleep(10 * (attempt + 1))
                fresh = presign(kind, name, [objects[oid]], token).get(oid)
                if fresh is None:
                    return oid, "done", 0  # stored meanwhile
                action = fresh
        return oid, "failed", 0

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        for oid, outcome, size in pool.map(one, sorted(todo, key=lambda item: objects[item[0]]["size"])):
            uploaded += size
            if outcome != "done":
                pending.append((first_path[oid], outcome))
    if pending:
        deferred = all(outcome == "deferred" for _, outcome in pending)
        state = "incomplete" if deferred else "failed"
        for path, outcome in pending:
            log(f"  not uploaded: {path} ({outcome})")
        return {**result, "state": state, "uploaded_bytes": uploaded,
                "reason": f"{len(pending)} files not uploaded" + (" before the time limit" if deferred else "")}

    actions = []
    for path in write:
        meta = source[path]
        if path in stored:
            actions.append({"action": "create", "path": path, "type": "lfs", "size": meta["size"],
                            "sha256": meta["sha256"], "content": "", "encoding": ""})
        else:
            actions.append({"action": "create", "path": path, "type": "normal", "size": meta["size"], "sha256": "",
                            "content": base64.b64encode(contents[path]).decode(), "encoding": "base64"})
    for path in delete:
        actions.append({"action": "delete", "path": path, "type": "normal", "size": 0, "sha256": "",
                        "content": "", "encoding": ""})
    batches = commit_batches(actions)
    for index, batch in enumerate(batches, 1):
        message = f"Mirror of https://huggingface.co/{'' if kind == 'models' else kind + '/'}{HF_ORG}/{name} at {sha}"
        if len(batches) > 1:
            message += f" ({index}/{len(batches)})"
        commit(kind, name, batch, message, token)
    log(f"{repo}: committed {len(actions)} changes in {len(batches)} commit(s) "
        f"after {time.monotonic() - started:.0f} s")

    after = ms_files(kind, name, token)
    left_write, left_delete = diff(source, after)
    if left_write or left_delete:
        return {**result, "state": "failed", "uploaded_bytes": uploaded,
                "reason": f"after the commit {len(left_write)} files still differ and {len(left_delete)} are extra: "
                          + ", ".join((left_write + left_delete)[:5])}
    return {**result, "state": "synced", "uploaded_bytes": uploaded}


# --------------------------------------------------------------------------------------------------------------
# Recording: mirrored.json and the README table
# --------------------------------------------------------------------------------------------------------------

def ms_orphans(eligible: set[str]) -> list[str]:
    """ModelScope repositories of ours that no copied Hugging Face repository explains (renamed, removed, or
    gated since): reported, never deleted."""
    found = []
    for kind in KINDS:
        page = 1
        while True:
            # ModelScope answers 400 for a page_size above 50.
            status, payload = get_json(f"{MS}/openapi/v1/{kind}?owner={MS_OWNER}&page_size=50&page_number={page}")
            if status != 200 or not isinstance(payload, dict):
                raise MirrorError(f"ModelScope did not list our {kind} (HTTP {status})")
            items = (payload.get("data") or {}).get(kind) or []
            for item in items:
                repo_id = str(item.get("id") or "")
                owner, _, name = repo_id.partition("/")
                if owner == MS_OWNER and f"{kind}/{name}" not in eligible:
                    found.append(f"{kind}/{name}")
            if len(items) < 50:
                break
            page += 1
    return sorted(found)


def render_table(entries: list[dict]) -> str:
    lines = ["| Hugging Face | ModelScope | Revision |", "|---|---|---|"]
    for entry in sorted(entries, key=lambda e: e["repo"]):
        kind, _, name = entry["repo"].partition("/")
        hf = f"https://huggingface.co/{'' if kind == 'models' else kind + '/'}{HF_ORG}/{name}"
        if entry["state"] == "skipped":
            continue
        lines.append(f"| [{HF_ORG}/{name}]({hf}) | [{MS_OWNER}/{name}]({entry['modelscope']}) | "
                     f"`{entry['hf_sha'][:12]}` |")
    skipped = [e for e in entries if e["state"] == "skipped"]
    if skipped:
        lines.append("")
        lines.append("Not copied: " + "; ".join(f"`{e['repo']}`, {e['reason']}" for e in sorted(skipped, key=lambda e: e["repo"])) + ".")
    return "\n".join(lines)


def record(results_dir: str | None) -> int:
    results = []
    if results_dir and Path(results_dir).is_dir():
        for path in sorted(Path(results_dir).rglob("*.json")):
            try:
                results.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                results.append({"repo": path.stem, "state": "failed", "reason": "unreadable result"})
    entries = plan()
    eligible = {e["repo"] for e in entries if e["state"] != "skipped"}
    state = {e["repo"]: {"hf_sha": e["hf_sha"], "modelscope": e["modelscope"], "state": e["state"],
                         **({"reason": e["reason"]} if e.get("reason") else {})} for e in entries}
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = README.read_text(encoding="utf-8")
    start, end = text.find(TABLE_START), text.find(TABLE_END)
    if start < 0 or end < start:
        raise MirrorError("README.md has no mirror table markers")
    table = render_table([e for e in entries if e["state"] == "in-sync" or e["state"] == "skipped"])
    README.write_text(text[:start + len(TABLE_START)] + "\n" + table + "\n" + text[end:], encoding="utf-8")

    problems = [f"{r.get('repo')}: {r.get('state')} ({r.get('reason', '')})" for r in results
                if r.get("state") not in ("synced", "in-sync", "incomplete")]
    problems += [f"{e['repo']}: still differs ({e.get('reason')})" for e in entries
                 if e["state"] == "sync" and not any(r.get("repo") == e["repo"] and r.get("state") == "incomplete"
                                                      for r in results)]
    try:
        orphans = ms_orphans(eligible)
    except MirrorError as err:
        orphans, problems = [], problems + [str(err)]
    problems += [f"{o}: on ModelScope, but not a copied Hugging Face repository (rename, removal or a new gate; "
                 "decide whether to delete it by hand)" for o in orphans]
    incomplete = [r for r in results if r.get("state") == "incomplete" and r.get("uploaded_bytes", 0) > 0]
    stuck = [r for r in results if r.get("state") == "incomplete" and not r.get("uploaded_bytes")]
    problems += [f"{r['repo']}: no progress before the time limit" for r in stuck]
    for entry in entries:
        log(f"{entry['repo']}: {entry['state']}" + (f" ({entry['reason']})" if entry.get("reason") else ""))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            # A copy that ran out of time but moved bytes goes on in a new run, even when another one failed.
            handle.write(f"continue={'true' if incomplete else 'false'}\n")
    for problem in problems:
        log(f"::error::{problem}")
    return 1 if problems else 0


# --------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--only", nargs="*", default=None)
    p_sync = sub.add_parser("sync")
    p_sync.add_argument("repo")
    p_sync.add_argument("--seconds", type=float, default=5 * 3600, help="stop starting uploads after this long")
    p_sync.add_argument("--result", help="write the result here as JSON")
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--only", nargs="*", default=None)
    p_record = sub.add_parser("record")
    p_record.add_argument("results", nargs="?")
    args = parser.parse_args(argv)
    try:
        only = getattr(args, "only", None)
        if only is None:  # a workflow passes the names it was started with through MIRROR_ONLY
            only = os.environ.get("MIRROR_ONLY", "").split() or None
        if args.command == "plan":
            entries = plan(only)
            for entry in entries:
                log(f"{entry['repo']}: {entry['state']}" + (f" ({entry['reason']})" if entry.get("reason") else ""))
            todo = [e["repo"] for e in entries if e["state"] == "sync"]
            out = os.environ.get("GITHUB_OUTPUT")
            if out:
                with open(out, "a", encoding="utf-8") as handle:
                    handle.write(f"repos={json.dumps(todo)}\nany={'true' if todo else 'false'}\n")
            return 0
        if args.command == "sync":
            token = os.environ.get("MODELSCOPE_TOKEN", "").strip()
            if not token:
                raise MirrorError("MODELSCOPE_TOKEN is not set")
            result = sync(args.repo, token, deadline=time.time() + args.seconds)
            log(f"{result['repo']}: {result['state']}" + (f" ({result['reason']})" if result.get("reason") else ""))
            if args.result:
                Path(args.result).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return 0 if result["state"] in ("synced", "in-sync", "incomplete") else 1
        if args.command == "verify":
            entries = plan(only)
            bad = [e for e in entries if e["state"] == "sync"]
            for entry in entries:
                log(f"{entry['repo']}: {entry['state']}" + (f" ({entry['reason']})" if entry.get("reason") else ""))
            return 1 if bad else 0
        if args.command == "record":
            return record(args.results)
    except MirrorError as err:
        log(f"::error::{err}")
        if args.command == "sync" and getattr(args, "result", None):
            Path(args.result).write_text(json.dumps({"repo": args.repo, "state": "failed", "reason": str(err)},
                                                    indent=2) + "\n", encoding="utf-8")
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

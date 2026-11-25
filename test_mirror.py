"""Offline tests for mirror.py: every network call is replaced, so nothing here reaches Hugging Face or ModelScope."""
import base64
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import mirror

SHA = "a" * 40


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PureFunctions(unittest.TestCase):
    def test_diff_writes_new_and_changed_and_deletes_extra(self):
        source = {"a": {"size": 1, "sha256": "x"}, "b": {"size": 2, "sha256": "y"}, "c": {"size": 3, "sha256": "z"}}
        copy = {"a": {"size": 1, "sha256": "x"}, "b": {"size": 2, "sha256": "other"}, "d": {"size": 4, "sha256": "w"}}
        self.assertEqual(mirror.diff(source, copy), (["b", "c"], ["d"]))

    def test_diff_catches_a_size_change_with_the_same_digest_field(self):
        self.assertEqual(mirror.diff({"a": {"size": 2, "sha256": "x"}}, {"a": {"size": 1, "sha256": "x"}}), (["a"], []))

    def test_cards_and_configuration_stay_inline(self):
        self.assertFalse(mirror.goes_to_storage("README.md", {"lfs": True, "size": 10}))
        self.assertFalse(mirror.goes_to_storage("sub/config.json", {"lfs": True, "size": 10}))
        self.assertTrue(mirror.goes_to_storage("model.safetensors", {"lfs": True, "size": 10}))
        self.assertFalse(mirror.goes_to_storage("g2p.py", {"lfs": False, "size": 90_000}))
        self.assertTrue(mirror.goes_to_storage("big.txt", {"lfs": False, "size": mirror.INLINE_MAX + 1}))

    def test_only_public_ungated_licensed_repositories_are_copied(self):
        card = {"cardData": {"license": "apache-2.0"}}
        self.assertIsNone(mirror.skip_reason({"private": False, "gated": False, "disabled": False, **card}))
        self.assertIn("gated", mirror.skip_reason({"private": False, "gated": "auto", **card}))
        self.assertIn("gated", mirror.skip_reason({"private": False, "gated": "manual", **card}))
        self.assertIn("private", mirror.skip_reason({"private": True, "gated": False, **card}))
        self.assertIn("private", mirror.skip_reason({"gated": False, **card}))  # a missing field is not taken as public
        self.assertIn("disabled", mirror.skip_reason({"private": False, "gated": False, "disabled": True, **card}))
        for no_license in ({}, {"cardData": {}}, {"cardData": {"license": ""}}, {"cardData": {"license": ["mit"]}}):
            self.assertIn("no license", mirror.skip_reason({"private": False, "gated": False, **no_license}))

    def test_a_custom_license_carries_its_name_and_link(self):
        info = {"cardData": {"license": "other", "license_name": "aplomb", "license_link": "LICENSE.md"}}
        self.assertEqual(mirror.card_license(info),
                         {"license": "other", "license_name": "aplomb", "license_link": "LICENSE.md"})
        self.assertEqual(mirror.card_license({"cardData": {"license": "mit", "license_name": "x"}}), {"license": "mit"})

    def test_unsafe_paths_are_refused(self):
        for path in ("../x", "/etc/passwd", "a/../b", "a//b", "a\\b", "", "a/./b", "a\nb"):
            with self.assertRaises(mirror.MirrorError, msg=path):
                mirror.safe_path(path)
        self.assertEqual(mirror.safe_path("voices/af_heart.pt"), "voices/af_heart.pt")

    def test_next_link(self):
        header = '<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel="next"'
        self.assertEqual(mirror.next_link(header), "https://huggingface.co/api/models/x/tree/main?cursor=abc")
        self.assertIsNone(mirror.next_link(None))
        self.assertIsNone(mirror.next_link('<https://x>; rel="prev"'))

    def test_commit_batches_respect_count_and_inline_bytes(self):
        actions = [{"content": "x" * 10} for _ in range(5)]
        with mock.patch.object(mirror, "COMMIT_MAX_ACTIONS", 2):
            self.assertEqual([len(b) for b in mirror.commit_batches(actions)], [2, 2, 1])
        with mock.patch.object(mirror, "COMMIT_MAX_INLINE", 25):
            self.assertEqual([len(b) for b in mirror.commit_batches(actions)], [2, 2, 1])
        big = [{"content": "x" * 100}]  # one file over the budget still goes, alone
        with mock.patch.object(mirror, "COMMIT_MAX_INLINE", 25):
            self.assertEqual([len(b) for b in mirror.commit_batches(big + actions[:1])], [1, 1])

    def test_rate_starts_slow_and_learns(self):
        rate = mirror.Rate()
        slow = rate.seconds_for(int(4e9))
        rate.add(int(1e9), 10.0)  # 100 MB/s
        self.assertLess(rate.seconds_for(int(4e9)), slow / 10)
        rate.add(10, 100.0)  # a small file does not count
        self.assertAlmostEqual(rate.seconds_for(int(1e9)), 12.5)

    def test_the_hub_token_never_follows_a_redirect_to_another_host(self):
        handler = mirror._DropAuthOnRedirect()
        req = mirror.urllib.request.Request("https://huggingface.co/empiriolabsai/x/resolve/a/model.safetensors",
                                            headers={"Authorization": "Bearer t", "User-Agent": "u"})
        away = handler.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/x?sig=s")
        self.assertIsNone(away.get_header("Authorization"))
        same = handler.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/api/resolve-cache/x")
        self.assertEqual(same.get_header("Authorization"), "Bearer t")

    def test_access_follows_the_hub(self):
        self.assertEqual(mirror.access({"gated": False}), {"visibility": "public"})
        self.assertEqual(mirror.access({"gated": "auto"}), {"visibility": "gated", "gated_mode": "auto"})
        self.assertEqual(mirror.access({"gated": "manual"}), {"visibility": "gated", "gated_mode": "manual"})

    def test_redact_drops_the_query(self):
        self.assertEqual(mirror.redact("https://h/p?X-Signature=s"), "https://h/p")

    def test_table_lists_copies_and_names_what_is_not_copied(self):
        table = mirror.render_table([
            {"repo": "models/canora-dutch", "state": "in-sync", "hf_sha": SHA,
             "modelscope": "https://www.modelscope.cn/models/empiriolabs/canora-dutch"},
            {"repo": "datasets/results", "state": "in-sync", "hf_sha": SHA,
             "modelscope": "https://www.modelscope.cn/datasets/empiriolabs/results"},
            {"repo": "models/aplomb-1", "state": "skipped", "hf_sha": SHA, "modelscope": "",
             "reason": "gated on Hugging Face: downloads need an accepted agreement"},
            {"repo": "models/aplomb-9", "state": "in-sync", "hf_sha": SHA, "access": "gated",
             "modelscope": "https://www.modelscope.cn/models/empiriolabs/aplomb-9"},
        ])
        self.assertIn("(https://www.modelscope.cn/models/empiriolabs/aplomb-9) (gated)", table)
        self.assertIn("[empiriolabsai/canora-dutch](https://huggingface.co/empiriolabsai/canora-dutch)", table)
        self.assertIn("(https://huggingface.co/datasets/empiriolabsai/results)", table)
        self.assertIn("Not copied: `models/aplomb-1`, gated", table)
        self.assertNotIn("—", table)


class FakeHub:
    """Hugging Face and ModelScope behind mirror.http, with storage and commits applied in memory."""

    def __init__(self, hf_files, ms_files=None, gated=False, license_id="apache-2.0"):
        self.hf = hf_files  # path -> bytes
        self.lfs = {p for p in hf_files if p.endswith((".pth", ".safetensors"))}
        self.ms = dict(ms_files) if ms_files is not None else None  # path -> bytes, None = not created
        self.stored: set[str] = set()
        self.gated, self.license = gated, license_id
        self.calls: list[tuple[str, str]] = []
        self.commits: list[dict] = []
        self.created: list[dict] = []
        self.access = {"private": False, "gated": False}  # ModelScope's, once the repository exists
        self.access_changes: list[dict] = []
        self.gated_reads: list[str] = []

    def hf_read(self, url, attempts=3):
        self.gated_reads.append(url)
        return 200, self.hf[url.split(f"/resolve/{SHA}/", 1)[1]]

    def http(self, method, url, *, headers=None, body=None, timeout=60, attempts=4):
        self.calls.append((method, url))
        if url == f"{mirror.MS}/openapi/v1/models/{mirror.MS_OWNER}/canora-test" and method == "GET":
            if self.ms is None:
                return 404, b"{}", {}
            return 200, json.dumps({"success": True, "data": {"id": "empiriolabs/canora-test", **self.access}}).encode(), {}
        if url == f"{mirror.MS}/openapi/v1/models/{mirror.MS_OWNER}/canora-test/settings" and method == "PATCH":
            fields = json.loads(body)
            self.access_changes.append(fields)
            self.access = {"private": False, "gated": fields["visibility"] == "gated",
                           **({"gated_mode": fields["gated_mode"]} if "gated_mode" in fields else {})}
            return 200, b'{"success": true}', {}
        if url.startswith(f"{mirror.HF}/api/models/{mirror.HF_ORG}/canora-test/tree/"):
            tree = []
            for path, data in self.hf.items():
                entry = {"type": "file", "path": path, "size": len(data)}
                if path in self.lfs:
                    entry["lfs"] = {"oid": digest(data), "size": len(data)}
                tree.append(entry)
            return 200, json.dumps(tree).encode(), {}
        if url == f"{mirror.HF}/api/models/{mirror.HF_ORG}/canora-test":
            return 200, json.dumps({"sha": SHA, "private": False, "gated": self.gated, "disabled": False,
                                    "cardData": {"license": self.license}}).encode(), {}
        if url.startswith(f"{mirror.HF}/{mirror.HF_ORG}/canora-test/resolve/{SHA}/"):
            path = url.split(f"/resolve/{SHA}/", 1)[1]
            return 200, self.hf[path], {}
        if url == f"{mirror.MS}/api/v1/models/{mirror.MS_OWNER}/canora-test":
            return (200 if self.ms is not None else 404), b"{}", {}
        if url.startswith(f"{mirror.MS}/api/v1/models/{mirror.MS_OWNER}/canora-test/repo/files"):
            if self.ms is None:
                return 404, b"{}", {}
            files = [{"Type": "blob", "Path": p, "Size": len(d), "Sha256": digest(d)} for p, d in self.ms.items()]
            return 200, json.dumps({"Code": 200, "Data": {"Files": files}}).encode(), {}
        if url == f"{mirror.MS}/openapi/v1/models" and method == "POST":
            payload = json.loads(body)
            self.created.append(payload)
            self.ms = {}
            self.access = {"private": False, "gated": payload.get("visibility") == "gated",
                           **({"gated_mode": payload["gated_mode"]} if "gated_mode" in payload else {})}
            return 200, b'{"success": true}', {}
        if url.endswith("/info/lfs/objects/batch"):
            objects = json.loads(body)["objects"]
            answer = [{"oid": o["oid"], "actions": {"upload": {"href": f"https://store.example/{o['oid']}?sig=s"}}}
                      for o in objects if o["oid"] not in self.stored]
            return 200, json.dumps({"Code": 200, "Data": {"objects": answer}}).encode(), {}
        if url.endswith("/commit/master"):
            payload = json.loads(body)
            self.commits.append(payload)
            for action in payload["actions"]:
                if action["action"] == "delete":
                    self.ms.pop(action["path"], None)
                elif action["type"] == "lfs":
                    assert action["sha256"] in self.stored, "a pointer to an object never stored"
                    self.ms[action["path"]] = next(d for p, d in self.hf.items() if digest(d) == action["sha256"])
                else:
                    self.ms[action["path"]] = base64.b64decode(action["content"])
            return 200, b'{"Success": true}', {}
        raise AssertionError(f"unexpected request {method} {url}")

    def stream_blob(self, url, action, size, oid, token, gated=False):
        assert action["href"].startswith("https://store.example/")
        self.stored.add(oid)


class SyncFlow(unittest.TestCase):
    def run_sync(self, hub, seconds=3600):
        with mock.patch.object(mirror, "http", hub.http), mock.patch.object(mirror, "stream_blob", hub.stream_blob), \
                mock.patch.object(mirror, "log", lambda *_: None):
            return mirror.sync("models/canora-test", "token-value", deadline=time.time() + seconds)

    def test_creates_the_repository_and_copies_every_file(self):
        files = {"README.md": b"---\nlicense: apache-2.0\n---\n# Canora", "canora.pth": b"weights" * 100,
                 "voices/a.safetensors": b"voice", "g2p.py": b"print(1)\n"}
        hub = FakeHub(files)
        result = self.run_sync(hub)
        self.assertEqual(result["state"], "synced")
        self.assertEqual(hub.ms, files)
        self.assertEqual(hub.created, [{"owner": "empiriolabs", "repo_name": "canora-test", "visibility": "public",
                                        "license": "apache-2.0"}])
        self.assertEqual(len(hub.commits), 1)
        self.assertTrue(hub.commits[0]["commit_message"].endswith(f"empiriolabsai/canora-test at {SHA}"))
        kinds = {a["path"]: a["type"] for a in hub.commits[0]["actions"]}
        self.assertEqual(kinds, {"README.md": "normal", "canora.pth": "lfs", "voices/a.safetensors": "lfs",
                                 "g2p.py": "normal"})

    def test_a_card_without_a_license_is_not_copied(self):
        hub = FakeHub({"README.md": b"# x"}, license_id=None)
        with self.assertRaises(mirror.MirrorError):
            self.run_sync(hub)
        self.assertEqual(hub.created, [])

    def test_only_changes_are_sent_and_extra_files_are_deleted(self):
        files = {"README.md": b"new card", "canora.pth": b"weights"}
        hub = FakeHub(files, ms_files={"README.md": b"old card", "canora.pth": b"weights", "stale.txt": b"x"})
        hub.stored.add(digest(b"weights"))
        result = self.run_sync(hub)
        self.assertEqual(result["state"], "synced")
        self.assertEqual(hub.ms, files)
        actions = {(a["action"], a["path"]) for a in hub.commits[0]["actions"]}
        self.assertEqual(actions, {("create", "README.md"), ("delete", "stale.txt")})

    def test_in_sync_makes_no_commit(self):
        files = {"README.md": b"card", "canora.pth": b"weights"}
        hub = FakeHub(files, ms_files=files)
        self.assertEqual(self.run_sync(hub)["state"], "in-sync")
        self.assertEqual(hub.commits, [])
        self.assertFalse(any(url.endswith("/objects/batch") for _, url in hub.calls))

    def test_objects_already_stored_are_not_uploaded_again(self):
        files = {"README.md": b"card", "canora.pth": b"weights"}
        hub = FakeHub(files, ms_files={})
        hub.stored.add(digest(b"weights"))
        uploads = []
        hub.stream_blob = lambda *args, **kw: uploads.append(args)
        self.assertEqual(self.run_sync(hub)["state"], "synced")
        self.assertEqual(uploads, [])

    def test_no_time_left_defers_without_committing(self):
        files = {"README.md": b"card", "canora.pth": b"w" * 1000}
        hub = FakeHub(files, ms_files={})
        result = self.run_sync(hub, seconds=-1)
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(hub.commits, [])

    def test_a_gated_repository_is_refused_without_a_hub_token(self):
        hub = FakeHub({"README.md": b"card"}, gated="auto")
        with mock.patch.dict(mirror.os.environ, {"HF_TOKEN": ""}):
            with self.assertRaises(mirror.MirrorError):
                self.run_sync(hub)
        self.assertFalse(any(method == "POST" for method, _ in hub.calls))

    def test_a_gated_repository_is_copied_gated_with_the_same_approval_mode(self):
        files = {"README.md": b"card", "config.json": b"{}", "model.safetensors": b"w" * 50}
        for mode in ("auto", "manual"):
            hub = FakeHub(files, gated=mode)
            with mock.patch.dict(mirror.os.environ, {"HF_TOKEN": "read-token"}), \
                    mock.patch.object(mirror, "hf_read", hub.hf_read):
                result = self.run_sync(hub)
            self.assertEqual(result["state"], "synced")
            self.assertEqual(hub.ms, files)
            self.assertEqual((hub.created[0]["visibility"], hub.created[0]["gated_mode"]), ("gated", mode))
            self.assertEqual(len(hub.gated_reads), 2)  # the files kept in git, read with the token

    def test_a_new_gate_on_the_hub_reaches_modelscope(self):
        files = {"README.md": b"card"}
        hub = FakeHub(files, ms_files=files, gated="auto")
        with mock.patch.dict(mirror.os.environ, {"HF_TOKEN": "read-token"}), \
                mock.patch.object(mirror, "hf_read", hub.hf_read):
            result = self.run_sync(hub)
        self.assertEqual(result["state"], "in-sync")
        self.assertEqual(hub.access_changes, [{"visibility": "gated", "gated_mode": "auto"}])

    def test_a_card_that_embeds_hub_files_is_named(self):
        self.assertIsNone(mirror.card_problem(b'<img src="https://media.empiriolabs.ai/assets/hf/x/v1/r.png">'))
        self.assertIsNone(mirror.card_problem(b'[page](https://huggingface.co/empiriolabsai/x/blob/main/LICENSE)'))
        self.assertIn("1 file", mirror.card_problem(
            b'<audio src="https://huggingface.co/empiriolabsai/x/resolve/main/samples/a.wav"></audio>'))
        self.assertIn("1 file", mirror.card_problem(
            b'<audio src="https://media.empiriolabs.ai/assets/hf/x/files/0123456789abcdef/a.wav"></audio>'))
        self.assertIsNone(mirror.card_problem(None))

    def test_a_digest_that_does_not_match_fails_the_upload(self):
        class Source:
            status = 200

            def __init__(self, data):
                self.data, self.pos = data, 0

            def read(self, n):
                block = self.data[self.pos:self.pos + n]
                self.pos += len(block)
                return block

            def close(self):
                pass

        class Conn:
            def __init__(self, *a, **k):
                pass

            def putrequest(self, *a, **k):
                pass

            def putheader(self, *a):
                pass

            def endheaders(self):
                pass

            def send(self, block):
                pass

            def close(self):
                pass

        with mock.patch.object(mirror.HF_OPENER, "open", lambda *a, **k: Source(b"tampered")), \
                mock.patch.object(mirror.httpclient, "HTTPSConnection", Conn):
            with self.assertRaises(mirror.MirrorError):
                mirror.stream_blob("https://huggingface.co/x", {"href": "https://store.example/o"}, 8,
                                   digest(b"original"), "t")


class Record(unittest.TestCase):
    def test_record_rewrites_the_table_and_fails_on_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(f"intro\n{mirror.TABLE_START}\nold\n{mirror.TABLE_END}\noutro\n", encoding="utf-8")
            state = Path(tmp) / "mirrored.json"
            entries = [{"repo": "models/a", "state": "in-sync", "hf_sha": SHA, "modelscope": "https://m/a"},
                       {"repo": "models/b", "state": "sync", "hf_sha": SHA, "modelscope": "https://m/b",
                        "reason": "1 to write, 0 to delete"}]
            with mock.patch.object(mirror, "README", readme), mock.patch.object(mirror, "STATE", state), \
                    mock.patch.object(mirror, "plan", lambda only=None: entries), \
                    mock.patch.object(mirror, "ms_orphans", lambda eligible: []), \
                    mock.patch.object(mirror, "log", lambda *_: None):
                self.assertEqual(mirror.record(None), 1)
            text = readme.read_text(encoding="utf-8")
            self.assertIn("empiriolabsai/a", text)
            self.assertNotIn("empiriolabsai/b", text)
            self.assertTrue(text.startswith("intro\n") and text.endswith("outro\n"))
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["models/b"]["state"], "sync")

    def test_a_failed_listing_still_records_and_fails_the_run(self):
        def broken(eligible):
            raise mirror.MirrorError("ModelScope did not list our models (HTTP 400)")

        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(f"{mirror.TABLE_START}\n{mirror.TABLE_END}\n", encoding="utf-8")
            state = Path(tmp) / "mirrored.json"
            entries = [{"repo": "models/a", "state": "in-sync", "hf_sha": SHA, "modelscope": "https://m/a"}]
            with mock.patch.object(mirror, "README", readme), mock.patch.object(mirror, "STATE", state), \
                    mock.patch.object(mirror, "plan", lambda only=None: entries), \
                    mock.patch.object(mirror, "ms_orphans", broken), \
                    mock.patch.object(mirror, "log", lambda *_: None):
                self.assertEqual(mirror.record(None), 1)
            self.assertIn("empiriolabsai/a", readme.read_text(encoding="utf-8"))
            self.assertTrue(state.exists())


if __name__ == "__main__":
    unittest.main()

"""Public preparation preserves token budgets and session-disjoint splits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from safetensors import safe_open


class PublicDataTest(unittest.TestCase):
    def test_saved_sets_preserve_sessions_tokens_and_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            expected = {}
            for source, start in (("agents", 1000), ("chat", 10000)):
                folder = data / f"Qwen3.8-Flash-Next-calib-{source}"
                folder.mkdir()
                rows = []
                for i in range(128):
                    session = f"{source}:{i}"
                    # Two windows of one session must never enter different splits.
                    expected[session] = start + i
                    rows.append({"id": i, "input_ids": [start + i] * 256})
                (folder / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            traces = data / "Qwen3.8-Flash-Next-evaluation/eval_traces"
            traces.mkdir(parents=True)
            for domain in ("nemotron", "multilingual"):
                (traces / f"{domain}_trace.json").write_text(json.dumps({"rows": [{"input_ids": [1] * 64, "response_ids": [2] * 64, "score_spans": [[64, 128]]}]}))
            script = Path(__file__).resolve().parents[1] / "scripts/s0_build_sets.py"
            subprocess.run([sys.executable, str(script), "--data", str(data), "--window", "128",
                            "--calib-tokens", "4096", "--calib2-tokens", "2048", "--eval-tokens", "2048"],
                           env={**os.environ, "T3_WORK_DIR": tmp}, check=True, capture_output=True, text=True)
            sessions = {}
            for name, budget in (("calib", 4096), ("calib2", 2048), ("eval_public", 2048)):
                with safe_open(data / "sets" / f"{name}.safetensors", framework="np") as f:
                    ids, offsets, meta = f.get_tensor("input_ids"), f.get_tensor("offsets"), f.metadata()
                labels = json.loads(meta["sessions"])
                self.assertEqual(set(json.loads(meta["sources"])), {"agents", "chat"},
                                 "the first source exhausted the entire corpus budget")
                self.assertEqual(int(offsets[0]), 0)
                self.assertEqual(int(offsets[-1]), len(ids))
                self.assertEqual(len(labels), len(offsets) - 1)
                self.assertLessEqual(len(ids), budget)
                for i, session in enumerate(labels):
                    part = ids[offsets[i]:offsets[i + 1]]
                    self.assertTrue((part == expected[session]).all(), "token/source identity changed")
                    self.assertGreaterEqual(len(part), 64)
                    self.assertLessEqual(len(part), 128)
                sessions[name] = set(labels)
            names = list(sessions)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    self.assertFalse(sessions[left] & sessions[right], "one source session leaked across splits")
            with safe_open(data / "sets/eval_nemotron.safetensors", framework="np") as f:
                self.assertEqual(f.get_tensor("input_ids").tolist(), [1] * 64 + [2] * 64)
                self.assertEqual(json.loads(f.metadata()["score_spans"]), [[[64, 128]]])


if __name__ == "__main__":
    unittest.main()

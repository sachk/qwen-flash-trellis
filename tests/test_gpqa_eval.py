import csv
import importlib.util
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import tempfile
import unittest
import threading


spec = importlib.util.spec_from_file_location(
    "gpqa_eval", Path(__file__).resolve().parents[1] / "scripts/gpqa_eval.py"
)
gpqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpqa)


class GpqaScoringTests(unittest.TestCase):
    def test_shuffled_choices_preserve_correct_answer(self):
        row = {
            "Question": "Which choice is correct?",
            "Correct Answer": "correct",
            "Incorrect Answer 1": "wrong one",
            "Incorrect Answer 2": "wrong two",
            "Incorrect Answer 3": "wrong three",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            item = gpqa.load_items(path, 42)[0]
        self.assertEqual(item["correct_letter"], "D")
        self.assertEqual(item["choices"], ["wrong two", "wrong one", "wrong three", "correct"])

    def test_final_answer_extraction_rejects_ambiguous_answers(self):
        allowed = set("ABCD")
        self.assertEqual(gpqa.parse_answer("Earlier guess: ANSWER: A\nFinal:\nANSWER: D", allowed), ("D", "ok"))
        self.assertEqual(gpqa.parse_answer("ANSWER: A,B", allowed), (None, "no_valid_answer"))
        self.assertEqual(gpqa.parse_answer("I have not selected an answer.", allowed), (None, "unparsed"))

    def test_failures_remain_in_denominator_and_partial_scores_are_not_final(self):
        answer = {"ok": True, "status": "ok", "correct": True, "usage_present": True,
                  "prompt_tokens": 5, "completion_tokens": 10, "reasoning_tokens": 6,
                  "total_tokens": 15, "cached_tokens": 0}
        failure = {"ok": False, "status": "transport_error", "correct": False}
        partial = gpqa.aggregate([answer], [], 2, 1.0, None)
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["accuracy_strict_denominator"])
        final = gpqa.aggregate([answer], [failure], 2, 1.0, None)
        self.assertTrue(final["complete"])
        self.assertIsNone(final["accuracy_strict_denominator"])
        self.assertFalse(final["valid_quality_result"])
        self.assertEqual(final["attempted_records"], 2)
        self.assertEqual(final["transport_errors"], 1)
        self.assertEqual(final["statuses"], {"ok": 1, "transport_error": 1})
        self.assertEqual(final["tokens"]["completion_tokens"]["total"], 10)

    def test_closed_sse_without_terminal_metadata_is_not_a_completed_answer(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                event = {"choices": [{"delta": {"content": "ANSWER: C"}}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                if self.path.startswith("/complete/"):
                    event = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = f"http://127.0.0.1:{server.server_port}"
            complete = gpqa.stream_chat(root + "/complete", {}, 5)
            interrupted = gpqa.stream_chat(root + "/interrupted", {}, 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(complete["ok"])
        self.assertFalse(interrupted["ok"])
        self.assertEqual(interrupted["error_type"], "IncompleteStream")

    def test_resumed_tokens_are_not_charged_to_current_phase_time(self):
        previous = {"ok": True, "status": "ok", "correct": True, "resumed": True,
                    "completion_tokens": 1000, "decode_window_s": 10.0}
        current = {"ok": True, "status": "ok", "correct": True,
                   "completion_tokens": 10, "decode_window_s": 1.0}
        summary = gpqa.aggregate([previous, current], [], 2, 2.0, None)
        self.assertEqual(summary["tokens"]["completion_tokens"]["total"], 1010)
        self.assertEqual(summary["throughput"]["completion_tokens_per_run_s"], 5.0)


if __name__ == "__main__":
    unittest.main()

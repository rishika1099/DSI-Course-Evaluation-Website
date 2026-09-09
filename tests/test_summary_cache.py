"""Cache regression checks. All AI responses are mocked; no API calls are made."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.runtime.caching import cache_data_api, cache_utils
from streamlit.runtime.caching.storage import local_disk_cache_storage as disk

from summary_cache import get_cached_response


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def app_functions(namespace):
    """Load only relevant functions, avoiding app startup and external services."""
    wanted = {
        "_hf_chat_summarize", "_amazon_template", "generate_review_summaries",
        "_generate_decision",
    }
    tree = ast.parse(APP_PATH.read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


class SummaryCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="summary-cache-test-")
        self.addCleanup(self.folder.cleanup)
        self.now = 0
        self.patches = [
            patch.object(disk, "get_cache_folder_path", return_value=self.folder.name),
            patch.object(cache_data_api.DataCaches, "get_storage_manager",
                         return_value=disk.LocalDiskCacheStorageManager()),
            patch.object(cache_utils, "TTLCACHE_TIMER", lambda: self.now),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        cache_data_api._data_caches.clear_all()
        self.addCleanup(cache_data_api._data_caches.clear_all)

    def response(self, prompt, generate, system="Summarize", max_tokens=180):
        return get_cached_response(prompt, system, max_tokens, generate)

    def test_success_survives_daily_expiry_and_memory_reset(self):
        generate = Mock(return_value=("Saved summary", "ok"))
        self.assertEqual(self.response("reviews", generate), ("Saved summary", "ok"))
        self.now += 2 * 24 * 3600
        self.response("reviews", generate)
        self.assertEqual(generate.call_count, 1)

        # Close in-memory storage without deleting the persisted files.
        cache_data_api._data_caches._function_caches.clear()
        unavailable = Mock(side_effect=AssertionError("Unexpected AI request"))
        self.assertEqual(self.response("reviews", unavailable), ("Saved summary", "ok"))
        unavailable.assert_not_called()

    def test_full_prompt_instructions_and_output_limit_are_cache_inputs(self):
        generate = Mock(return_value=("Summary", "ok"))
        prefix = "x" * 3600
        self.response(prefix + "old review", generate)
        self.response(prefix + "new review", generate)
        self.response(prefix + "new review", generate, system="Extract tips")
        self.response(prefix + "new review", generate, system="Extract tips", max_tokens=220)
        self.assertEqual(generate.call_count, 4)

    def test_failure_retries_after_five_minutes_and_success_then_persists(self):
        generate = Mock(side_effect=[(None, "error: quota exceeded"), ("Recovered", "ok")])
        self.assertEqual(self.response("reviews", generate), (None, "error: quota exceeded"))
        self.assertFalse(list(Path(self.folder.name).glob("*.memo")))
        self.now = 299
        self.response("reviews", generate)
        self.assertEqual(generate.call_count, 1)
        self.now = 301
        self.assertEqual(self.response("reviews", generate), ("Recovered", "ok"))
        self.now += 2 * 24 * 3600
        self.response("reviews", generate)
        self.assertEqual(generate.call_count, 2)

    def test_none_is_valid_saved_ai_output_but_blank_is_not(self):
        no_complaints = Mock(return_value=("NONE", "ok"))
        self.response("complaints", no_complaints)
        self.now += 301
        self.assertEqual(self.response("complaints", no_complaints), ("NONE", "ok"))
        no_complaints.assert_called_once()
        blank = Mock(side_effect=[("  ", "ok"), ("Actual summary", "ok")])
        self.assertEqual(self.response("tips", blank), (None, "error: empty response"))
        self.now += 301
        self.assertEqual(self.response("tips", blank), ("Actual summary", "ok"))

    def review_app(self, generate=None):
        self.generate = generate or Mock(return_value=("AI summary", "ok"))
        self.secrets = {"HF_API_TOKEN": "test-token-not-real"}
        namespace = {
            "st": SimpleNamespace(secrets=self.secrets),
            "pd": pd,
            "get_cached_response": get_cached_response,
            "_request_hf_chat_summary": self.generate,
            "_bucket_comments": lambda comments: {
                "positive": comments, "negative": [], "tip": comments,
            },
        }
        return app_functions(namespace)

    def reviews(self):
        return (
            "Z: The projects were useful practice, and the instructor explained the course concepts clearly.",
            "A: Start assignments early and attend office hours; the weekly exercises helped with the exam.",
        )

    def stats(self, avg_use=9):
        return tuple(dict(n=2, avg_use=avg_use, avg_diff=5, liked_pct=100,
                          sentiment=50, style="Project-driven").items())

    def test_reordered_comments_and_new_ratings_reuse_ai_with_fresh_template(self):
        app = self.review_app()
        summaries = app["generate_review_summaries"]
        first = summaries("Course", self.reviews(), self.stats())
        self.assertEqual(self.generate.call_count, 4)
        self.now += 2 * 24 * 3600
        second = summaries("Course", self.reviews()[::-1], self.stats(avg_use=2))
        self.assertEqual(self.generate.call_count, 4)
        self.assertEqual(first["ai"], second["ai"])
        self.assertNotEqual(first["template"], second["template"])
        self.assertIn("not very useful", second["template"])
        self.secrets.clear()
        third = summaries("Course", self.reviews(), self.stats())
        self.assertEqual(third["ai"], first["ai"])
        self.assertEqual(self.generate.call_count, 4)

    def test_added_edited_removed_and_duplicate_comments_generate_new_text(self):
        app = self.review_app()
        summaries = app["generate_review_summaries"]
        original = self.reviews() + ("Office hours helped me prepare for the final.",)
        variants = [
            original,
            original + ("The lecture examples were helpful.",),
            (original[0] + " Updated.",) + original[1:],
            original[:-1],
            original + (original[0],),
        ]
        for index, comments in enumerate(variants, start=1):
            summaries("Course", comments, self.stats())
            self.assertEqual(self.generate.call_count, index * 4)

    def test_partial_failure_keeps_other_sections_and_original_quote_order(self):
        generate = Mock(side_effect=[
            ("Overall", "ok"), (None, "error: quota"),
            ("NONE", "ok"), ("- Start early", "ok"),
            ("- Helpful projects", "ok"),
        ])
        app = self.review_app(generate)
        summaries = app["generate_review_summaries"]
        first = summaries("Course", self.reviews(), self.stats())
        self.assertEqual(first["positive"], list(self.reviews()))
        self.assertTrue(first["positive_is_quotes"])
        self.assertIsNone(first["negative"])
        self.assertEqual(generate.call_count, 4)
        self.now += 301
        second = summaries("Course", self.reviews(), self.stats())
        self.assertEqual(generate.call_count, 5)
        self.assertEqual(second["ai"], "Overall")
        self.assertEqual(second["positive"], "- Helpful projects")
        self.assertFalse(second["positive_is_quotes"])
        self.assertEqual(second["errors"], [])

    def test_comparison_reuses_text_but_refreshes_for_changed_ratings(self):
        app = self.review_app()
        compare = app["_generate_decision"]
        courses = ("Course A", "Course B")
        stats = tuple((course, dict(self.stats())) for course in courses)
        comments = tuple((course, self.reviews()) for course in courses)
        first = compare(courses, stats, comments)
        self.now += 2 * 24 * 3600
        self.assertEqual(compare(courses, stats, comments), first)
        self.assertEqual(self.generate.call_count, 1)
        updated = ((courses[0], dict(self.stats(avg_use=2))), stats[1])
        compare(courses, updated, comments)
        self.assertEqual(self.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()

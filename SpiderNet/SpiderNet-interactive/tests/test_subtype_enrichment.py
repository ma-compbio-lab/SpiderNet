"""Offline regression checks: python -m unittest discover -s tests."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from modules.module2_subtype import service as s


LIBRARIES = {"GO_Biological_Process_2023", "KEGG_2021_Human"}
ROW = [1, "Example term", 0.001, 4.0, 25.0, ["TP53", "EGFR"], 0.01, 0, 0]
GENES = ["TP53", "EGFR", "BRCA1"]


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = patch.object(s, "_enrichr_libraries", return_value=LIBRARIES)
        self.catalogue.start()
        self.addCleanup(self.catalogue.stop)
        self.session_patch = patch("requests.Session")
        self.session = self.session_patch.start().return_value.__enter__.return_value
        self.addCleanup(self.session_patch.stop)
        self.session.post.return_value.json.return_value = {"userListId": 1}
        self.session.get.return_value.json.return_value = {lib: [ROW] for lib in LIBRARIES}

    def test_one_submission_for_two_libraries(self):
        records, status = s._fetch_enrichment(GENES, ["go", "kegg"], "human", 12)
        self.assertEqual(self.session.post.call_count, 1)
        self.assertEqual(self.session.get.call_count, 2)
        self.assertEqual(records["go"][0]["Adjusted P-value"], 0.01)
        self.assertEqual(records["go"][0]["Gene count"], 2)
        self.assertEqual(status["kegg"]["library"], "KEGG_2021_Human")

    def test_rate_limit_stops_without_fallback_or_fake_terms(self):
        response = requests.Response()
        response.status_code = 429
        self.session.post.return_value.raise_for_status.side_effect = requests.HTTPError(response=response)
        records, status = s._fetch_enrichment(GENES, ["go", "kegg"], "human", 12)
        self.assertEqual(records, {"go": [], "kegg": []})
        self.assertEqual(self.session.post.call_count, 1)
        self.session.get.assert_not_called()
        self.assertIn("429", status["go"]["message"])
        fig = s.build_enrichment_bubble([], "GO_BP", status=status["go"])
        self.assertFalse(fig["data"])
        self.assertFalse(fig["layout"]["xaxis"]["visible"])

    def test_partial_success_is_retained(self):
        good = MagicMock()
        good.json.return_value = {"GO_Biological_Process_2023": [ROW]}
        self.session.get.side_effect = [good, requests.Timeout()]
        records, status = s._fetch_enrichment(GENES, ["go", "kegg"], "human", 12)
        self.assertEqual(len(records["go"]), 1)
        self.assertEqual(status["go"]["state"], "ok")
        self.assertEqual(status["kegg"]["state"], "error")

    def test_small_gene_list_does_not_contact_service(self):
        _, status = s._fetch_enrichment(["TP53"], ["go", "kegg"], "human", 12)
        self.assertEqual(status["go"]["state"], "insufficient_genes")
        self.session.post.assert_not_called()

    def test_missing_library_is_not_replaced_after_network_failure(self):
        s._enrichr_libraries.return_value = {"KEGG_2021_Human"}
        records, status = s._fetch_enrichment(GENES, ["go", "kegg"], "human", 12)
        self.assertEqual(status["go"]["state"], "error")
        self.assertEqual(records["go"], [])
        self.assertEqual(self.session.get.call_count, 1)

    def test_empty_success_is_not_reported_as_failure(self):
        self.session.get.return_value.json.return_value = {lib: [] for lib in LIBRARIES}
        _, status = s._fetch_enrichment(GENES, ["go", "kegg"], "human", 12)
        self.assertEqual(status["go"]["state"], "empty")

    def test_old_failed_cache_retries_only_selected_failure(self):
        good = {"Term": "Existing term", "Adjusted P-value": 0.02, "Gene count": 3}
        error = {"Term": "Enrichment failed (human): No GeneSets are valid !!!", "Adjusted P-value": 1.0}
        deg = {"params": {"max_genes_for_enrichment": 100, "top_enrich_terms": 12},
               "deg_full": {"5": [{"gene": "TP53", "padj": 0.001}]},
               "marker_tables": {"5": {"up": [{"gene": g} for g in GENES]}},
               "enrichment": {"5": {"up": {"go": [error], "kegg": [good]}},
                              "6": {"up": {"go": [error]}}}}
        original = copy.deepcopy(deg)
        _, status, changed = s._ensure_selected_enrichment(deg, "5", "up", "human", 100, 12)
        self.assertTrue(changed)
        self.assertEqual(status["go"]["state"], "ok")
        self.assertEqual(deg["enrichment"]["5"]["up"]["kegg"], [good])
        self.assertEqual(deg["enrichment"]["6"], original["enrichment"]["6"])
        self.assertEqual(deg["deg_full"], original["deg_full"])
        self.assertEqual(deg["marker_tables"], original["marker_tables"])
        self.assertEqual(self.session.get.call_count, 1)
        _, _, changed = s._ensure_selected_enrichment(deg, "5", "up", "human", 100, 12)
        self.assertFalse(changed)
        self.assertEqual(self.session.post.call_count, 1)

    def test_new_failures_retry_on_next_click(self):
        deg = {"marker_tables": {"5": {"up": [{"gene": g} for g in GENES]}}}
        self.session.post.side_effect = requests.Timeout()
        _, status, _ = s._ensure_selected_enrichment(deg, "5", "up", "human", 100, 12)
        self.assertEqual(status["go"]["state"], "error")
        self.session.post.side_effect = None
        _, status, _ = s._ensure_selected_enrichment(deg, "5", "up", "human", 100, 12)
        self.assertEqual(status["go"]["state"], "ok")
        # Changed display/gene limits must not reuse incompatible enrichment.
        _, _, changed = s._ensure_selected_enrichment(deg, "5", "up", "human", 2, 5)
        self.assertTrue(changed)
        self.assertEqual(status["go"]["state"], "insufficient_genes")

    def test_legacy_error_is_never_a_plot_point(self):
        fig = s.build_enrichment_bubble([{"Term": "Enrichment failed: old error"}], "GO_BP")
        self.assertFalse(fig["data"])

    def test_atomic_cache_write(self):
        import pickle
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "deg.pkl"
            s._save_deg_cache(path, {"deg_full": {"5": []}})
            self.assertEqual(pickle.loads(path.read_bytes()), {"deg_full": {"5": []}})
            self.assertEqual(list(Path(root).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()

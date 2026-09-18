"""Regression checks for filtering, visible errors, and retryable enrichment."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from modules.module4_perturb import service as s


class PerturbEnrichmentTests(unittest.TestCase):
    def setUp(self):
        s._ENRICH_CACHE.clear()
        self.df = pd.DataFrame({
            'gene': ['ANXA1', 'ANXA2', 'AREG', 'UP', 'DIRECT'],
            'log2FC': [-.079, -.050, -.054, .051, -.07],
            'padj': [0.] * 5, 'pvalue': [0.] * 5, 'neglog10_padj': [300.] * 5,
            'mean_before': [1.] * 5, 'mean_after': [.95] * 5,
            'perturbation_role': ['not directly perturbed'] * 4 + ['directly perturbed'],
        })

    def test_direction_and_scope_do_not_follow_table_row_count(self):
        self.assertEqual(len(s.filter_de_display_table(self.df, .05, .05)), 5)
        self.assertEqual(s.genes_for_enrichment(self.df, 'up', 'indirect_only', .05, .05), ['UP'])
        self.assertEqual(s.genes_for_enrichment(self.df, 'down', 'indirect_only', .05, .05), ['ANXA1', 'ANXA2', 'AREG'])
        self.assertEqual(len(s.genes_for_enrichment(self.df, 'down', 'all', .05, .05)), 4)

    def test_insufficient_genes_no_request_and_explicit_figure(self):
        with patch('modules.module2_subtype.service._fetch_enrichment') as fetch:
            data = s.build_run_response(SimpleNamespace(name='test', setup={}), self.df, {}, 'key', .05, .05)
        fetch.assert_not_called()
        self.assertEqual(data['summary']['n_display_genes'], 5)
        self.assertEqual(data['summary']['n_enrichment_genes'], 1)
        self.assertEqual(data['summary']['enrichment_direction_counts'], {'up': 1, 'down': 3})
        self.assertEqual(data['enrichment_status']['go']['state'], 'insufficient_genes')
        self.assertIn('Selected genes: 1', data['go_figure']['layout']['annotations'][0]['text'])
        self.assertEqual(data['go_figure']['data'], [])

    def test_failure_retry_preserves_successful_library(self):
        row = {'Term': 'test', 'Adjusted P-value': .01, 'Gene count': 3, 'Genes': 'A;B;C'}
        success = {'state': 'ok', 'library': 'test'}
        error = {'state': 'error', 'message': 'Enrichr rate limit (HTTP 429). Wait a few minutes, then click Compute DEG + GO/KEGG again.'}
        with patch('modules.module2_subtype.service._fetch_enrichment', side_effect=[
            ({'go': [row], 'kegg': []}, {'go': success, 'kegg': error}),
            ({'kegg': [row]}, {'kegg': success}),
        ]) as fetch:
            go, kegg, status = s.cached_enrichment(['A', 'B', 'C'], 'human')
            self.assertFalse(go.empty)
            self.assertTrue(kegg.empty)
            self.assertIn('Retry enrichment', status['kegg']['message'])
            figure = s.bubble_figure(kegg, status=status['kegg'])
            self.assertIn('429', figure['layout']['annotations'][0]['text'])
            s.cached_enrichment(['A', 'B', 'C'], 'human')
            self.assertEqual(fetch.call_args.args[1], ['kegg'])
            s.cached_enrichment(['A', 'B', 'C'], 'human')
            self.assertEqual(fetch.call_count, 2)

    def test_successful_empty_response_is_distinct_and_cached(self):
        with patch('modules.module2_subtype.service._fetch_enrichment', return_value=(
            {'go': [], 'kegg': []}, {k: {'state': 'empty', 'library': 'test'} for k in ('go', 'kegg')}
        )) as fetch:
            _, _, status = s.cached_enrichment(['A', 'B', 'C'], 'human')
            s.cached_enrichment(['A', 'B', 'C'], 'human')
            self.assertEqual(status['go']['state'], 'empty')
            self.assertEqual(fetch.call_count, 1)


if __name__ == '__main__':
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from modules.module3_cascade import service as s

class CascadeEnrichmentTests(unittest.TestCase):
    def test_library_and_species_are_preserved_and_error_is_visible(self):
        error={"state":"error","message":"Enrichr request failed (HTTP 429)."}
        with patch("modules.module2_subtype.service._fetch_enrichment",return_value=({"go":[]},{"go":error})) as fetch:
            result=s._run_go(["A","B","C"],"mouse")
        self.assertEqual(fetch.call_args.args[2],"mouse")
        self.assertEqual(fetch.call_args.kwargs["library_candidates"],{"go":["GO_Biological_Process_2021"]})
        fig=s._build_go_bubble(result["df"],"GO",s._theme_spec("dark"),message=result["message"])
        self.assertIn("429",fig["layout"]["annotations"][0]["text"])
        self.assertEqual(fig["data"],[])

    def test_retry_reuses_deg_and_only_retries_failed_go(self):
        s._DEGGO_CACHE.clear()
        ds=SimpleNamespace(name="audit",config={"SPECIES":"human"})
        pair={"cache_key":"audit","sig_df_numeric":pd.DataFrame([{"pair_key":"pair","MI_first_int":1,"MI_second_int":2}])}
        empty=pd.DataFrame()
        cells={"position_dfs":{},"n_triplets":2,"n_slices_with_hits":1}
        deg={"up_df":pd.DataFrame({"gene":["A","B","C"]}),"down_df":pd.DataFrame({"gene":[]}),"deg_df":empty,"message":"cached DE"}
        fail={"state":"error","message":"Network unavailable","df":empty}
        ok={"state":"empty","message":"No terms returned","df":empty}
        with patch.object(s,"_collect_triplet_cells_across_slices",return_value=cells) as collect, patch.object(s,"_exclude_interest_from_baseline",return_value=empty), patch.object(s,"_run_deg",return_value=deg) as run_deg, patch.object(s,"_run_go",side_effect=[ok,fail,ok,ok]) as run_go:
            for _ in range(3):s.run_cascade_deggo(ds,pair,pair_key="pair",cell1_type="A",cell2_type="B",cell3_type="C")
            self.assertEqual(run_deg.call_count,3)
            self.assertEqual(collect.call_count,2)
            self.assertEqual(run_go.call_count,4)
        s._DEGGO_CACHE.clear()

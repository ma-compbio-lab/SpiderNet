import unittest
from modules.module3_cascade import service as s

class CascadeTopNTests(unittest.TestCase):
    def test_topn_changes_display_only(self):
        triple = {"celltype_triples": [f"A{i} -> B -> C" for i in range(60)], "proportions": [.2-i*.001 for i in range(60)]}
        payload = {"pair_payload": {"pair": triple}}
        for n in [10,20,50]:
            result=s.build_stem_response(payload,"pair",top_n=n)
            points=result["stem_figure"]["data"][1]
            self.assertEqual(len(points["x"]),n)
            self.assertEqual(points["customdata"][0],"A0 -> B -> C")
        self.assertEqual(len(triple["proportions"]),60)

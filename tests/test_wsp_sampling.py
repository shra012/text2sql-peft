"""Keep diagnostic prompt interventions out of ordinary scored cohorts."""

import unittest
from argparse import Namespace

from t2s.wsp.tool import pick


class PracticeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "p00001"}, {"id": "x00002"}, {"id": "zschema_p00001", "probe_only": True}]

    def test_random_sampling_excludes_prompt_interventions(self):
        args = Namespace(ids=None, ids_file=None, sample=100, seed=42)
        self.assertEqual({row["id"] for row in pick(self.rows, args, 512)}, {"p00001", "x00002"})

    def test_counterfactuals_remain_available_for_explicit_diagnostics(self):
        args = Namespace(ids="zschema_p00001", ids_file=None, sample=100, seed=42)
        self.assertEqual(pick(self.rows, args, 512), [self.rows[2]])


if __name__ == "__main__":
    unittest.main()

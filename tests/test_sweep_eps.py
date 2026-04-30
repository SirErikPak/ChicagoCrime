import unittest
import sys
from pathlib import Path
import pandas as pd

# ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Src.sweep_eps import sweep_eps_grid


class SweepEpsSmokeTest(unittest.TestCase):
    def test_basic_diagnostics(self):
        pivot = pd.DataFrame([[1, 0, 2], [0, 3, 1]], columns=["a", "b", "c"]) 
        eps_grid = [1e-8, 1e-4, 1e-2, 1e-1]
        out = sweep_eps_grid(pivot, eps_grid, plot=False, verbose=False, near_zero_threshold=None)
        self.assertIsInstance(out, dict)
        df = out["diagnostics_df"]
        self.assertIsInstance(df, pd.DataFrame)
        # expected columns
        for col in [
            "max_abs_clr",
            "pct_rows_large_clr",
            "rank_stability_spearman",
            "rank_stability_kendall",
        ]:
            self.assertIn(col, df.columns)
        self.assertNotIn("pct_cells_near_zero", df.columns)
        self.assertNotIn("clr_var_near_zero", df.columns)
        # meta minimal contract
        meta = out["meta"]
        self.assertIn("auto_select", meta)
        self.assertIn("chosen_eps", meta)
        self.assertIsNone(meta["chosen_eps"])
        # clr_dict keys match eps count
        clr_keys = out["clr_dict"].keys()
        self.assertEqual(len(clr_keys), len(eps_grid))

    def test_near_zero_columns_enabled(self):
        pivot = pd.DataFrame([[1, 0, 2], [0, 3, 1]], columns=["a", "b", "c"])
        out = sweep_eps_grid(pivot, [1e-4, 1e-2], plot=False, verbose=False, near_zero_threshold=1e-3)
        df = out["diagnostics_df"]
        self.assertIn("pct_cells_near_zero", df.columns)
        self.assertIn("clr_var_near_zero", df.columns)


if __name__ == "__main__":
    unittest.main()

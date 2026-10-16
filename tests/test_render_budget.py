"""Fast, CPU-only tests of arithmetic and evidence handling, not neural execution."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from drosophilos.bench.render_budget import main, summarize


def sample():
    return dict(neurons=686351, nodes=8, neural_ms=1982017.3,
                faults=0, timeouts=0, bad_outputs=0, wall_s=28359.755647182465,
                wrong=0, missing=0, frames=2, width=40, height=25,
                neurons_per_node=686351)


class RenderBudgetTests(unittest.TestCase):
    def test_saved_doom2_arithmetic(self):
        out = summarize(sample(), target_seconds=60)
        self.assertAlmostEqual(out["amortized_wall_s_per_requested_frame"], 14179.877823591232)
        self.assertAlmostEqual(out["amortized_neural_s_per_requested_frame"], 991.00865)
        self.assertAlmostEqual(out["wall_to_target_ratio"], 236.33129705985386)
        self.assertEqual(out["total_neurons"], 5490808)
        self.assertEqual(out["requested_pixels"], 2000)
        self.assertEqual(out["outcome"], "reported_clean")

    def test_saved_doom4_arithmetic(self):
        data = sample()
        data.update(neural_ms=2710715.2, wall_s=64746.8699631691,
                    width=24, height=15, neurons_per_node=1440390)
        out = summarize(data)
        self.assertAlmostEqual(out["amortized_wall_s_per_requested_frame"], 32373.43498158455)
        self.assertAlmostEqual(out["amortized_neural_s_per_requested_frame"], 1355.3576)
        self.assertEqual(out["requested_pixels"], 720)
        self.assertNotIn("wall_to_target_ratio", out)

    def test_missing_counter_is_unknown(self):
        data = sample()
        del data["missing"]
        out = summarize(data)
        self.assertEqual(out["outcome"], "incomplete_status")
        self.assertEqual(out["missing_status_fields"], ["missing"])

    def test_failures_are_not_success_even_with_missing_counters(self):
        for key in ("wrong", "missing", "bad_outputs", "faults", "timeouts"):
            with self.subTest(key=key):
                data = sample()
                data[key] = 1
                del data["bad_outputs" if key != "bad_outputs" else "wrong"]
                self.assertEqual(summarize(data)["outcome"], "reported_failures")

    def test_invalid_positive_fields(self):
        for key in ("frames", "width", "height", "wall_s", "neural_ms", "nodes", "neurons_per_node"):
            for value in (0, -1, math.inf, math.nan, True, "2", None):
                with self.subTest(key=key, value=value):
                    data = sample()
                    data[key] = value
                    with self.assertRaises(ValueError):
                        summarize(data)

    def test_fractional_counts_and_bad_counters(self):
        for key in ("frames", "width", "nodes", "wrong", "missing"):
            data = sample()
            data[key] = 1.5
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize(data)
        for value in (-1, True, None, math.inf):
            data = sample()
            data["faults"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(data)

    def test_invalid_target(self):
        for target in (0, -1, math.nan, math.inf, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                summarize(sample(), target_seconds=target)

    def test_non_object(self):
        for data in (None, [], 42):
            with self.subTest(data=data), self.assertRaises(ValueError):
                summarize(data)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(sample()), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main([str(path), "--target-seconds", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue())[0]["outcome"], "reported_clean")
            path.write_text("not json", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(path)]), 2)
                self.assertEqual(main([str(path) + ".missing"]), 2)


if __name__ == "__main__":
    unittest.main()

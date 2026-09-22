"""Reconstruct pressure fits and recompute the reported tabulated results."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from expression_search import Expression, features
from prepare_pressure_remediation import prepare_problem


def read_json(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def main():
    context = read_json("pressure_problem.json")
    settings = context["effective_settings"]
    relations = read_json("pressure_relations.json")
    expressions = {}
    for relation in relations:
        expression = Expression.parse(relation["expression"], settings["grammar"])
        expressions[relation["seed"]] = (expression, relation["constants"])
        problem = prepare_problem(expression, context["targets"], settings)
        solved = linprog(**{key: problem[key] for key in
                           ("c", "A_ub", "b_ub", "A_eq", "b_eq", "bounds")}, method="highs")
        if not solved.success:
            raise RuntimeError(solved.message)
        np.testing.assert_allclose(solved.fun, relation["minimax_objective"], rtol=1e-8, atol=1e-10)
        values = expression.evaluate(features(context["targets"]), relation["constants"])
        targets = np.array([row["pressure_ratio"] for row in context["targets"]])
        assert np.min(values - targets) >= -1e-10
        print(f"Relation {relation['relation']}: {len(targets)} targets, "
              f"minimax pressure excess {solved.fun:.10f}")

    with (ROOT / "data/pressure_validation.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 117
    inputs = set()
    holdout_inputs = set()
    holdout_height = []
    for row in rows:
        key = (row["condition_id"], float(row["allowable"]))
        inputs.add(key)
        expression, constants = expressions[int(row["seed"])]
        predicted = float(expression.evaluate({
            "C": float(row["allowable"]), "E_over_p0": float(row["E_over_p0"]),
            "k_over_p0": float(row["k_over_p0"])}, constants))
        np.testing.assert_allclose(predicted, float(row["pressure_ratio"]), rtol=1e-12, atol=1e-12)
        for chord in ("width", "height"):
            reserve = ((float(row["allowable"]) - float(row[chord + "_closure"]))
                       * float(row["initial_" + chord + "_m"]) * 1000)
            np.testing.assert_allclose(reserve, float(row[chord + "_margin_mm"]), atol=1e-9)
            assert reserve >= 0
        if row["population"] == "interior_holdout":
            holdout_inputs.add(key)
            holdout_height.append(float(row["height_margin_mm"]))
    assert len(inputs) == 39 and len(holdout_inputs) == 12
    print(f"FE observations: {len(rows)}/{len(rows)} satisfy both chord limits at {len(inputs)} inputs")
    print(f"Interior holdout: {len(holdout_inputs)} inputs; minimum height reserve "
          f"{min(holdout_height):.6f} mm")

    forward = read_json("forward_unequal_step.json")
    observed = np.array([row["observed"]["moment_mm"] for row in forward["intervals"]])
    predicted = np.array([row["predicted"]["moment_mm"] for row in forward["intervals"]])
    cumulative = predicted.sum(axis=0) / observed.sum(axis=0) - 1
    np.testing.assert_allclose(cumulative, forward["cumulative"]["moment_relative_error"], atol=1e-12)
    print(f"Unequal-step cumulative moment errors: width {100*cumulative[0]:.2f}%, "
          f"height {100*cumulative[1]:.2f}%")
    print(f"Largest individual-step moment difference: {100*np.max(np.abs(predicted/observed-1)):.2f}%")


if __name__ == "__main__":
    main()

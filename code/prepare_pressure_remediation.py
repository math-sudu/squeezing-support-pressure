"""Construct the current affine minimax pressure problem."""
import ast
import numpy as np
from expression_search import features
from pressure_design import closure_margins


def affine_coefficients(expression, x):
    """Return offset and exact coefficient columns, rejecting nonlinear constants."""
    shape = np.asarray(x["C"]).shape
    names = expression.constants

    def empty(value):
        return np.broadcast_to(value, shape).copy(), np.zeros(shape + (len(names),)), False

    def visit(node):
        if isinstance(node, ast.Constant):
            return empty(float(node.value))
        if isinstance(node, ast.Name):
            if node.id not in names:
                return empty(np.asarray(x[node.id], dtype=float))
            offset, matrix, _ = empty(0.)
            matrix[..., names.index(node.id)] = 1.
            return offset, matrix, True
        if isinstance(node, ast.BinOp):
            a, A, ac = visit(node.left)
            b, B, bc = visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + b, A + B, ac or bc
            if isinstance(node.op, ast.Sub):
                return a - b, A - B, ac or bc
            if isinstance(node.op, ast.Mult) and not (ac and bc):
                return a * b, A * b[..., None] + B * a[..., None], ac or bc
            if isinstance(node.op, ast.Div) and not bc and np.all(np.abs(b) > expression.grammar["denominator_tolerance"]):
                return a / b, A / b[..., None], ac
            raise ValueError("Expression is not affine in its coefficients")
        if isinstance(node, ast.Call):
            a, _, active = visit(node.args[0])
            if active:
                raise ValueError("Nonlinear function contains fitted coefficients")
            function = {"sqrt": np.sqrt, "square": np.square, "log1p": np.log1p}[node.func.id]
            return empty(function(a))
        raise ValueError("Unsupported parsed expression node")

    with np.errstate(all="raise"):
        offset, matrix, _ = visit(expression.tree)
    if not np.isfinite(offset).all() or not np.isfinite(matrix).all():
        raise ValueError("Nonfinite affine decomposition")
    return offset, matrix


def nonexceeding_witness(path, queries, allowable):
    """Pick a measured witness; never round a positive closure mismatch to zero."""
    candidates = []
    for index, state in enumerate(path):
        if state["accepted"] and max(state["width_closure"], state["height_closure"]) <= allowable:
            candidates.append({"kind": "accepted_path_state", "index": index,
                               "pressure_ratio": state["pressure_ratio"],
                               "checkpoint": state["checkpoint"], "checkpoint_sha256": state["checkpoint_sha256"],
                               "observation": state})
    for index, query in enumerate(queries):
        observation = query.get("observation")
        if (query["status"] == "verified_query" and query["solver"]["accepted"] and observation is not None
                and max(observation["width_closure"], observation["height_closure"]) <= allowable):
            if query["solver"]["pressure_ratio"] != query["interpolated_pressure"]:
                raise ValueError("Reference observation pressure differs from its interpolant")
            candidates.append({"kind": "verified_inverse_query", "index": index,
                               "pressure_ratio": query["interpolated_pressure"],
                               "incoming_checkpoint": query["incoming_checkpoint"], "observation": observation})
    if not candidates:
        raise ValueError("No accepted non-exceeding pressure witness")
    witness = min(candidates, key=lambda c: (c["pressure_ratio"], c["kind"], c["index"]))
    witness["margins"] = closure_margins(witness.pop("observation"), allowable)
    original = next(q for q in queries if q["allowable"] == allowable)
    witness["original_interpolated_pressure"] = original["interpolated_pressure"]
    witness["original_reference_mismatch"] = max(original["observation"][c + "_closure"] for c in ("width", "height")) - allowable
    witness["pressure_increment_over_original_interpolant"] = witness["pressure_ratio"] - original["interpolated_pressure"]
    return witness


def prepare_problem(expression, rows, base):
    """Build a linear minimax problem. This function never invokes an optimizer."""
    x = features(rows)
    offset, matrix = affine_coefficients(expression, x)
    y = np.asarray([r["pressure_ratio"] for r in rows])
    physics = base["physics"]
    shape = physics["grid_shape"]
    variables = ("C", "E_over_p0", "k_over_p0")
    grid = np.meshgrid(*(np.linspace(*physics[name + "_interval"], n) for name, n in zip(variables, shape)), indexing="ij")
    po, pm = affine_coefficients(expression, {name: values.ravel() for name, values in zip(variables, grid)})
    do = np.diff(po.reshape(shape), axis=0).ravel()
    dm = np.diff(pm.reshape(tuple(shape) + (len(expression.constants),)), axis=0).reshape(-1, len(expression.constants))
    n = len(rows)
    A = np.vstack([np.column_stack([-matrix, np.zeros(n)]),
                   np.column_stack([matrix, -np.ones(n)]),
                   np.column_stack([-pm, np.zeros(len(po))]),
                   np.column_stack([pm, np.zeros(len(po))]),
                   np.column_stack([dm, np.zeros(len(do))])])
    b = np.concatenate([offset - y, y - offset, po, 1 - po, -do])
    boundary_count = shape[1] * shape[2]
    eq_A = np.column_stack([pm[:boundary_count], np.zeros(boundary_count)])
    eq_b = 1 - po[:boundary_count]
    objective = np.zeros(len(expression.constants) + 1)
    objective[-1] = 1.
    return {"c": objective, "A_ub": A, "b_ub": b, "A_eq": eq_A, "b_eq": eq_b,
            "bounds": [tuple(base["fitting"]["bounds"])] * len(expression.constants) + [(0., None)],
            "target_offset": offset, "target_matrix": matrix}

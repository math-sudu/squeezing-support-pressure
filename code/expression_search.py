"""Restricted expression search: shared parser, fitting, physics and selection."""
from __future__ import annotations

import ast
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
BINARY = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}


class ExpressionError(ValueError):
    pass


def settings():
    return json.loads((ROOT / "config/expression_search.json").read_text(encoding="utf-8"))


@dataclass
class Expression:
    text: str
    tree: ast.AST
    constants: list[str]
    nodes: int
    canonical: str
    grammar: dict

    @classmethod
    def parse(cls, text, grammar):
        if not isinstance(text, str) or len(text) > grammar["maximum_text_length"]:
            raise ExpressionError("Expression is not bounded text")
        try:
            tree = ast.parse(text, mode="eval").body
        except (SyntaxError, RecursionError) as exc:
            raise ExpressionError("Malformed expression") from exc
        constants = set()

        def visit(node):
            if isinstance(node, ast.Name):
                if node.id in grammar["constant_names"]:
                    constants.add(node.id)
                elif node.id not in grammar["variables"]:
                    raise ExpressionError("Unknown variable or constant: " + node.id)
                return 1
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if node.value not in grammar["fixed_literals"]:
                    raise ExpressionError("Use fitted constant slots for non-unit literals")
                return 1
            if isinstance(node, ast.BinOp) and BINARY.get(type(node.op)) in grammar["binary_operators"]:
                return 1 + visit(node.left) + visit(node.right)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in grammar["functions"] and len(node.args) == 1 and not node.keywords):
                return 1 + visit(node.args[0])
            raise ExpressionError("Unsupported syntax: " + type(node).__name__)

        try:
            nodes = visit(tree)
        except RecursionError as exc:
            raise ExpressionError("Expression too deep") from exc
        if nodes > grammar["maximum_nodes"]:
            raise ExpressionError("Node budget exceeded")
        return cls(text, tree, sorted(constants), nodes, ast.dump(tree), grammar)

    def evaluate(self, features, constants):
        if set(constants) != set(self.constants):
            raise ExpressionError("Constant identity mismatch")
        env = {**features, **constants}

        def visit(n):
            if isinstance(n, ast.Name):
                return np.asarray(env[n.id], dtype=float)
            if isinstance(n, ast.Constant):
                return float(n.value)
            if isinstance(n, ast.BinOp):
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Add):
                    return a+b
                if isinstance(n.op, ast.Sub):
                    return a-b
                if isinstance(n.op, ast.Mult):
                    return a*b
                if np.any(np.abs(b) <= self.grammar["denominator_tolerance"]):
                    raise ExpressionError("Division domain violation")
                return a/b
            a = visit(n.args[0])
            if n.func.id == "square":
                return np.square(a)
            if n.func.id == "sqrt":
                if np.any(a < 0):
                    raise ExpressionError("Square-root domain violation")
                return np.sqrt(a)
            if np.any(a <= -1):
                raise ExpressionError("Log1p domain violation")
            return np.log1p(a)

        try:
            with np.errstate(all="raise"):
                result = np.broadcast_to(visit(self.tree), np.asarray(features["C"]).shape).copy()
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise ExpressionError(str(exc)) from exc
        if not np.isfinite(result).all():
            raise ExpressionError("Nonfinite output")
        return result

    def continuity_certificate(self, domain, constants):
        """Outward-rounded interval arithmetic; sufficient, potentially conservative."""
        env = {**domain, **{k:(v,v) for k,v in constants.items()}}

        def bounds(lo, hi):
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ExpressionError("Unbounded interval")
            return float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf))

        def visit(n):
            if isinstance(n, ast.Name):
                return env[n.id]
            if isinstance(n, ast.Constant):
                return n.value, n.value
            if isinstance(n, ast.BinOp):
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Add):
                    return bounds(a[0]+b[0], a[1]+b[1])
                if isinstance(n.op, ast.Sub):
                    return bounds(a[0]-b[1], a[1]-b[0])
                if isinstance(n.op, ast.Div):
                    eps = self.grammar["denominator_tolerance"]
                    if b[0] <= eps and b[1] >= -eps:
                        raise ExpressionError("Denominator interval reaches zero")
                    b = bounds(1/b[1], 1/b[0])
                products = [x*y for x in a for y in b]
                interval = bounds(min(products), max(products))
                if (a[0] >= 0 and b[0] >= 0) or (a[1] <= 0 and b[1] <= 0):
                    return max(0, interval[0]), interval[1]
                return interval
            a = visit(n.args[0])
            if n.func.id == "square":
                interval = bounds(0 if a[0] <= 0 <= a[1] else min(x*x for x in a), max(x*x for x in a))
                return max(0, interval[0]), interval[1]
            if n.func.id == "sqrt":
                if a[0] < 0:
                    raise ExpressionError("Square-root interval reaches negative values")
                interval = bounds(math.sqrt(a[0]), math.sqrt(a[1]))
                return max(0, interval[0]), interval[1]
            if a[0] <= -1:
                raise ExpressionError("Log1p interval reaches -1")
            return bounds(math.log1p(a[0]), math.log1p(a[1]))

        try:
            interval = visit(self.tree)
            return {"certified": True, "output_interval": interval,
                    "scope": "Operation domains and continuity only; dependency overestimation retained."}
        except (ExpressionError, OverflowError, ValueError) as exc:
            return {"certified": False, "reason": str(exc)}


def features(rows):
    return {name: np.asarray([r[name] for r in rows], dtype=float) for name in ("C", "E_over_p0", "k_over_p0")}


def fit_constants(expression, rows, config):
    if not rows:
        raise ExpressionError("No fitting data")
    x, y = features(rows), np.asarray([r["pressure_ratio"] for r in rows])
    ids = [r["condition_id"] for r in rows]
    weights = np.asarray([1 / math.sqrt(ids.count(i)) for i in ids])
    counts = {"residual_calls": 0, "invalid_residual_calls": 0}
    starts, best = [], None
    begin = time.perf_counter()

    def residual(values):
        counts["residual_calls"] += 1
        try:
            p = expression.evaluate(x, dict(zip(expression.constants, values)))
            return (p-y)*weights
        except ExpressionError:
            counts["invalid_residual_calls"] += 1
            return np.full(len(y), config["invalid_trial_residual"])

    for init in (config["initial_values"] if expression.constants else [None]):
        attempt = {"initial_value": init}
        try:
            if expression.constants:
                result = least_squares(residual, np.full(len(expression.constants), init),
                                       bounds=config["bounds"], max_nfev=config["maximum_function_evaluations"])
                values = dict(zip(expression.constants, result.x.tolist()))
                attempt.update({"success": bool(result.success), "nfev": result.nfev, "message": result.message,
                                "jacobian_rank": int(np.linalg.matrix_rank(result.jac))})
            else:
                values = {}
                attempt.update({"success": True, "nfev": 0, "jacobian_rank": 0})
            p = expression.evaluate(x, values)
            cost = float(np.sum(((p-y)*weights)**2))
            attempt.update({"constants": values, "weighted_squared_error": cost})
            if attempt["success"] and (best is None or cost < best["weighted_squared_error"]):
                best = attempt
        except (ExpressionError, ValueError, FloatingPointError) as exc:
            attempt.update({"success": False, "reason": str(exc)})
        starts.append(attempt)
    return {"success": best is not None, "best": best, "starts": starts, **counts,
            "wall_seconds": time.perf_counter()-begin, "constant_count": len(expression.constants)}


def physics_check(expression, constants, config):
    domain = {name:config[name+"_interval"] for name in ("C", "E_over_p0", "k_over_p0")}
    grid = np.meshgrid(*(np.linspace(*domain[k], n) for k,n in zip(domain,config["grid_shape"])), indexing="ij")
    x = {k:v.ravel() for k,v in zip(domain,grid)}
    certificate = expression.continuity_certificate(domain, constants)
    try:
        y = expression.evaluate(x, constants).reshape(config["grid_shape"])
        out = ((y < -config["pressure_tolerance"]) | (y > 1+config["pressure_tolerance"]))
        increase = np.diff(y, axis=0)
        report = {"finite": True, "probe_count": y.size, "raw_min": float(y.min()), "raw_max": float(y.max()),
                  "range_violations": int(out.sum()), "boundary_max_error": float(np.abs(y[0]-1).max()),
                  "monotonic_violations": int((increase > config["monotonic_difference_tolerance"]).sum()),
                  "maximum_positive_pressure_increment": float(max(0, increase.max()))}
        report["passed"] = (not out.any() and report["boundary_max_error"] <= config["boundary_tolerance"]
                            and report["monotonic_violations"] == 0 and certificate["certified"])
    except ExpressionError as exc:
        report = {"passed": False, "finite": False, "reason": str(exc)}
    return {**report, "continuity": certificate, "projection_applied": False,
            "scope": "Sampled range/boundary/monotonicity, interval-certified operation domains; no full continuum monotonicity proof."}


def pressure_errors(expression, constants, rows):
    if not rows:
        return None
    p = expression.evaluate(features(rows), constants)
    error = p-np.asarray([r["pressure_ratio"] for r in rows])
    return {"points": len(rows), "mean_absolute_error": float(np.abs(error).mean()),
            "maximum_absolute_error": float(np.abs(error).max()), "mean_signed_error": float(error.mean()),
            "negative_pressure_errors": int((error < -1e-8).sum()),
            "minimum_signed_error": float(error.min()),
            "note": "Pointwise pressure errors against accepted FE states; a negative error is not an FE-verified convergence overshoot."}


class SearchSession:
    def __init__(self, fit_rows, selection_rows, config, maximum_attempts):
        self.fit_rows, self.selection_rows, self.config = fit_rows, selection_rows, config
        if {r["condition_id"] for r in fit_rows} & {r["condition_id"] for r in selection_rows}:
            raise ValueError("Fitting and selection condition identities overlap")
        self.maximum_attempts, self.attempts, self.seen = maximum_attempts, [], set()

    def submit(self, text, provenance):
        if len(self.attempts) >= self.maximum_attempts:
            raise ValueError("Candidate budget exhausted")
        row = {"attempt":len(self.attempts)+1, "expression":text, "provenance":provenance}
        self.attempts.append(row)
        start = time.perf_counter()
        try:
            expression = Expression.parse(text, self.config["grammar"])
            row.update({"nodes":expression.nodes, "canonical":expression.canonical})
            if expression.canonical in self.seen:
                row["status"] = "duplicate"
                return row
            self.seen.add(expression.canonical)
            fit = fit_constants(expression, self.fit_rows, self.config["fitting"])
            row["fit"] = fit
            if not fit["success"]:
                row["status"] = "fit_failed"
                return row
            constants = fit["best"]["constants"]
            row["physics"] = physics_check(expression, constants, self.config["physics"])
            row["fit_errors"] = pressure_errors(expression, constants, self.fit_rows)
            row["fit_by_condition"] = {i: pressure_errors(expression, constants, [r for r in self.fit_rows if r["condition_id"]==i])
                                       for i in sorted({r["condition_id"] for r in self.fit_rows})}
            row["selection_errors"] = pressure_errors(expression, constants, self.selection_rows)
            row["selection_by_condition"] = {i: pressure_errors(expression, constants, [r for r in self.selection_rows if r["condition_id"]==i])
                                              for i in sorted({r["condition_id"] for r in self.selection_rows})}
            row["status"] = "admissible" if row["physics"]["passed"] else "physical_rejection"
        except (ExpressionError, ValueError) as exc:
            row.update({"status": "invalid_expression", "reason": str(exc)})
        finally:
            row["wall_seconds"] = time.perf_counter()-start
        return row

    def pareto(self):
        candidates = [r for r in self.attempts if r["status"] == "admissible" and r["selection_errors"]]
        score = lambda r:(r["selection_errors"]["maximum_absolute_error"], r["nodes"])
        return [r["attempt"] for r in candidates if not any(
            all(a <= b for a,b in zip(score(s),score(r))) and any(a < b for a,b in zip(score(s),score(r)))
            for s in candidates)]


class GeneticProposals:
    """One bounded subtree-mutation/crossover GP; evaluation is always SearchSession."""
    def __init__(self, seed, grammar, population_size=16):
        self.rng = np.random.default_rng(seed)
        self.grammar, self.population_size, self.population = grammar, population_size, []

    def subtree(self, depth=2):
        if depth == 0 or self.rng.random() < .3:
            leaves = self.grammar["variables"] + self.grammar["constant_names"] + ["0", "1"]
            return ast.parse(str(self.rng.choice(leaves)), mode="eval").body
        if self.rng.random() < .25:
            return ast.Call(func=ast.Name(id=str(self.rng.choice(self.grammar["functions"])), ctx=ast.Load()),
                            args=[self.subtree(depth-1)], keywords=[])
        return ast.BinOp(left=self.subtree(depth-1), op=[ast.Add(), ast.Sub(), ast.Mult(), ast.Div()][int(self.rng.integers(4))],
                         right=self.subtree(depth-1))

    def parent(self):
        pool = [self.population[int(i)] for i in self.rng.integers(len(self.population), size=min(3,len(self.population)))]
        return copy.deepcopy(min(pool, key=lambda r:r[0])[1])

    def propose(self):
        tree = self.parent() if self.population else self.subtree(3)
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.Call, ast.Name, ast.Constant))]
        target = nodes[int(self.rng.integers(len(nodes)))]
        replacement = self.parent() if self.population and self.rng.random() < .5 else self.subtree(2)
        class Replace(ast.NodeTransformer):
            def visit(self, node):
                return replacement if node is target else super().visit(node)
        return ast.unparse(ast.fix_missing_locations(Replace().visit(tree)))

    def observe(self, row):
        if row["status"] != "admissible":
            return
        error = row["selection_errors"] or row["fit_errors"]
        self.population.append(((error["maximum_absolute_error"], row["nodes"]), ast.parse(row["expression"], mode="eval").body))
        self.population.sort(key=lambda p:p[0])
        self.population = self.population[:self.population_size]

"""Bracket the inverse only on an accepted monotonic unloading path."""
import math


def closure(row):
    chords = [float(row["width_closure"]), float(row["height_closure"])]
    if not all(math.isfinite(v) for v in chords):
        raise ValueError("Nonfinite chord measurement")
    return max(chords)


def bracket_pressure(records, allowable):
    if not math.isfinite(allowable) or allowable < 0:
        raise ValueError("Allowable closure must be finite and nonnegative")
    if not records or any(not row.get("accepted") for row in records):
        raise ValueError("An accepted path is required")
    values = [closure(row) for row in records]
    pressures = [float(row["pressure_ratio"]) for row in records]
    if not all(math.isfinite(v) for v in values + pressures):
        raise ValueError("Nonfinite path")
    if any(p < 0 or p > 1 for p in pressures):
        raise ValueError("Pressure ratio lies outside the prescribed unloading domain")
    for i in range(1, len(records)):
        if pressures[i] >= pressures[i-1] or values[i] < values[i-1] - 1e-12:
            raise ValueError("Path is not monotonic unloading with increasing closure")
        if values[i] == values[i-1] == allowable:
            raise ValueError("Flat interval does not identify a unique inverse")
    for i in range(1, len(records)):
        if values[i-1] - 1e-12 <= allowable <= values[i] + 1e-12:
            if values[i] == values[i-1]:
                raise ValueError("Flat interval does not identify a unique inverse")
            f = min(1., max(0., (allowable - values[i-1]) / (values[i] - values[i-1])))
            return {"previous_index": i-1, "next_index": i,
                    "pressure_lower": pressures[i], "pressure_upper": pressures[i-1],
                    "interpolated_pressure": pressures[i-1] + f*(pressures[i]-pressures[i-1]),
                    "allowable": allowable}
    return None


def closure_margins(observation, allowable):
    """Signed chord margins; a positive overshoot is retained at any size."""
    maximum = closure(observation)
    result = {"allowable": allowable, "actual_max_closure": maximum,
              "signed_max_closure_mismatch": maximum - allowable,
              "positive_overshoot": max(0., maximum - allowable),
              "raw_positive_mismatch": maximum > allowable,
              "controlling_chord": "height" if observation["height_closure"] >= observation["width_closure"] else "width"}
    for chord in ("width", "height"):
        length = float(observation[f"initial_{chord}_m"])
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Invalid initial chord length")
        delta = float(observation[f"{chord}_closure"]) - allowable
        result[f"{chord}_signed_margin"] = -delta
        result[f"{chord}_signed_overshoot_mm"] = delta * length * 1000
    return result

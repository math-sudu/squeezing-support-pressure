# Data and code for support-pressure design in horseshoe tunnels

This supplement accompanies **Explicit support-pressure design for horseshoe
tunnels under paired convergence constraints**, by Pengcheng Zhu and Tielin Chen.

The archive contains the numerical inputs, pressure relations, source code and
tabulated observations used to check the reported pressure-design and directional
plastic-moment results. It includes the recorded expression proposals and their
mechanical construction settings.

## Reproduce the reported numerical summaries

Use Python 3.10 or newer. From the extracted archive directory:

```text
python -m pip install -r requirements.txt
python reproduce.py
```

The script reconstructs and solves the three linear programs from 147 pressure
targets, evaluates the pressure expressions at 39 material-allowance inputs,
checks both directional margins for all 117 recorded finite-element evaluations,
and recomputes the unequal-step plastic-moment errors from the tabulated
predictions and observations. It writes no files and makes no network calls.
The reproduction script needs NumPy and SciPy; PyTorch is used by the supplied
finite-element, PINN and constitutive implementations. The supplied script was
checked with NumPy 2.4.4 and SciPy 1.16.2; the mechanics code was developed with
PyTorch 2.8.0 in double precision.

## Contents

| File or directory | Contents |
|---|---|
| `data/pressure_problem.json` | All 147 construction targets, the expression grammar and linear-program settings |
| `data/pressure_relations.json` | Expressions A-C, their five final coefficients and minimax objectives |
| `data/pressure_validation.csv` | 81 development and 36 interior-holdout finite-element observations, with pressure, normalized closure and dimensional margins |
| `data/forward_unequal_step.json` | Two unequal unloading increments, their independently recorded predictions, PINN observations and cumulative errors |
| `data/forward_construction.json` | Construction-interval results for the directional activation model |
| `data/forward_proposal.json` | The proposed coupled equation, regional displacement span and material-state update |
| `data/pressure_coefficient_bounds.json` | Coefficient-box and optimal-solution sensitivity results |
| `data/critical_clearance/` | Recorded time-step and spatial-resolution observations at the critical pressure query |
| `data/pinn_tables/` | Directional convergence, plastic-moment and regional summaries at the saved pressure stages |
| `data/pressure_proposals/` | Recorded pressure-expression proposals for the three search seeds |
| `config/` | Geometry, materials, expression-search and finite-element configurations |
| `figure_data/` | CSV data used in the pressure, material-coverage and clearance-margin figures |
| `code/` | Expression evaluation, linear-program construction, finite-element solver, directional projection, material-return and three-mode model implementations |
| `code/vendor/saved_pinn/` | PINN field representation, history and constitutive implementations |

## Variables and units

`C` or `allowable` is the common normalized width-height closure limit.
`E_over_p0` is Young's modulus divided by the initial pressure; `k_over_p0` is
the strength intercept divided by that pressure. `pressure_ratio` is maintained
wall pressure divided by initial pressure. These four quantities are dimensionless.
Coordinates and displacements in the mechanics code are in metres. Columns ending
in `_mm` are in millimetres. A positive directional margin is remaining allowance:
`(allowable - closure) * initial_chord_length_m * 1000`.

Relations A, B and C correspond to seeds 1729, 2718 and 3141. The validation table
contains three relations at each of 39 material-allowance inputs, including 12
inputs from four interior-holdout materials. Its observations are finite-element
outputs, not values inferred from the fitted pressure expressions.

In the forward tables, width and height are the two output directions; tensor
components are ordered `xx`, `xy`, `yy`, `zz`. Plastic moments, actual chord
shortening, pressure projections and residual contributions are separate
quantities. Their names and units are retained in the JSON and CSV files.

## Scope of this compact archive

The reproduction command evaluates the formulas and recomputes statistics from
the supplied numerical observations. It does not repeat the original finite-element
or PINN solves. Full field arrays, solver checkpoints, trained-state checkpoints,
exported state-dependent model arrays, and reconstructible linear-program matrices
are omitted. The numerical implementations and configurations are supplied for
further computation; running their low-level functions requires constructing the
corresponding state and loading history.

Correspondence: Tielin Chen, tlchen1@bjtu.edu.cn.

# CPModel-JAX

Exact-hat counterfactuals for the multi-country, multi-sector model of
[Caliendo and Parro (2015)](https://doi.org/10.1093/restud/rdu035).

The solver uses JAX automatic differentiation and a matrix-free Newton–Krylov method.

## Install

Python 3.12+ is required.

```sh
uv add git+https://github.com/YutaroKimata/CPModel-JAX.git
```

## Use

```python
from cpmodel_jax import solve_eha

result = solve_eha(**inputs)
print(100 * (result["welfare_ratio"] - 1))
```

## Inputs

`N` is the number of countries and `J` the number of sectors.

Bilateral arrays use **importer, exporter, sector** axes. IO shares use **country, output sector, input sector** axes.

| Key | Shape |
| --- | --- |
| `theta` | `(J,)` |
| `alpha` | `(N,J)` |
| `beta` | `(N,J)` |
| `gamma` | `(N,J,J)` |
| `baseline_net_trade_value` | `(N,N,J)` |
| `baseline_tariff_rates` | `(N,N,J)` |
| `counterfactual_tariff_rates` | `(N,N,J)` |
| `iceberg_cost_ratio` | `(N,N,J)` |
| `technology_scale_ratio` | `(N,J)` |

The last three are optional counterfactual shocks.

## Outputs

Ratios are counterfactual relative to baseline.

| Key | Shape |
| --- | --- |
| `wage_ratio` | `(N,)` |
| `sector_price_ratio` | `(N,J)` |
| `unit_cost_ratio` | `(N,J)` |
| `trade_share_ratio` | `(N,N,J)` |
| `expenditure_ratio` | `(N,J)` |
| `trade_value_ratio` | `(N,N,J)` |
| `net_trade_value_ratio` | `(N,N,J)` |
| `output_ratio` | `(N,J)` |
| `income_ratio` | `(N,)` |
| `consumer_price_ratio` | `(N,)` |
| `welfare_ratio` | `(N,)` |
| `real_wage_ratio` | `(N,)` |

## NAFTA example

The included example reproduces the NAFTA tariff counterfactual in Caliendo and Parro (2015):

```sh
uv run examples/nafta_example.py
```

The authors' replication data are downloaded automatically. The test suite verifies the welfare decomposition against Table 2 at its published precision.

## Development

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run python -m build
```

MIT license; see [LICENSE](LICENSE).

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)


def _prepare(
    theta,
    alpha,
    beta,
    gamma,
    baseline_net_trade_value,
    baseline_tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=None,
    technology_scale_ratio=None,
):
    counterfactual_tariff_rates = baseline_tariff_rates if counterfactual_tariff_rates is None else counterfactual_tariff_rates
    log_iceberg_cost_ratio = np.log(1.0 if iceberg_cost_ratio is None else iceberg_cost_ratio)
    log_technology_ratio = np.log(1.0 if technology_scale_ratio is None else technology_scale_ratio)

    baseline_output = baseline_net_trade_value.sum(axis=0)
    trade_weights = baseline_net_trade_value * (1.0 + baseline_tariff_rates)
    baseline_expenditure = trade_weights.sum(axis=1)
    trade_weights /= baseline_expenditure[:, None, :]

    with np.errstate(divide="ignore"):
        log_trade_weights = np.log(trade_weights, out=trade_weights)

    baseline_value_added = (beta * baseline_output).sum(axis=1)
    baseline_exports = baseline_output.sum(axis=1)
    baseline_net_imports = baseline_net_trade_value.sum(axis=(1, 2))
    baseline_gross_imports = baseline_expenditure.sum(axis=1)
    baseline_deficit = baseline_net_imports - baseline_exports
    baseline_income = baseline_value_added + baseline_gross_imports - baseline_exports

    log_tariff_ratio = np.log1p(counterfactual_tariff_rates) - np.log1p(baseline_tariff_rates)
    log_trade_cost_ratio = log_tariff_ratio + log_iceberg_cost_ratio
    log_cost_shock = log_technology_ratio - theta * log_trade_cost_ratio
    log_trade_weights += log_cost_shock

    counterfactual_net_factor = 1.0 / (1.0 + counterfactual_tariff_rates)
    net_trade_value_ratio_factor = (1.0 + baseline_tariff_rates) * counterfactual_net_factor

    data = {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "baseline_output": baseline_output,
        "baseline_value_added": baseline_value_added,
        "baseline_deficit": baseline_deficit,
        "baseline_income": baseline_income,
        "baseline_expenditure": baseline_expenditure,
        "counterfactual_net_factor": counterfactual_net_factor,
        "net_trade_value_ratio_factor": net_trade_value_ratio_factor,
        "log_trade_weights": log_trade_weights,
        "log_cost_shock": log_cost_shock,
    }
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float64), data)


def _state(log_ratios, data):
    n, j = data["beta"].shape
    log_wage_ratio = log_ratios[:n]
    log_sector_price_ratio = log_ratios[n : n + n * j].reshape(n, j)
    log_expenditure_ratio = log_ratios[n + n * j :].reshape(n, j)

    wage_ratio = jnp.exp(log_wage_ratio)
    expenditure_ratio = jnp.exp(log_expenditure_ratio)
    counterfactual_expenditure = data["baseline_expenditure"] * expenditure_ratio

    log_intermediate_cost_ratio = jnp.einsum("njk,nk->nj", data["gamma"], log_sector_price_ratio)
    log_unit_cost_ratio = data["beta"] * log_wage_ratio[:, None] + log_intermediate_cost_ratio

    log_trade_weights = data["log_trade_weights"] - data["theta"] * log_unit_cost_ratio[None, :, :]
    log_share_denominator = logsumexp(log_trade_weights, axis=1)
    counterfactual_trade_shares = jnp.exp(log_trade_weights - log_share_denominator[:, None, :])

    counterfactual_trade_value = counterfactual_trade_shares * counterfactual_expenditure[:, None, :]
    counterfactual_net_trade_value = counterfactual_trade_value * data["counterfactual_net_factor"]

    counterfactual_output = counterfactual_net_trade_value.sum(axis=0)
    counterfactual_labor_income = wage_ratio * data["baseline_value_added"]
    counterfactual_labor_demand = (data["beta"] * counterfactual_output).sum(axis=1)
    counterfactual_net_imports = counterfactual_net_trade_value.sum(axis=(1, 2))
    counterfactual_gross_imports = counterfactual_expenditure.sum(axis=1)
    counterfactual_tariff_revenue = counterfactual_gross_imports - counterfactual_net_imports
    counterfactual_income = counterfactual_labor_income + data["baseline_deficit"] + counterfactual_tariff_revenue

    return {
        "wage_ratio": wage_ratio,
        "expenditure_ratio": expenditure_ratio,
        "log_unit_cost_ratio": log_unit_cost_ratio,
        "log_share_denominator": log_share_denominator,
        "counterfactual_trade_shares": counterfactual_trade_shares,
        "counterfactual_expenditure": counterfactual_expenditure,
        "counterfactual_trade_value": counterfactual_trade_value,
        "counterfactual_net_trade_value": counterfactual_net_trade_value,
        "counterfactual_output": counterfactual_output,
        "counterfactual_labor_income": counterfactual_labor_income,
        "counterfactual_labor_demand": counterfactual_labor_demand,
        "counterfactual_income": counterfactual_income,
    }


@jax.jit
def residual(log_ratios, data):
    n, j = data["beta"].shape
    log_wage_ratio = log_ratios[:n]
    log_sector_price_ratio = log_ratios[n : n + n * j].reshape(n, j)

    state = _state(log_ratios, data)

    log_wage_ratio_target = jnp.log(state["counterfactual_labor_demand"]) - jnp.log(data["baseline_value_added"])
    raw_wage_residual = log_wage_ratio - log_wage_ratio_target
    wage_residual = raw_wage_residual[:-1] - raw_wage_residual[-1]

    numeraire_residual = jnp.log(state["counterfactual_labor_income"].sum()) - jnp.log(data["baseline_value_added"].sum())

    price_residual = log_sector_price_ratio + state["log_share_denominator"] / data["theta"]

    counterfactual_final_demand = data["alpha"] * state["counterfactual_income"][:, None]
    counterfactual_intermediate_demand = jnp.einsum("nkj,nk->nj", data["gamma"], state["counterfactual_output"])
    counterfactual_demand = counterfactual_final_demand + counterfactual_intermediate_demand
    expenditure_residual = (state["counterfactual_expenditure"] - counterfactual_demand) / data["baseline_expenditure"]

    residual_value = jnp.concatenate((
        wage_residual,
        numeraire_residual.reshape(1),
        price_residual.ravel(),
        expenditure_residual.ravel()
    ))

    return jnp.where(jnp.all(state["counterfactual_income"] > 0), residual_value, jnp.nan)


def _preconditioner(data, z):
    n, j = data["beta"].shape
    state = _state(z, data)

    wage_weights = state["counterfactual_labor_income"]
    wage_weights /= wage_weights.sum()

    index = jnp.arange(n)
    home_share = state["counterfactual_trade_shares"][index, index, :]

    expenditure_inverse_diag = data["baseline_expenditure"] / state["counterfactual_expenditure"]

    def apply(v):
        vw = v[:n]
        vp = v[n : n + n * j].reshape(n, j)
        vx = v[n + n * j :].reshape(n, j)

        du_last = vw[-1] - jnp.dot(wage_weights[:-1], vw[:-1])
        mw = jnp.concatenate((vw[:-1] + du_last, du_last.reshape(1)))

        io_effect = jnp.einsum("njk,nk->nj", data["gamma"], vp)
        mp = vp + home_share * io_effect

        mx = expenditure_inverse_diag * vx

        return jnp.concatenate((mw, mp.ravel(), mx.ravel()))

    return apply


@jax.jit(static_argnames=("f",))
def _evaluate(f, data, z):
    residual_value = f(z, data)
    return residual_value, jnp.max(jnp.abs(residual_value)), jnp.linalg.norm(residual_value)


def _forcing_term(residual_norm, previous_norm, previous_eta):
    if previous_norm is None:
        return previous_eta

    alpha = (1 + np.sqrt(5)) / 2

    eta = 0.9 * float(residual_norm / previous_norm) ** alpha
    safeguard = 0.9 * previous_eta**alpha

    if safeguard > 0.1:
        eta = max(eta, safeguard)

    return min(eta, 0.9)


@jax.jit(static_argnames=("f", "preconditioner"))
def _direction(f, data, z, eta, preconditioner=None):
    residual_value, jvp = jax.linearize(lambda x: f(x, data), z)

    M = None if preconditioner is None else preconditioner(data, z)

    return gmres(jvp, -residual_value, M=M, tol=eta, solve_method="incremental")[0]


def _line_search(f, data, z, direction, residual_norm, max_trials):
    residual_norm = float(residual_norm)
    step_size = 1.0

    for _ in range(max_trials):
        next_z = z + step_size * direction
        next_residual, next_error, next_norm = _evaluate(f, data, next_z)

        if float(next_norm) <= (1 - 1e-4 * step_size) * residual_norm:
            return next_z, next_residual, next_error, next_norm

        step_size *= 0.5

    raise RuntimeError("line search failed")


def newton_krylov(
    f,
    data,
    z0,
    *,
    tol=1e-5,
    max_iter=60,
    max_trials=25,
    preconditioner=None,
):
    z = jnp.asarray(z0, dtype=jnp.float64)
    residual_value, error, residual_norm = _evaluate(f, data, z)

    previous_norm = None
    eta = 0.1
    iteration = 0

    while iteration < max_iter and float(error) > tol:
        eta = _forcing_term(residual_norm, previous_norm, eta)
        direction = _direction(f, data, z, eta, preconditioner)

        previous_norm = residual_norm
        z, residual_value, error, residual_norm = _line_search(f, data, z, direction, residual_norm, max_trials)
        iteration += 1

    if not float(error) <= tol:
        raise RuntimeError("Newton-Krylov did not converge")

    return {
        "z": z,
        "residual": residual_value,
        "iterations": iteration,
    }


@jax.jit
def _results(log_ratios, data):
    n, j = data["beta"].shape
    log_sector_price_ratio = log_ratios[n : n + n * j].reshape(n, j)

    state = _state(log_ratios, data)

    sector_price_ratio = jnp.exp(log_sector_price_ratio)
    unit_cost_ratio = jnp.exp(state["log_unit_cost_ratio"])

    log_trade_weight_ratio = data["log_cost_shock"] - data["theta"] * state["log_unit_cost_ratio"][None, :, :]
    trade_share_ratio = jnp.exp(log_trade_weight_ratio - state["log_share_denominator"][:, None, :])
    trade_value_ratio = trade_share_ratio * state["expenditure_ratio"][:, None, :]
    net_trade_value_ratio = trade_value_ratio * data["net_trade_value_ratio_factor"]

    output_ratio = state["counterfactual_output"] / data["baseline_output"]
    income_ratio = state["counterfactual_income"] / data["baseline_income"]
    consumer_price_ratio = jnp.exp((data["alpha"] * log_sector_price_ratio).sum(axis=1))
    welfare_ratio = income_ratio / consumer_price_ratio
    real_wage_ratio = state["wage_ratio"] / consumer_price_ratio

    labor_error = jnp.max(jnp.abs(state["counterfactual_labor_demand"] / state["counterfactual_labor_income"] - 1))

    values = {
        "wage_ratio": state["wage_ratio"],
        "sector_price_ratio": sector_price_ratio,
        "unit_cost_ratio": unit_cost_ratio,
        "trade_share_ratio": trade_share_ratio,
        "expenditure_ratio": state["expenditure_ratio"],
        "trade_value_ratio": trade_value_ratio,
        "net_trade_value_ratio": net_trade_value_ratio,
        "output_ratio": output_ratio,
        "income_ratio": income_ratio,
        "consumer_price_ratio": consumer_price_ratio,
        "welfare_ratio": welfare_ratio,
        "real_wage_ratio": real_wage_ratio,
        "counterfactual_trade_shares": state["counterfactual_trade_shares"],
        "counterfactual_expenditure": state["counterfactual_expenditure"],
        "counterfactual_trade_value": state["counterfactual_trade_value"],
        "counterfactual_net_trade_value": state["counterfactual_net_trade_value"],
        "counterfactual_output": state["counterfactual_output"],
        "counterfactual_income": state["counterfactual_income"],
    }
    return values, labor_error


def solve_eha(
    *,
    theta,
    alpha,
    beta,
    gamma,
    baseline_net_trade_value,
    baseline_tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=None,
    technology_scale_ratio=None,
    initial_log_ratios=None,
    tolerance=1e-5,
    max_iter=60,
    max_trials=25,
):
    start = perf_counter()

    data = _prepare(
        theta,
        alpha,
        beta,
        gamma,
        baseline_net_trade_value,
        baseline_tariff_rates,
        counterfactual_tariff_rates,
        iceberg_cost_ratio,
        technology_scale_ratio,
    )
    n, j = data["beta"].shape

    if initial_log_ratios is None:
        initial_log_ratios = jnp.zeros(n + 2 * n * j, dtype=jnp.float64)

    root = newton_krylov(
        residual,
        data,
        initial_log_ratios,
        tol=tolerance,
        max_iter=max_iter,
        max_trials=max_trials,
        preconditioner=_preconditioner,
    )

    log_ratios = root["z"]
    values, labor_error = _results(log_ratios, data)

    values = jax.device_get(values)

    diagnostics = {
        "residual_inf": float(jnp.max(jnp.abs(root["residual"]))),
        "labor_error": float(labor_error),
    }

    return {
        **values,
        "diagnostics": diagnostics,
        "iterations": root["iterations"],
        "log_ratios": np.asarray(log_ratios),
        "residual": np.asarray(root["residual"]),
        "wall_seconds": perf_counter() - start,
    }

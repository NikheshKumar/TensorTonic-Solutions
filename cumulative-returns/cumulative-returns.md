## What Are Cumulative Returns?

Cumulative returns represent the total return on an investment over a period, accounting for compounding effects. They show how much an initial investment has grown or declined over time.

$$
R_{\text{cum}}(t) = \prod_{i=1}^{t} (1 + r_i) - 1
$$

where $r_i$ is the return at period $i$.

---

## The Formula

For a series of periodic returns $r_1, r_2, ..., r_T$:

$$
R_{\text{cum}} = (1 + r_1)(1 + r_2)...(1 + r_T) - 1
$$

**Alternative form (log returns):**

If using log returns $\ell_i = \ln(1 + r_i)$:

$$
R_{\text{cum}} = e^{\sum_{i=1}^{T} \ell_i} - 1
$$

Log returns sum to give cumulative return.

---

## Worked Example

**Daily returns:** [0.02, -0.01, 0.03, 0.01, -0.02]

**Step-by-step calculation:**

Period 1: $R_1 = 1.02 - 1 = 0.02$ (2%)

Period 2: $R_2 = 1.02 \times 0.99 - 1 = 1.0098 - 1 = 0.0098$ (0.98%)

Period 3: $R_3 = 1.0098 \times 1.03 - 1 = 1.040094 - 1 = 0.040094$ (4.01%)

Period 4: $R_4 = 1.040094 \times 1.01 - 1 = 1.0505 - 1 = 0.0505$ (5.05%)

Period 5: $R_5 = 1.0505 \times 0.98 - 1 = 1.0295 - 1 = 0.0295$ (2.95%)

**Final cumulative return:** 2.95%

**Verification:**

$$
R_{\text{cum}} = 1.02 \times 0.99 \times 1.03 \times 1.01 \times 0.98 - 1 = 1.0295 - 1 = 0.0295
$$

---

## Simple vs Compounded Returns

**Simple sum (incorrect):**

$$
0.02 + (-0.01) + 0.03 + 0.01 + (-0.02) = 0.03 = 3\%
$$

**Compounded (correct):**

$$
2.95\%
$$

**Difference:** Compounding accounts for returns earning returns.

**Small returns:** Difference is minimal.

**Large returns:** Difference is substantial.

---

## Initial Value Normalization

If starting with principal $P_0$:

$$
V_t = P_0 \prod_{i=1}^{t} (1 + r_i)
$$

**Normalized to 1:**

$$
V_t = \prod_{i=1}^{t} (1 + r_i)
$$

This is the growth factor. Subtract 1 to get cumulative return.

**Interpretation:** $V_t = 1.0295$ means $1 grew to $1.0295.

---

## Cumulative Return Series

Compute cumulative return at each time point:

$$
R_{\text{cum}}(t) = \prod_{i=1}^{t} (1 + r_i) - 1
$$

**Example series:**

- $t=1$: $R_1 = 0.02$
- $t=2$: $R_2 = 0.0098$
- $t=3$: $R_3 = 0.040094$
- $t=4$: $R_4 = 0.0505$
- $t=5$: $R_5 = 0.0295$

**Visual:** Plot shows investment growth trajectory over time.

---

## Maximum Drawdown Connection

Maximum drawdown measures peak-to-trough decline:

$$
\text{MDD} = \max_{t} \left[\max_{s \leq t} R_{\text{cum}}(s) - R_{\text{cum}}(t)\right]
$$

Requires cumulative returns to identify peak and subsequent trough.

**Use case:** Risk assessment. Shows worst loss from peak.

---

## Annualized Returns

Convert cumulative return to annualized rate:

$$
r_{\text{annual}} = \left(1 + R_{\text{cum}}\right)^{\frac{1}{T}} - 1
$$

where $T$ is the number of years.

**Example:** 10% cumulative return over 2 years:

$$
r_{\text{annual}} = (1.10)^{0.5} - 1 = 1.0488 - 1 = 0.0488 = 4.88\%
$$

**Interpretation:** Average annual rate that produces the cumulative return.

---

## Logarithmic vs Arithmetic Returns

**Arithmetic returns:**

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

**Logarithmic returns:**

$$
\ell_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(1 + r_t)
$$

**Cumulative conversion:**

Arithmetic: $(1 + r_1)(1 + r_2)...(1 + r_T) - 1$

Logarithmic: $e^{\ell_1 + \ell_2 + ... + \ell_T} - 1$

**Advantage of log returns:** Additive property simplifies calculations.

---

## Portfolio Cumulative Returns

For portfolio with weights $w_i$ and asset returns $r_{i,t}$:

$$
r_{p,t} = \sum_{i=1}^{N} w_i r_{i,t}
$$

**Portfolio cumulative return:**

$$
R_{p,\text{cum}} = \prod_{t=1}^{T} (1 + r_{p,t}) - 1
$$

**Note:** Portfolio cumulative return is NOT the weighted average of individual cumulative returns.

Must compound portfolio period returns.

---

## Time-Weighted vs Money-Weighted Returns

**Time-weighted (geometric):**

$$
R_{\text{TW}} = \prod_{t=1}^{T} (1 + r_t) - 1
$$

Measures investment performance independent of cash flows.

**Money-weighted (IRR):**

Solves:

$$
0 = \sum_{t=0}^{T} \frac{CF_t}{(1 + R_{\text{MW}})^t}
$$

where $CF_t$ includes contributions and withdrawals.

**Use case:** Time-weighted for comparing fund managers. Money-weighted for investor's actual return.

---

## Volatility and Cumulative Returns

Given periodic return volatility $\sigma$:

**Expected cumulative return (approximate):**

$$
E[R_{\text{cum}}] \approx T \mu - \frac{T \sigma^2}{2}
$$

where $\mu$ is mean periodic return.

**Volatility drag:** Higher volatility reduces cumulative returns due to compounding of losses.

**Example:** 10% average return with 20% volatility produces less than 10% annualized over long periods.

---

## Benchmark Comparison

**Relative cumulative return:**

$$
R_{\text{rel}} = \frac{1 + R_{\text{asset}}}{1 + R_{\text{benchmark}}} - 1
$$

**Interpretation:**

- $R_{\text{rel}} > 0$: Outperformed benchmark
- $R_{\text{rel}} < 0$: Underperformed benchmark

**Example:** Asset returned 15%, benchmark returned 10%:

$$
R_{\text{rel}} = \frac{1.15}{1.10} - 1 = 0.0455 = 4.55\%
$$

---

## Reinvestment Assumption

Cumulative returns assume all gains are reinvested:

**Dividends:** Automatically reinvested at prevailing price.

**Interest:** Compounded rather than withdrawn.

**No withdrawals:** Full capital remains invested.

**Reality:** Actual investor returns may differ due to consumption, taxes, fees.

---

## Multi-Period Decomposition

Break cumulative return into components:

$$
1 + R_{\text{cum}} = (1 + r_1)(1 + r_2)...(1 + r_T)
$$

**Attribution analysis:** Which periods contributed most to total return?

**Example:** Identify that 80% of gains occurred in 3 specific months.

**Application:** Performance attribution, understanding return drivers.

---

## Sharpe Ratio with Cumulative Returns

Sharpe ratio uses periodic returns:

$$
S = \frac{\mu - r_f}{\sigma}
$$

**Not directly computable from cumulative return alone.**

Need full return series to calculate mean and standard deviation.

**Common mistake:** Using only starting and ending values loses information about volatility path.

---

## Cumulative Returns in Backtesting

**Strategy evaluation:**

1. Generate trading signals
2. Compute period returns based on positions
3. Calculate cumulative returns
4. Compare to buy-and-hold

**Equity curve:** Plot of cumulative returns over time.

**Metrics derived:**

- Total return
- Maximum drawdown
- Sharpe ratio
- Calmar ratio (return / max drawdown)

---

## Transaction Costs Impact

Each trade incurs cost $c$:

$$
r_{t,\text{net}} = r_{t,\text{gross}} - c
$$

**Cumulative impact:**

$$
R_{\text{cum,net}} = \prod_{t=1}^{T} (1 + r_t - c_t) - 1
$$

**High-frequency trading:** Small per-trade costs compound to significant drag.

**Example:** 0.1% cost per trade, 100 trades:

$$
(1 - 0.001)^{100} = 0.9048
$$

9.5% loss from costs alone.

---

## Survivorship Bias

Historical cumulative returns often suffer from survivorship bias:

**Bias:** Only successful assets remain in dataset.

**Result:** Overstated historical returns.

**Example:** Mutual fund database includes only funds that survived. Failed funds excluded.

**Correction:** Include delisted and failed investments.

---

## Distributional Properties

For log returns $\ell_t \sim N(\mu, \sigma^2)$:

$$
\sum_{t=1}^{T} \ell_t \sim N(T\mu, T\sigma^2)
$$

**Cumulative return distribution:**

$$
R_{\text{cum}} = e^{\sum \ell_t} - 1
$$

follows log-normal distribution (shifted and scaled).

**Implications:** Positive skew, fat right tail, bounded below at -1.

---

## Real vs Nominal Returns

**Nominal return:** Raw return without inflation adjustment.

**Real return:** Inflation-adjusted return.

$$
r_{\text{real}} = \frac{1 + r_{\text{nominal}}}{1 + i} - 1
$$

where $i$ is inflation rate.

**Cumulative real return:**

$$
R_{\text{real,cum}} = \frac{1 + R_{\text{nominal,cum}}}{\prod_{t=1}^{T} (1 + i_t)} - 1
$$

**Interpretation:** Actual purchasing power change.

---

## Geometric Mean Return

The per-period geometric mean return:

$$
\bar{r}_g = \left(\prod_{t=1}^{T} (1 + r_t)\right)^{\frac{1}{T}} - 1
$$

**Relationship to cumulative return:**

$$
1 + R_{\text{cum}} = (1 + \bar{r}_g)^T
$$

**Interpretation:** Constant per-period return that produces the same cumulative return.

**Property:** $\bar{r}_g \leq \bar{r}_a$ (geometric mean $\leq$ arithmetic mean).

---

## Excess Returns

Return above risk-free rate:

$$
r_{t,\text{excess}} = r_t - r_{f,t}
$$

**Cumulative excess return:**

$$
R_{\text{excess,cum}} = \frac{1 + R_{\text{cum}}}{\prod_{t=1}^{T} (1 + r_{f,t})} - 1
$$

**Use case:** Evaluating active management. Did manager beat risk-free alternative?

---

## Rolling Cumulative Returns

Compute cumulative return over rolling windows:

$$
R_{\text{cum}}(t, w) = \prod_{i=t-w+1}^{t} (1 + r_i) - 1
$$

**Example:** 12-month rolling cumulative return.

**Application:** Visualize performance over various time horizons. Identify consistent vs sporadic outperformance.
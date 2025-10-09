---
layout: cover
mdc: true
author: Gerardo Duran-Martin
date: 2025-10-10
title: Bayesian adaptation and sequential decision making in financial environments
info: Kalman filtering for Bayesian continual learning
transition: none
theme: academic
coverAuthor: Gerardo Duran-Martin - OMI, University of Oxford
coverAuthorUrl: https://grdm.io
# class: text-white
---

# Bayesian learning, adaptation, and decision making in financial environments


---
layout: center
zoom: 1.5
---

## Joint work with

```
Á Cartea        F Drissi                K Murphy
G Palmari       L Sanchez-Betancourt    AY Shestopaloff
```

---

## Machine learning, decision making, and financial environments

We construct data-driven (machine learning) approaches for decision making in finance.

<v-clicks>

* More data $\not\rightarrow$ better performance (lookback windows for re-training).

* Underlying assumptions in model change (regime changes).

* We are not considering the correct modelling assumptions (misspecification).

* Low signal-to-noise ratio (SNR).

</v-clicks>

<v-click> 

We propose 
(generalised) Bayesian approaches to **learning**, **adaptation**, and sequential **decision making** in misspecified and non-stationary environments.

</v-click> 

---

## Setup

The agent observes a stream of rewards $r_t \in \reals$ and states (contexts) $s_{t+1} \in {\cal S}$
from the environment after choosing action $a_t \in {\cal A}$ and being at state $s_t \in {\cal S}$.

$$
    (r_t, s_{t+1}) \sim p_{\rm env}(\cdot \mid s_t, a_t).
$$

$$
    \underbrace{s_1, a_1, r_1}_{\data_1}, \underbrace{s_2, a_2, r_2}_{\data_2} \ldots, 
$$

The environment $p_{\rm env}$ is unknown to the agent and the **agent does not model it** (model-free / no planning).

<!-- | Env | reward $(r_t)$ | state $(s_t)$ | method
| --- | --- | --- | --- |
| market maker | ... | inventory, cash | RL |
| market maker with negligible volume | ... | volume imbalance, inventory | Bandit |
| Trading agents: choose among $K$ possible strategies | return of trading strategy e.o.d. | LOB, past performance, macroeconomic variables | Bandit

$-->

Agent seeks to maximise the reward.

---

### Model-agnostic Bayesian learning


Conditioned on states $s_t$ and model parameters $\vtheta$
the agents models rewards
as Gaussian with known variance $\sigma^2$
$$
    p(r_t \mid \vtheta, s_t, a_t) = {\cal N}(r_t \mid h(\vtheta, s_t, a_t),\,\sigma^2)
$$

Here,
is the parametric model of choice $h(\vtheta, s, a)$ (feed-forward NN, CNN, linear model).

<v-click>

The agents internal _beliefs_ are given by the prior density
$$
    p(\vtheta) = {\cal N}(\vtheta \mid \vmu_0, \vSigma_0)
$$

</v-click>

<v-click>

Assuming that rewards are conditionally independent given $\vtheta$,
the agent's _beliefs_ are given by the posterior density

$$
    p(\vtheta \mid \data_{1:t}) \propto
    \overbrace{p(\vtheta)}^\text{prior}
    \;
    \underbrace{
        \prod_{\tau=1}^t p(r_\tau \mid \vtheta, s_\tau, a_\tau)
    }_\text{likelihood}
$$

</v-click>

---

## Model-agnostic (and recursive) Bayesian agents

Equivalently, we consider the recursive formulation of Bayes' rule
$$
    p(\vtheta \mid \data_{1:t}) \propto p(\vtheta \mid \data_{1:t-1})\,p(r_t \mid \vtheta, s_t, a_t)
$$

For example

<v-click>

$$
\begin{array}{c c c c c c c}
p(\vtheta) &
\rightarrow &
p(\vtheta \mid \data_1) &
\rightarrow &
\ldots & 
\rightarrow &
p(\vtheta \mid \data_{1:t}) \\
& & \uparrow & & & &  \uparrow \\
& & \data_1 =(s_1, a_1, r_1) & & \ldots & & \data_t =(s_t, a_t, r_t)
\end{array}
$$

</v-click>

---

## Example: Neural network classification training example

* Continual learning.
* Single-pass observations.
* No train/test-split, no mini-batches, no _multiple epochs_,

![](./public/moons-c-static.gif)


---

## Thompson sampling: Bayesian exploration and exploitation

Consider a dataset $\data_{1:t}$, with
$\data_t = (s_t, a_t, r_t)$
and posterior density characterised by $p(\vtheta \mid \data_{1:t})$,
the policy defined by TS is
$$
    \pi_t(a \mid s)
    = \int {\bf 1}\Big(a = \arg\max_{\hat a} h(\vtheta, s_t, \hat{a})\Big)\,p(\vtheta \mid \data_{1:t})\,{\rm d}\vtheta.
$$



---

### Thompson sampling algorithm

For $t \in \mathbb{N}$, having $s_{t+1} \in {\cal S}$ and $p(\vtheta \mid \data_{1:t})$,
<div class="boxed-content">

1. $\hat{\vtheta} \sim p(\vtheta \mid \data_{1:t})$ // sample from posterior
1. $a_{t+1} = \argmax_{a \in {\cal A}} h(\hat{\vtheta}, s_{t+1}, a)$ // Choose  action
1. $(s_{t+2},\,r_{t+1}) \sim p_{\rm env}(\cdot \mid s_{t+1}, a_{t+1})$
1. $\data_{t+1} \gets (s_{t+1}, a_{t+1}, r_{t+1})$
1. $p(\vtheta \mid \data_{1:t+1}) \propto p(\vtheta \mid \data_{1:t})\,p(\data_{t+1} \mid \vtheta)$ // Update beliefs

</div>


---

## Bayes and regime changes

Bayesian updates and sequential decision making (through TS) fails under model misspecification and regime changes.

<v-click>

Below, we show a neural networks' _lack of plasticity_ trained using a sequential Bayesian procedure.

![](./public/changes-moons-c-static.gif)

</v-click>

---

## Bayes and regime changes (cont'd)

In finance, adapting to _regime changes_ is commonly done through lookback windows – $\ell$.

We can think of a lookback window as the _time since the last changepoint_.


<v-click>

If we know that a changepoint occurred $\ell$ steps ago,
we modify our posterior (conditioned on $\ell$) as
$$
    p(\vtheta \mid \ell, \data_{1:t}) = p(\vtheta \mid \data_{t - \ell + 1: t}).
$$
At time $t$, $\ell \in \{0, \ldots, t\}$. If $\ell = t$, the conditional posterior is
a pre-defined prior, i.e., $p(\vtheta \mid \ell, \data_{1:t}) = p(\vtheta)$.

</v-click>

<v-click>

However,
* A long lookback risks biasing results within a regime (too much data).
* A short lookback risks high variance (not enough data).

</v-click>


---

## Bayesian changepoints

In practice, we do not know _when_ (or if) a changepoint ocurred.

<v-click>

To go around this, we follow a Bayesian approach:
let  $\ell_t$ be random (latent) variable denoting the time since the last changepoint
(or size of lookback).

</v-click>

<v-click>

$$
\begin{aligned}
    p(\vtheta \mid \data_{1:t})
    &= \sum_{\ell_t=0}^t p(\vtheta, \ell_t \mid \data_{1:t})\\
    &= \sum_{\ell_t=0}^t p(\ell_t \mid \data_{1:t})\,p(\vtheta \mid \ell_t,  \data_{1:t})\\
    &= \sum_{\ell_t=0}^t
    \underbrace{
    p(\ell_t \mid \data_{1:t})
    }_\text{evidence lookback}
    \;\;
    \underbrace{
    p(\vtheta \mid \data_{t-\ell_t+1:t})
    }_\text{conditional posterior}
\end{aligned}
$$

</v-click>



---
zoom: 0.9
---

## Finding a posterior density for the lookback

Suppose
$$
    p(\ell_t \mid \ell_{t-1}) = \lambda\,\mathbf{1}_{\ell_t=0} + (1 - \lambda)\,\mathbf{1}_{\ell_t = \ell_{t-1}+1}
$$

with $\lambda \in (0,1)$ the hazard rate. 

<v-click>

The posterior for the lookback takes the recursive form
$$
\begin{aligned}
    p(\ell_t \mid \data_{1:t})
    &= \sum_{\ell_{t-1}=0}^{t-1} p(\ell_t, \ell_{t-1} \mid \data_{1:t})\\
    &= p(r_t \mid s_t, a_t, \data_{t - \ell_t:t})\,\sum_{\ell_{t-1}=0}^{t-1} p(\ell_{t-1} \mid \data_{1:t-1})\,p(\ell_t \mid \ell_{t-1}).
\end{aligned}
$$
with posterior predictive density
$$
    p(r_t \mid s_t, a_t, \data_{t - \ell_t:t})
    = \int p(r_t \mid \vtheta, s_t, a_t)\,p(\vtheta \mid \data_{t - \ell_t:t}) \d\vtheta.
$$

</v-click>

---
zoom: 0.9
---

### Our ALTS algorithm: Adaptive-lookback Thompson Sampling

For $t \in \mathbb{N}$, having $s_{t+1} \in {\cal S}$ and $\{p(\vtheta \mid \ell_t, \data_{1:t})\}_{\ell_t=0}^t$,
<div class="boxed-content">

1. For $\ell_t = 0, \ldots, t$:
    * $\hat{\vtheta}^{(\ell_t)} \sim p(\vtheta \mid \ell_t,\,\data_{1:t})$
1. $\hat{\vtheta} \gets \sum_{\ell_t=0}^t \hat{\vtheta}^{(\ell_t)} p(\ell_t \mid \data_{1:t})$
1. $a_{t+1} \gets \argmax_{a \in {\cal A}} h(\hat{\vtheta}, s_{t+1}, a)$ // Choose action
1. $(s_{t+2},\,r_{t+1}) \sim p_{\rm env}(\cdot \mid a_{t+1})$
1. $\data_{t+1} \gets (s_{t+1}, a_{t+1}, r_{t+1})$
1. For $\ell_{t+1} = 0, \ldots, t+1$:
    * $\ell_t = \ell_{t+1} - 1$
    * $p(\vtheta \mid \ell_{t+1},\,\data_{1:t+1}) \propto p(\vtheta \mid \ell_{t}, \data_{1:t})\,p(\data_{t+1} \mid \vtheta)$ 
    * $p(r_{t+1} \mid s_{t+1},\,\ell_{t+1},\,\data_{1:t}) = \int p(r_{t+1} \mid \vtheta, s_{t+1}, a_{t+1})\,p(\vtheta \mid \data_{t - \ell_t:t}) \d\vtheta$
    * $p(\ell_{t+1} \mid \data_{1:t+1}) = p(y_{t+1} \mid s_{t+1}, a_{t+1}, \data_{t - \ell_{t+1}:t})\,\sum_{\ell_{t}=0}^t p(\ell_{t} \mid \data_{1:t})\,p(\ell_{t+1} \mid \ell_{t})$
</div>

For $\ell_{t} = -1$, the conditional posterior is the pre-defined prior, i.e., $p(\vtheta \mid \ell_{t}, \data_{1:t}) = p(\vtheta)$.


---
zoom: 0.95
---

## Application: Optimal market making under regime changes

<v-click>

- **Asset price dynamics with regime-dependent drift:**

  $$
  dS_t = \boldsymbol{m}_t^\top \boldsymbol{\mu} \,dt + \sigma\, dW_t,
  $$

  where $\boldsymbol{m}_t$ is a Markov chain with transition matrix $\boldsymbol{\Delta}$, and drift vector $\boldsymbol{\mu}$.
- Direction of price is important: ratio buyers / sellers depends on regime.

</v-click>

<v-click>

- **Buy/sell order arrivals at depths** $\delta^b, \delta^a$ **follow**

  $$
  \Lambda^b(\delta^b) = c \, e^{-\kappa \delta^b}, 
  \quad
  \Lambda^a(\delta^a) = c \, \boldsymbol{m}_t^\top \boldsymbol{\gamma} \, e^{-\kappa \delta^a},
  $$

  where $\boldsymbol{\gamma}$ is the imbalance vector.

</v-click>

<v-click>

- **Oracle objective:** maximize expected terminal wealth penalized by inventory risk:

  $$
  u^{\boldsymbol{\Delta}}(t,x,q,e^i) =
  \mathbb{E}_t\!\left[x_T + q_T(S_T - \alpha q_T) - \phi \int_t^T q_s^2 \, ds \right],
  $$

  where $q_t$ is the inventory and $x_t$ is the cash.

</v-click>

---

### Rewards for model-agnostic setting

- We assume market market has negligible volume / impact.
- Each arm $i$: fixed quoting strategy $a_i = \{\delta^{i,b}, \delta^{i,a}\}$  
- At each time $t$, the _Bayesian_ agent observes the reward

$$
r_t =
\underbrace{
\delta^{b}\Delta N_t^{b} + \delta^{a}\Delta N_t^{a}
}_\text{spread revenue}
-
\underbrace{
    \phi\,q_{t^-}\operatorname{sgn}(\Delta q_t)(1 - I_t)
}_\text{asymmetric quote}
-
\underbrace{
    \phi\,|q_{t^-}|I_t
}_\text{risk aversion}
$$

where  
* $\Delta N_t^{b}, \Delta N_t^{a}$ = numbers of buy/sell LOs filled between $t\!-\!1$ and $t$,  
* $\Delta q_t = \Delta N_t^{b} - \Delta N_t^{a}$ is the inventory change,
* $I_t = \mathbf{1}_{\{\,\Delta N_t^{b}=0,\; dN_t^{a}=0\,\}}$ is a no-trade indicator.

<!-- - First two terms → profits from executed LOs  
- third term → reversal term. Encourages asymmetric quotes.
- Last terms → penalize inventory accumulation  
- Encourages quotes that **reduce inventory** while **capturing spread** -->

<v-click>

Although $r_t$ arises from the above structure, ALTS treats it as Gaussian:

$$
p(r \mid a, s, \vtheta) 
= \mathcal{N}\!\big(r \mid \vtheta^\top \phi(a),\, \rho^2\big),
$$

with $\phi(a)$ a random Fourier feature mapping and $\rho > 0$ fixed.

</v-click>

---
layout: center
---

# Experiments

---

## Results: Static TS

![](./cadaptive_single_path_static.png)

---

## Results: ALTS

![](./cadaptive_single_path.png)


---

## Results: well-specified and misspecified setups

![](./pnl_plots_ALTS_vs_optimal_comparison.png)

---

## Why not RL?

* Easy in principle, unstable in practice (ongoing work).

![](./cum_rewards_rl.png)
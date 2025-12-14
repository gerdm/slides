---
layout: cover
background: https://github.com/user-attachments/assets/777e9a19-7ddf-429e-819d-6ae09df78f89
title: BONE
mdc: true
author: Gerardo Duran-Martin
date: 2025-09-04
info: Generalised Bayesian online learning in non-stationary environments
---

# Generalised (B)ayesian (O)nline learning in (N)on-stationary (E)nvironments

**Gerardo Duran-Martin**, Leandro Sánchez-Betancourt, Alexander Shestopaloff, and Kevin Murphy

---

## Motivating example: hourly electricity load
Seven features (lagged by one hour): pressure (kPa), cloud cover (\%), humidity (\%), temperature \(C\) , wind direction (deg), and wind speed (KmH).

One target variable: hour-ahead electricity load (kW).

![day-ahead electricity forecasting](./figures/day-ahead-dataset.png){style="max-width:60%" class="horizontal-center"}


---

## How do we create agents that continually learn and adapt to their environment?

<v-click>
Two requirements:
 
 * The ability to learn continually from streaming datasets.
 * The ability to adapt when the data-generating process changes.
</v-click>

<v-click>

We present a Bayesian perspective to tackle these two challenges:
* We update beliefs with new evidence using Bayes' rule.
* We adapt to regime changes as defined by hierarchical state-space models.

</v-click>

<v-click>

**We find that various methods in the literature can be seen as instances of our framework.**
* Contextual bandits.
* Test-time adaptation.
* Reinforcement learning.
* (Online) continual learning --- catastrophic forgetting / plasticity-stability tradeoff.
* Dataset shift --- covariance shift, prior probability shift, domain shift, etc.

</v-click>



---
layout: two-cols-header
---

# Online (incremental / streaming / sequential) learning
Choices of **(M.1)** and **(A.1)**

::left::
* Targets $\vy_t \in \reals^o$, features $\vx_t\in\reals^M$.
* Datapoint $\data_t = (\vx_t, \vy_t)$.
* Dataset $\data_{1:t} = (\data_1, \ldots, \data_t)$
* **Goal**: Estimate $y_{t+1}$ given $x_{t+1}$ and $\data_{1:t}$

::right::

![dataset incremental](./figures/incremental-learning-full.png){style="max-width:80%" class="horizontal-center"}

---
layout: center
---

# (M.1) and (A.1): Bayesian online learning
Choices of measurement function and posterior computation

---

## Choice of measurement function (M.1)
Let $p(y \mid \vtheta,\,\vx)$ be a probabilistic _observation_ model with
$\mathbb{E}[\vy \cond \vtheta, \vx] = h(\vtheta, \vx)$ the _conditional measurement function_.

* $h(\vtheta, \vx) = \vtheta^\intercal\,\vx$ (linear regression)
* $h(\vtheta, \vx) = \sigma(\vtheta^\intercal\,\vx)$ (logistic regression)
* $h(\vtheta, \vx) = \vtheta_2^\intercal\,{\rm Relu}(\vtheta_1^\intercal\,\vx + b_1) + b_2$ (single-hidden layer neural-network)
* $h(\vtheta, \vx) = ...$ (Your favourite neural network architecture)

---

## Bayes for online learning: posterior estimation

Consider a prior $p(\vtheta)$,
a measurement model $p(y \mid \vtheta,\,\vx)$,
and
a dataset $\data_{1:t}$.

<v-click>

**Static Bayes.**  
The _batch_ posterior over model parameters at time $t$ is
$$
    p(\vtheta \mid \data_{1:t}) \propto p(\vtheta)\,\prod_{\tau=1}^t p(\vy_\tau \mid \vtheta, \vx_\tau)
$$

</v-click>

<v-click>

**Recursive Bayes.**  
Having the _prior_ $p(\vtheta \mid \data_{1:t-1})$, the _recursive_ posterior over model parameters
at time $t$ is

$$
    p(\vtheta \mid \data_{1:t}) \propto p(\vtheta \mid \data_{1:t-1})\,p(\vy_t \mid \vtheta, \vx_t).
$$

</v-click>

<v-click>

**Generalised recursive Bayes.**
$$
    p(\vtheta \cond \data_{1:t}) \propto
        p(\vtheta \cond \data_{1:t-1})
    \,
        \exp(-\ell(\vtheta; \vy_t, \vx_t)).
$$

In _classical_ Bayes, $\ell(\vtheta; \vy_t, \vx_t) = -\log p(\vy_t \cond \vtheta, \vx_t)$

</v-click>


---

## Recursive posterior estimation and prediction (A.1)
Generalised (recursive) Bayesian online learning amounts to finding a sequence of posterior densities
and making one-step-ahead predictions.

$$
\begin{aligned}
    p(\vtheta) &\to& p(\vtheta \cond \data_1) &\to& p(\vtheta \cond \data_{1:2}) &\to& \ldots &\to& p(\vtheta \cond \data_{1:t})\\
    \downarrow & & \downarrow & & \downarrow & & & &  \downarrow \\
    \hat{\vy}_{1} & & \hat{\vy}_{2} & & \hat{\vy}_{3} & & & & \hat{\vy}_{t+1} \\
    \uparrow & & \uparrow & & \uparrow & & & &  \uparrow \\
    \vx_1 & & \vx_2 & & \vx_2 & & & &  \vx_{t+1} \\
\end{aligned}
$$

---
zoom: 0.9
---

## One-step-ahead Bayesian prediction

Let $p(y \mid \vtheta,\,\vx)$ be a probabilistic _observation_ model with
$\mathbb{E}[\vy \cond \vtheta, \vx] = h(\vtheta, \vx)$ the _conditional measurement function_.
Given $\vx_{t+1}$ and $\data_{1:t}$,
one can do the following predictions.


<v-click> 

**Bayesian posterior predictive mean**
$$
    \hat{\vy}_{t+1}
    = \mathbb{E}[h(\vtheta_t, \vx_{t+1}) \cond \data_{1:t}]
    = \int
    \underbrace{h(\vtheta_t, \vx_{t+1})}_\text{measurement fn.}
    \overbrace{p(\vtheta_t \cond \data_{1:t})}^\text{posterior density}
    \,{\rm d}\vtheta_t.
$$

</v-click>

<v-click> 

**MAP approximation**
$$
    \hat{\vy}_{t+1}
    = \mathbb{E}[h(\vtheta_t, \vx_{t+1}) \mid \data_{1:t}]
    \approx h(\vtheta_*, \vx_{t+1}),
$$

with $\vtheta_* = \argmax_{\vtheta}\,p(\vtheta \mid \data_{1:t})$

</v-click>

<v-click> 

**Posterior sampling**

$$
    \hat{\vy}_{t+1} = h(\hat{\vtheta}, \vx_{t+1}),
$$

with $\hat{\vtheta} \sim p(\vtheta \mid \data_{1:t})$.

</v-click>

---

##  Choice of algorithm for posterior approximation (A.1)

How do we find the recursive estimates for $p(\vtheta \mid \data_{1:t})$?

<v-click>

### Example: (Recursive) variational Bayes methods
Suppose $q_0(\vtheta) = {\cal N}(\vtheta \mid \vmu_0, \vSigma_0)$, then
$$
    q_{t}(\vtheta) \triangleq {\cal N}(\vtheta \mid \vmu_t, \vSigma_t),
$$

with 
$$
    \vmu_t, \vSigma_t = \argmin_{\vmu, \vSigma}{\bf D}_\text{KL}
    \Big(
        {\cal N}(\vtheta \cond \vmu, \vSigma) \,\|\,
        q_{t-1}(\vtheta)\,\exp(-\ell(\vtheta; \vy_t, \vx_t))
    \Big).
$$


A convenient choice: density is fully specified by the first two moments only.

</v-click>

----

# A running example: online classification with neural networks
* Online Bayesian learning using **(M.1)** a two hidden-layer neural network and **(A.1)** moment-matched LG.
* Data is seen only once.
* We evaluate the exponentially-weighted moving average of the accuracy in the one-step-ahead forecast.

![linreg](./figures/sequential-moons.gif) {class="horizontal-center"}

---

### The choice of measurement model **(M.1)**: two hidden-layer neural network

Take
$$
    h(\vtheta, \vx) = \sigma(\phi_\vtheta(\vx)),
$$

with $\phi_\theta: \reals^M \to \reals$ a two-layered neural network with real-valued output unit

Take the model $\hat{p}(y_t \cond \theta, x_t) = {\rm Bern}(y_t \cond h(\theta, x_t))$.

```python
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(1)
        return jax.nn.sigmoid(x)
```


---

## The choice of posterior **(A.1)**: moment-matched LG

Bayes rule under linearisation and Gaussian assumption (second-order SGD-type update).


$$
\begin{aligned}
\vH_t &= \nabla_\theta h(\vmu_{t-1}, \vx_t) & \text{(Jacobian)}\\
\hat{y}_t & = h(\vmu_{t-1}, \vx_t) & \text{ (one-step-ahead prediction)} \\
\vR_t &= \hat{y}_t\,(1 - \hat{y}_t) & \text{ (moment-matched variance)}\\
\vS_t &= \vH_t\,\vSigma_{t-1}\,\vH_t^\intercal + \vR_t\\
{\bf K}_t &= \vSigma_{t-1}\vH_t\,\vS_t^{-1} & \text{(gain matrix)}\\
\hline
\vmu_t &\gets \vmu_{t-1} + {\bf K}_t\,(y_t - \hat{y}_t) & \text{(update mean)}\\
\vSigma_t &\gets \vSigma_{t-1} - \vK_t\,\vS_t\vK_t^\intercal & \text{(update covariance)}\\
p(\vtheta_t \cond \data_{1:t}) &\gets {\cal N}(\vtheta_t \cond \vmu_t, \vSigma_t) &\text{(posterior density)}
\end{aligned}
$$



---

## Online classification with neural networks
* Single pass of data
* Evaluate the exponentially-weighted moving average of the one-step-ahead accuracy

![sequential classification with static dgp](./figures/moons-c-static.gif)


---

# Changes in the data-generating process


* Simply applying Bayes rule on a parametrised model fails under model misspecification (typical in ML problems) and lack of model capacity.

* In this case, conditioning on more data does not lead to better performance.


<v-click>

**Example: Non-stationary moons dataset: an online continual learning example**  
DGP changes every 200 steps. Agent is not aware of changepoints.

**Goal**: Estimate the class $\vy_{t+1}$ given data $\data_{1:t}$ and $\vx_{t+1}$.

![non-stationary-moons-split](./figures/mooons-dataset-split.png)

</v-click>

---

## Static Bayesian updates --- non-stationary moons
* Online Bayesian learning using (M.1) a single hidden-layer neural network and (A.1) moment-matched LG.
* Keep applying Bayes rule as new data streams in.
* We observe so-called *lack of plasticity*.

![sequential classification with varying dgp](./figures/changes-moons-c-static.gif)


---
layout: two-cols-header
---

## Tackling non-stationarity in a Bayesian way: we cannot update what we don't model.

::left::

Fix **(M.1)** and **(A.1)**.

Non-stationarity is modelled through
* **(M.2)** an auxiliary variable,
* **(M.3)** the effect of the auxiliary variable on the _prior_, and
* **(A.2)** a _posterior_ over choices of auxiliary variables.

Considering choices of **(M.2)** and **(M.3)** recovers various ways in which non-stationarity has been tackled.

::right::

![overview of BONE methods](./figures/bone-methods-overview.png){style="max-width:80%"}


---
layout: center
---

# (M.2) and (M.3): Choice of auxiliary variable $\psi_t$ and conditional prior $\pi$
What we mean by a _regime_ and how prior beliefs change subject to the regime.



---

## Modifying prior beliefs based on regime (M.3)

* (M.2) a choice of _auxiliary variable_ $\psi_t$ to track regimes, and
* (M.3) a _conditional prior_ $\pi$ (not necessarily posterior at time $t$) and

<v-click>

$$
    \underbrace{q(\vtheta_t; \psi_t, \data_{1:t})}_{\text{posterior (A.1)}}
     \propto
    \underbrace{\pi(\vtheta_t \cond \psi_t, \data_{1:t-1})}_\text{past information (M.3)}\,
    \overbrace{
        \exp\left(-\ell(\vy_t;\,\vtheta_t, \vx_t)\right)
    }^\text{current information (M.1)}
$$


The idea of the conditional prior $\pi$ as a "forgetting operator"
is formalised in [Kullhavy and Zarrop, 1993](https://www.tandfonline.com/doi/abs/10.1080/00207179308923034?casa_token=JRF8WBcxJE4AAAAA:vYbOsSi9K5xySA34i9pVPFiEwanUNMyv2WLkzl5odJeePdOpECS55PgVgZA0Z6GUd0O-SKmZkRU).

</v-click>

<v-click>

**Choice of conditional prior (M.3) --- the Gaussian case**
$$
    \pi(\vtheta_t \cond \psi_t,\, \data_{1:t-1}) =
    {\cal N}\big(\vtheta_t \cond g_{t}(\psi_t, \data_{1:t-1}), G_{t}(\psi_t, \data_{1:t-1})\big),
$$


* $g_t(\cdot, \data_{1:t-1}): \Psi_t \to \reals^m$ --- mean vector of model parameters.
* $G_t(\cdot, \data_{1:t-1}): \Psi_t \to \reals^{m\times m}$ --- covariance matrix of model parameters.

</v-click>


---

## Runlength with prior reset (RL-PR)
* $\psi_t \in \{0, 1, 2, \ldots, t\}$.
* Number of timesteps since the last changepoint (adaptive lookback window).
* Corresponds to the Bayesian online changepoint detection (BOCD) algorithm ([Adams and MacKay, 2007](https://arxiv.org/abs/0710.3742)).

![Runlength auxiliary variable](./figures/auxvar-rl.png) {style="max-width:70%" class="horizontal-center"}

<v-click>

$$
    \begin{aligned}
        g_t(\psi_t, \data_{1:t-1}) &= \mu_0\,\mathbb{1}(\psi_t  = 0) + \mu_{(\psi_{t-1})}\mathbb{1}(\psi_t > 0),\\
        G_t(\psi_t, \data_{1:t-1}) &= \Sigma_0\,\mathbb{1}(\psi_t  = 0) + \Sigma_{(\psi_{t-1})}\mathbb{1}(\psi_t > 0),\\
    \end{aligned}
$$

where  $\mu_{(\psi_{t-1})}, \Sigma_{(\psi_{t-1})}$ denotes the posterior belief using observations
from indices $t - \psi_t$ to $t - 1$.
$\mu_0$ and $\Sigma_0$ are pre-defined prior mean and covariance.

</v-click>


---

## Changepoint probability (CPP)
* Changepoint probabilities.
* $\psi_t \in (0, 1]$.
* Mean reversion to the prior as a function of the probability of a changepoint.

![Changepoint probability auxiliary variable](./figures/auxvar-cpp.png) {style="max-width:70%" class="horizontal-center"}


<v-click>

$$
    \begin{aligned}
        g(\upsilon_t, \data_{1:t-1}) &= \upsilon_t \mu_{t-1} + (1 - \upsilon_t)  \mu_0 \,,\\
        G(\upsilon_t, \data_{1:t-1}) &=  \upsilon_t^2 \Sigma_{t-1} + (1 - \upsilon_t^2)  \Sigma_0\,.
    \end{aligned}
$$

</v-click>



---
layout: center
---

# (A.2) Weighting function for regimes
How do we weight over choices of $\psi_t$ given data $\data_{1:t}$?

---

## (A.2) The recursive Bayesian choice

Assume a Markovian dependence between auxiliary variables, i.e.,
$$
    p(\psi_t \cond \psi_{1:t-1}, \data_{1:t-1}) = p(\psi_t \cond \psi_{t-1}, \data_{1:t-1}).
$$

Then, we can write the posterior over auxiliary variables as

$$
\begin{aligned}
    \nu_t(\psi_t)
    &= p(\psi_t \cond \data_{1:t})\\
    &= 
    p(y_t \cond x_t, \psi_t, \data_{1:t-1})\,
    \sum_{\psi_{t-1} \in \Psi_{t-1}}
    p(\psi_{t-1} \cond \data_{1:t-1})\,
    p(\psi_t \cond \psi_{t-1}, \data_{1:t-1}).
\end{aligned}
$$

<v-clicks>

* This assumption yields recursive update methods under discrete $\Psi_t$ (e.g., runlengths).
* Alternative choices are possible (e.g., loss-based approaches).

</v-clicks>


---

# BONE --- Bayesian online learning in non-stationary environments
* (M.1) A model for observations (conditioned on features $x_t$) --- $h(\theta, x_t)$.
* (M.2) An auxiliary variable for regime changes --- $\psi_t$.
* (M.3) A model for prior beliefs (conditioned on $\psi_t$ and data $\data_{1:t-1}$) ---
.
* (A.1) An algorithm to weight over choices of $\theta$ (conditioned on data $\data_{1:t}$) ---

$q(\theta;\,\psi_t, \data_{1:t}) \propto \pi(\theta \cond \psi_t, \data_{1:t-1}) p(y_t \cond \theta, x_t)$.

* (A.2) An algorithm to weight over choices of $\psi_t$ (conditioned on data $\data_{1:t}$).

---
layout: two-cols-header
---

# BONE (generalised) posterior predictive

::left::

![BONE SSM](./figures/BONE-SSM.png){style="max-width:80%"}


::right::

$$
\begin{aligned}
    \hat{\vy}_t
    &:= \sum_{\psi_t \in \Psi_t}
    \underbrace{\nu(\psi_t \cond \data_{1:t})}_{\text{(A.2: weight)}}\,
    \int
    \underbrace{h(\theta_t, \vx_{t+1})}_{\text{(M.1: model)}}\,
    \underbrace{q(\theta_t;\, \psi_t, \data_{1:t})}_{\text{(A.1: posterior)}}
    d\theta_t,
\end{aligned}
$$

with

$$
   q(\theta_t;\,\psi_t, \data_{1:t})
    \propto \underbrace{\pi(\theta_t;\, \psi_t, \data_{1:t-1})}_\text{(M.3: prior)}\,
    \underbrace{\exp(-\ell(\vy_t; \theta_t, \vx_t))}_\text{(M.1: loss)}
$$


---

# Back to the non-stationary moons example
Suppose measurement model **(M.1)** is a two hidden layer neural network
with linearised moment-matched Gaussian **(A.1)**.

Consider three combinations of **(M.2)**, **(M.3)**, and **(A.2)**:
1. Runlenght with prior reset and a single hypothesis --- `RL[1]-PR`.
2. Changepoint probability with OU dynamics --- `CPP-OU`.

---

## RL-PR
* When changepoint detected: reset back to initial weights
* Auxiliary variable **(M.1)** proposed in [Adams and Mackay, 2007](https://arxiv.org/abs/0710.3742) --- BOCD.

$$
    \begin{aligned}
        g_t(r_t, \data_{1:t-1}) &= \mu_0\,\mathbb{1}(r_t  = 0) + \mu_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
        G_t(r_t, \data_{1:t-1}) &= \Sigma_0\,\mathbb{1}(r_t  = 0) + \Sigma_{(r_{t-1})}\mathbb{1}(r_t > 0).\\
    \end{aligned}
$$

![rl-pr-sequential-classification](./figures/changes-moons-rl-pr.gif)

---

## CPP-OU
* Revert to prior proportional to the probability of a changepoint.
* Auxiliary variable **(M.1)** proposed in [Titsias et. al., 2023](https://arxiv.org/abs/2306.08448) to OCL in classification.

$$
    \begin{aligned}
        g(\upsilon_t, \data_{1:t-1}) &= \upsilon_t \mu_{t-1} + (1 - \upsilon_t)  \mu_0 \,,\\
        G(\upsilon_t, \data_{1:t-1}) &=  \upsilon_t^2 \Sigma_{t-1} + (1 - \upsilon_t^2)  \Sigma_0\,.
    \end{aligned}
$$

![cpp-ou-sequential-classification](./figures/changes-moons-cpp-ou.gif)

---

# Creating a new method: RL-OUPR
* Combine gradual and abrupt changes
* Single runlength (`RL[1]`) as choice auxiliary variable **(M.1)**.

<v-click>

**The choice of conditional prior (M.3)**:  
* Reset if the hypothesis of a a changepoint is below some thresold $\varepsilon$.
* OU-like reversion rate otherwise.


$$
\begin{aligned}
    g_t(r_t, \data_{1:t-1}) &=
    \begin{cases}
        \mu_0\,(1 - \nu_t(r_t)) + \mu_{(r_t)}\,\nu_t(r_t) & \nu_t(r_t) > \varepsilon,\\
        \mu_0 & \nu_t(r_t) \leq \varepsilon,
    \end{cases}\\
   G_t(r_t, \data_{1:t-1}) &=
    \begin{cases}
        \Sigma_0\,(1 - \nu_t(r_t)^2) + \Sigma_{(r_t)}\,\nu_t(r_t)^2 & \nu_t(r_t) > \varepsilon,\\
        \Sigma_0 & \nu_t(r_t) \leq \varepsilon.
    \end{cases}
\end{aligned}
$$

</v-click>


---

### RL[1]-OUPR --- choice of weighting function (A.2)
* Posterior predictive ratio test


$$
    \nu_t(r_t^{(1)}) =
    \frac{p(y_t \cond r_t^{(1)}, x_t, \data_{1:t-1})\,(1 - \pi)}
    {p(y_t \cond r_t^{(0)}, x_t, \data_{1:t-1})\,\pi + p(y_t \cond r_t^{(1)}, x_t, \data_{1:t-1})\,(1-\pi)}.
$$

Here, $r_{t}^{(1)} = r_{t-1} + 1$ and $r_{t}^{(0)} = 0$.


---

## RL[1]-OUPR
* A novel combination of **(M.2)** and **(M.3)** ---  reset or forget
    * Revert to prior proportional to probability of changepoint (forget).
    * Reset completely if prior is the most likely hypothesis (reset).

![rl-oupr-sequential-classification](./figures/changes-moons-rl-oupr.gif)

---

## Unified view of examples in the literature

![BONE-methods-examples](./figures/methods-bone.png)




---

## Electricity forecasting results (2018-2020)
![day ahead forecasting results](./figures/day-ahead-results.png)


---

# Conclusion
We introduce a framework for Bayesian online learning in non-stationary environments (BONE).
BONE methods are written as instances of:

**Three modelling choices**
* (M.1) Measurement model
* (M.2) Auxiliary variable
* (M.3) Conditional prior

**Two algorithmic choices**
* (A.1) weighting over model parameters (posterior computation)
* (A.2) weighting over choices of auxiliary function


---
layout: end
---

[gerdm.github.io/posts/bone-slides](https://gerdm.github.io/posts/bone-slides)  
[arxiv.org/abs/2411.10153](https://arxiv.org/abs/2411.10153)
---
layout: cover
background: https://github.com/user-attachments/assets/777e9a19-7ddf-429e-819d-6ae09df78f89
class: text-center
title: BONE
mdc: true
author: Gerardo Duran-Martin
date: 2025-09-04
info: Generalised Bayesian online learning in non-stationary environments
---

# Generalised (B)ayesian (O)nline learning in (N)on-stationary (E)nvironments

**Gerardo Duran-Martin**, Leandro Sánchez-Betancourt, Alexander Shestopaloff, and Kevin Murphy

---

## How do we create agents that can continually learn and adapt to their environment?

<v-click>
Two requirements:

 * The ability to learn continually from streaming datasets.
 * The ability to adapt when the data-generating process changes.
</v-click>

<v-click>

We present a Bayesian perspective to tackle these two challenges:
* We update beliefs with new evidence using Bayes' rule
* We adapt to regime changes as defined by hierarchical state-space models


</v-click>

----

# Motivation
Learning and adaptation (in non-stationary environments) has been studied under different names and different setups in AI/ML:

* Contextual bandits.
* Test-time adaptation.
* Reinforcement learning.
* (Online) continual learning --- catastrophic forgetting / plasticity-stability tradeoff.
* Dataset shift --- covariance shift, prior probability shift, domain shift, etc.


<v-click>

Various _Bayesian_ methods have been proposed to tackle some of the problems above.

</v-click>

<v-click>
 
However,
* there hasn't been a clear way to distinguish these Bayesian methods and
* there hasn't been enough recognition of methods outside the ML literature that also tackle
non-stationarity (and that can be applied to ML problems).


</v-click>


---

## A unifying framework

**We seek to unify methods that perform Bayesian online learning in non-stationary environments.**

Allows us to

<v-clicks>

1. Categorise the various methods that tackle inference under _non-stationarity_ under a common (algorithmic) language
     *  machine learning --- contextual bandits / continual learning / reinforcement learning
     * statistics --- segmentation / switching state-space models
     * engineering --- system identification / filtering
1. Plug-and-play implementation in Jax: [github.com/gerdm/BONE](https://github.com/gerdm/BONE).
1. Extend application of existing methods (use method developed for field A and apply to field B).
1. Develop new methods.

</v-clicks>

---
layout: two-cols-header
---

# Methods under the BONE framework are motivated by the following state-space model

::left::

* **(M.1)** measurement model model $h(\vtheta, \vx)$,
* **(A.1)** algorithm to update model parameters $\vtheta$ (posterior over parameters),
* **(M.2)** auxiliary variable to track _regimes_ $\psi_t$,
* **(M.3)** conditional prior that modify beliefs based on auxiliary variable $\pi(\vtheta\cond\psi, \vx)$, and
* **(A.2)** algorithm to weigh choices of auxiliary variables (posterior over regimes).

::right::

![BONE SSM](./figures/BONE-SSM.png){style="max-width:80%"}


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

##  Bayes for online learning: Choice of (M.1) and (A.1)


How do we find the recursive estimates for $p(\vtheta \mid \data_{1:t})$?

* Closed-form solution intractable except in a few special cases, e.g., linear Gaussian with Gaussian priors or conjugate priors.
* In most cases, we resort to an approximation. We denote the approximation to the posterior by the algorithmic choice **(A.1)**.



---

## Example: (Recursive) variational Bayes methods
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


---

## Example: Moment-matched linear Gaussian (LG)
* Suppose $p(\vtheta \cond \data_{1:t-1}) = {\cal N}(\vtheta \cond \vmu_{t-1}, \vSigma_{t-1})$.
* Linearise measurement function **(M.1)** around the previous mean $\vmu_{t-1}$
and model as linear Gaussian.

Consider the likelihood

$$
\begin{aligned}
    h(\vtheta, x) &\approx \bar{h}_t(\vtheta, x) = H_t\,(\vtheta_t - \vmu_{t-1}) +  h(\vmu_{t-1}, x_t),\\
    p(\vy_t \cond \vtheta, \vx_t) &\approx {\cal N}(\vy_t \cond  \bar{h}_t(\vtheta, x), R_t),
\end{aligned}
$$

with $R_t$ is the moment-matched variance of the observation process.

Because the likelihood is linear-Gaussian and the prior is Gaussian, we obtain
$$
    p(\vtheta \cond \data_{1:t}) = {\cal N}(\vtheta \cond \vmu_t, \vSigma_t).
$$


The one-step-ahead forecast is

$$
\begin{aligned}
    \hat{\vy}_{t+1} &= \mathbb{E}[\bar{h}(\vtheta_t, \vx_{t+1}) \cond \data_{1:t}] = h(\vmu_{t}, \vx_{t+1}).
\end{aligned}
$$



---


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

* Simply applying Bayes rule on a parametrised model fails under model misspecification (pretty much ML problem)
and lack of model capacity.

* In this case, conditioning on more data does not lead to better performance.

---

## Non-stationary moons dataset: an online continual learning example
DGP changes every 200 steps. Agent is not aware of changepoints.

**Goal**: Estimate the class $\vy_{t+1}$ given data $\data_{1:t}$ and $\vx_{t+1}$.

![non-stationary-moons-split](./figures/mooons-dataset-split.png)

---

## The full dataset (without knowledge of the task boundaries)
![non-stationary-moons-full](./figures/moons-dataset-full.png){style="max-width:70%" class="horizontal-center"}

---

## Static Bayesian updates --- non-stationary moons
* Online Bayesian learning using (M.1) a single hidden-layer neural network and (A.1) moment-matched LG.
* Keep applying Bayes rule as new data streams in.
* We observe so-called *lack of plasticity*.

![sequential classification with varying dgp](./figures/changes-moons-c-static.gif)


---


## Is being "Bayesian" is not enough for adaptivity?

Some machine learning works have implied that being "Bayesian" endows adaptivity
and that is the _lack_ of a good approximation to the posterior that limits the ability to adapt.

> Continual learning [...] is a form of online learning in which data continuously arrive in a possibly non i.i.d way. [...]
> There already exists an extremely general framework for continual learning: Bayesian inference.
> --- Nguyen et al., “Variational Continual Learning” (2017)

In this work, we argue that the approximation to the posterior **(A.1)** and choice of measurement model **(M.1)**
can (and should) be considered separately from the adaptation strategy.

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

# (M.2): Choice of auxiliary variable $\psi_t$
What we mean by a _regime_.


---

## Runlenght (RL)
* Number of timesteps since the last changepoint (adaptive lookback window).
* $\psi_t = r_t \geq 0$.

![Runlength auxiliary variable](./figures/auxvar-rl.png)

---

## Runlenght with changepoint count (RLCC)
* Number of timesteps since the last changepoint (lookback window) and count of the number of changepoints.
* $\psi_t = (r_t, c_t)$.

![Runlength and changepoint count auxiliary variable](./figures/auxvar-rlcc.png)

---

## Changepoint location (CPL)
* Binary sequence of changepoint locations.
* Encodes number of timesteps since last changepoint, count of changepoints, and location of changepoints.

![Changepoint location auxiliary variable](./figures/auxvar-cpl.png)

---

## Changepoint location (CPL) alt.
* Binary sequence of values belonging to the current regime.
* Allows for non-consecutive data conditioning.

![Changepoint location auxiliary variable](./figures/auxvar-cpl2.png)

---

## Changepoint probability (CPP)
* Changepoint probabilities.
* $\psi_t = \nu_t \in [0,1]$.

![Changepoint probability auxiliary variable](./figures/auxvar-cpp.png)

---

## Mixture of experts (ME)
* Choices of over a fixed number of models.

![Mixture of experts auxiliary variable](./figures/auxvar-me.png)

---

## Constant (C)
* Single choice of model.
* $\psi_t = c$.

![Constant auxiliary variable](./figures/auxvar-cst.png)

---

## Summary of auxiliary variables

![table of auxiliary variables](./figures/auxvar-table.png)


---
layout: center
---

# (M.3): Choice of conditional prior $\pi$
How do prior beliefs change, subject the value of $\psi_t$


---

# Modifying prior beliefs based on regime (M.3)
* Construct the posterior based on two modelling choices:
    * (M.3) a _conditional prior_ $\pi$ (not necessarily posterior at time $t$) and
    * (M.1) a choice of loss function $\ell$.


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

---

## Choice of conditional prior (M.3) --- the Gaussian case


$$
    \pi(\vtheta_t \cond \psi_t,\, \data_{1:t-1}) =
    {\cal N}\big(\vtheta_t \cond g_{t}(\psi_t, \data_{1:t-1}), G_{t}(\psi_t, \data_{1:t-1})\big),
$$


* $g_t(\cdot, \data_{1:t-1}): \Psi_t \to \reals^m$ --- mean vector of model parameters.
* $G_t(\cdot, \data_{1:t-1}): \Psi_t \to \reals^{m\times m}$ --- covariance matrix of model parameters.

---

## `C-Static` --- constant update with static auxvar
* $\psi_t = c$.
* Classical (static) Bayesian update:

$$
\begin{aligned}
    g_t(c, \data_{1:t-1}) &= \mu_{t-1}\\
    G_t(c, \data_{1:t-1}) &= \Sigma_{t-1}\\
\end{aligned}
$$


---

## `RL-PR` --- runlength with prior reset
* $\psi_t = r_t$.
* Corresponds to the Bayesian online changepoint detection (BOCD) algorithm ([Adams and MacKay, 2007](https://arxiv.org/abs/0710.3742)).


$$
    \begin{aligned}
        g_t(r_t, \data_{1:t-1}) &= \mu_0\,\mathbb{1}(r_t  = 0) + \mu_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
        G_t(r_t, \data_{1:t-1}) &= \Sigma_0\,\mathbb{1}(r_t  = 0) + \Sigma_{(r_{t-1})}\mathbb{1}(r_t > 0),\\
    \end{aligned}
$$


where  $\mu_{(r_{t-1})}, \Sigma_{(r_{t-1})}$ denotes the posterior belief using observations
from indices $t - r_t$ to $t - 1$.
$\mu_0$ and $\Sigma_0$ are pre-defined prior mean and covariance.

---

## `CPP-OU` --- changepoint probability with Ornstein-Uhlenbeck process
* $\psi_t = \upsilon_t \in (0, 1]$.
* Mean reversion to the prior as a function of the probability of a changepoint:

$$
    \begin{aligned}
        g(\upsilon_t, \data_{1:t-1}) &= \upsilon_t \mu_{t-1} + (1 - \upsilon_t)  \mu_0 \,,\\
        G(\upsilon_t, \data_{1:t-1}) &=  \upsilon_t^2 \Sigma_{t-1} + (1 - \upsilon_t^2)  \Sigma_0\,.
    \end{aligned}
$$


---


## `CPL-sub` --- changepoint location with subset of data
* $\psi_t = s_{1:t} \in \{0,1\}^t$.


$$
\begin{aligned}
    g_t(s_{1:t}, \data_{1:t-1}) &= \mu_{(s_{1:t})},\\
    G_t(s_{1:t}, \data_{1:t-1}) &= \Sigma_{(s_{1:t})},\\
\end{aligned}
$$

where  $\mu_{(s_{1:t})}, \Sigma_{(s_{1:t})}$ denote the posterior beliefs using observations that are 1-valued.

---

## `C-ACI` --- constant with additive covariance inflation

* $\psi_t = c$.
* Random-walk assumption. _Inject_ noise at every new timestep. Special case of a linear state-space-model (SSM).

$$
\begin{aligned}
    g_t(c, \data_{1:t-1}) &= \mu_{t-1},\\
    G_t(c, \data_{1:t-1}) &= \Sigma_{t-1} + Q_t.\\
\end{aligned}
$$

Here, $Q_t$ is a positive semi-definitive matrix. Typically, $Q_t = \alpha {\bf I}$ with $\alpha > 0$.


---
layout: center
---

## A prediction conditioned on $\psi_t$ and $\data_{1:t}$


$$
    \hat{y}_{t+1}^{(\psi_t)}
    = \mathbb{E}_{q_t}[h(\theta_t;\, x_{t+1}) \cond \psi_t]
    := \int h(\theta_t;\, x_{t+1})\,q(\theta_t;\,\psi_t, \data_{1:t})d \theta_t\,.
$$



---
layout: center
---

# (A.2) Weighting function for regimes
An algorithmic choice to weight over elements $\psi_t \in \Psi_t$.

---

## (A.2) The recursive Bayesian choice

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


For some $\psi_t$ and $p(\psi_t \cond \psi_{t-1})$, this method yields recursive update methods.

---

## (A.2) A loss-based approach
Suppose
, take

$$
    \nu_t(\alpha_t) = 1 - \frac{\ell(y_{t+1}, \hat{y}_{t+1}^{(\alpha_t)})}
    {\sum_{k=1}^K \ell(y_{t+1}, \hat{y}_{t+1}^{(k)})},
$$

with $\ell$ a loss function (lower is better).

---

## (A.2) An empirical-Bayes (point-estimate approach)
For $\psi_t = \upsilon_t \in [0,1]$,


$$
    \upsilon_t^* = \argmax_{\upsilon \in [0,1]} p(y_t \cond x_t, \upsilon, \data_{1:t-1}).
$$

Then,

$$
    \nu(\upsilon_t) = \delta(\upsilon_t = \upsilon_t^*).
$$


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

# BONE (generalised) posterior predictive


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
3. Constant auxiliary variable with additive covariance inflation --- `C-ACI`.

---

## RL[1]-PR
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

## C-ACI
* Constantly forget past information.
* Used in [Chang et. al., 2023](https://arxiv.org/abs/2305.19535) for scalable non-stationary online learning.

$$
\begin{aligned}
    g_t(c, \data_{1:t-1}) &= \mu_{t-1},\\
    G_t(c, \data_{1:t-1}) &= \Sigma_{t-1} + Q_t.\\
\end{aligned}
$$

![c-aci-sequential-classification](./figures/changes-moons-c-aci.gif)


---

# Creating a new method: RL-OUPR
* Combine gradual and abrupt changes
* Single runlength (`RL[1]`) as choice auxiliary variable **(M.1)**.

---

## RL[1]-OUPR --- choice of (M.2) and (M.3)

* Reset if the hypothesis of a a changepoint is below some thresold $\varepsilon$.
* OU-like reversion rate otherwise.


$$
    g_t(r_t, \data_{1:t-1}) =
    \begin{cases}
        \mu_0\,(1 - \nu_t(r_t)) + \mu_{(r_t)}\,\nu_t(r_t) & \nu_t(r_t) > \varepsilon,\\
        \mu_0 & \nu_t(r_t) \leq \varepsilon,
    \end{cases}
$$



$$
   G_t(r_t, \data_{1:t-1}) =
    \begin{cases}
        \Sigma_0\,(1 - \nu_t(r_t)^2) + \Sigma_{(r_t)}\,\nu_t(r_t)^2 & \nu_t(r_t) > \varepsilon,\\
        \Sigma_0 & \nu_t(r_t) \leq \varepsilon.
    \end{cases}
$$


---

## RL[1]-OUPR --- choice of (A.2)
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

## Sequential classification --- comparison
In-sample hyperparameter optimisation.

![comparison-sequential-classification](./figures/changes-mooons-comparison.png)

---

## Unified view of examples in the literature

![BONE-methods-examples](./figures/methods-bone.png)


---

# Experiment: hourly electricity load
Seven features:
* pressure (kPa), cloud cover (\%), humidity (\%), temperature (C) , wind direction (deg), and wind speed (KmH).

One target variable:
* hour-ahead electricity load (kW).

Features are lagged by one-hour.

----

## Experiment: hourly electricity load

![day-ahead electricity forecasting](./figures/day-ahead-dataset.png){style="max-width:80%" class="horizontal-center"}

---

## Electricity forecasting during _normal_ times
![day ahead forecasting normal](./figures/day-ahead-forecast-normal.png){style="max-width:80%" class="horizontal-center"}


---

## Electricity forecasting before and after Covid _shock_
![day ahead forecasting shock](./figures/day-ahead-forecast.png){style="max-width:80%" class="horizontal-center"}

---

## Electricity forecasting
![day ahead forecasting shock](./figures/day-ahead-forecast-rlpr.png){style="max-width:80%" class="horizontal-center"}

---

## Electricity forecasting results (2018-2020)
![day ahead forecasting results](./figures/day-ahead-results.png)


---


## Experiment --- heavy-tailed linear regression

![heavy-tailed-linear regression panel](./figures/segments-tdist-lr.png){class="horizontal-center"}

---

## Heavy tailed linear regression (sample run)

![heavy-tailed-linear-regression](./figures/outliers-all-panel.gif){style="max-width:80%" class="horizontal-center"}

---

# Even more experiments!

* Bandits.
* Online continual learning.
* Segmentation and prediction with dependence between changepoints.

See paper for details.


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
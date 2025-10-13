---
layout: cover
mdc: true
background: /small-discrete-dynamical-system.png
author: Gerardo Duran
date: 2025-09-10
title: Bayesian filtering for online continual learning
info: Kalman filtering for Bayesian continual learning
transition: none
theme: academic
coverAuthor: Gerardo Duran-Martin - OMI, University of Oxford
coverAuthorUrl: https://grdm.io
coverBackgroundUrl: /presentation.png
class: text-white
---

# From spaceships to neural networks

Bayesian filtering as the language of online continual learning


---
layout: center
zoom: 1.5
---

## It takes a village...

```
M Altamirano      M Jones               K Murphy
F-X Briol         A Kara                L Sanchez-Betancourt
Á Cartea          J Knoblauch           AY Shestopaloff
P Chang
```


---

# What this talk is about

<v-click>

A major challenge of AI/ML is the creation of learning agents or learning mechanisms
that learn and adapt over time.

</v-click>

<v-click>

Current learning mechanisms
  * Do not update _beliefs_ on the face of new information.
  * Do not handle / model non-stationarity in the data.
  * Are sensitive to outliers (corrupted data).


This is a known problem: dataset drift, online continual learning, plasticity-stability trade-off, ...

</v-click>


<v-click>

We can leverage the vast literature on **Bayesian filtering** to understand and develop methods for online / incremental / streaming / continual learning learning.

</v-click>

<v-click>


Bayes is not (only) for uncertainty;  Bayes for belief (information) propagation.

</v-click>




---
layout: two-cols-header
---

# Part I: spaceships and filters — the birth of the Kalman filter (1960s)
Tracking and forecasting the position of a spaceship.

::left::

![](./figures/animation.gif){style="max-width:100%;"}

::right::


The linear-state-space model
$$
\begin{aligned}
  \vtheta_t &= \vF_t\,\vtheta_{t-1} + \vu_t,\\
  \vy_t &= \vH_t\,\vtheta_t + \ve_t.
\end{aligned}
$$

* $\vtheta_t$ are **unknown** (latent) positions.
* $\vy_t$ are **known** (observed) measurements.
* $\var[\vu_t] = \vQ_t$, $\var[\ve_t] = \vR_t$.
* $\vH_t$ is the projection matrix.
* $\vF_t$ is the transition matrix.


----

## The mathematical problem

Given the observations up to time $t$, what's the best (in an L2 sense)
estimate of the _true_ position of the spaceship?

$$
\tag{1}
    \boldsymbol{\mu}_t  = \arg\min_{\boldsymbol\mu}\mathbb{E}[\|\vtheta_t - \boldsymbol \mu||_2^2 \mid \vy_{1:t}]
$$

* $\vtheta_t$: true position of the spaceship.
* $\vmu_t$: our best estimate of the position of the spaceship.
* $\vy_{1:t} \equiv \{\vy_1, \ldots, \vy_t\}$: observations / measurements.

<v-click>

A big problem back in the 1960's was that people did not know how to solve this quadratic equation efficiently. <sup>1</sup>

</v-click>

<!-- They needed
* Know the current position of the spaceship (state-estimation),
* Know where the spaceship will be in the future (forecasting) -->

<Footnotes separator>
  <Footnote :number=1>
    Battin, Richard H. "Space guidance evolution-a personal narrative." Journal of Guidance, Control, and Dynamics 5.2 (1982): 97-110.
    </Footnote>
</Footnotes>

---


## The Kalman filter
* Rudolf E. Kalman proposed a way to solve this problem recursively <sup>1</sup>:
Start with some initial estimate of the position of your spaceship and error estimate  $(\vmu_0, \vSigma_0)$
and iterate.

**predict step**  
$$
\begin{aligned}
  \vmu_{t|t-1} &= \vF_{t}\,\vmu_{t-1}\\
  \vSigma_{t|t-1} &= \vF_t\,\vSigma_{t-1}\,\vF_t + \vQ_t\\
\end{aligned}
$$

**Update step**
$$
\begin{aligned}
  \vS_t &= \vH_t\,\vSigma_{t|t-1}\,\vH_t^\intercal + \vR_t\\
  \vK_t &= \vSigma_{t|t-1}\,\vH_t\,\vS_t^{-1}\\
  \vmu_t &= \vmu_{t-1} + \vK_t\,(\vy_t - \vH_t\,\vmu_{t|t-1})\\
  \vSigma_t &= \vSigma_{t|t-1} - \vK_t\,\vS_t\,\vK_t
\end{aligned}
$$


<Footnotes separator>
  <Footnote :number=1>
    Kalman, Rudolph Emil. "A new approach to linear filtering and prediction problems." (1960): 35-45.
  </Footnote>
</Footnotes>

---
layout: two-cols-header
---

## Kalman filter example
Tracking a 2d-object from 1d-observations

::left::

![](./figures/animation-filtering.gif){style="max-width:100%;"}

::right::
```python
def predict(bel):
    mu, Sigma = bel
    
    mu = F @ mu
    Sigma = F @ Sigma @ F.T + Q
    bel = mu, Sigma
    
    return bel
```

```python
def update(bel, obs):
    mu_pred, Sigma_pred = bel

    St = H @ Sigma_pred @ H.T + R
    Kt = jnp.linalg.solve(St, H @ Sigma_pred).T
    mu = mu_pred + Kt @ (obs - H @ mu_pred)
    Sigma = Sigma_pred - Kt @ St @ Kt.T

    bel = (mu, Sigma)
    return bel
```




---

# Bayesian filters
A few year after Kalman's seminal paper,
researchers observed that the KF equations could be seen as finding the posterior density
of a multivariate Gaussian.<sup>1</sup>

$$
    {\color{red} \vmu_t}
    = \arg\min_{\vmu}\mathbb{E}[\|\vtheta_t - \vmu\|_2^2 \mid \vy_{1:t}]
    \iff
    p(\vtheta_t \mid \vy_{1:t}) = {\cal N}(\vtheta_t \mid {\color{red} \vmu_t},\,\vSigma_t),
    
$$


with $p(\vtheta_t \mid \vy_{1:t})$ found using Bayes' rule:
$$
  \underbrace{
  p(\vtheta_t \mid \vy_{1:t})
  }_\text{posterior}
  \propto
  \underbrace{p(\vy_t \mid \vtheta_t)}_\text{likelihood}\,
  \underbrace{p(\vtheta_t \mid \vy_{1:t-1})}_\text{prior}
$$


<v-click>

A big motivation for a Bayesian approach was to _unify_ the various methods that have been presented to that point
(not uncertainty quantification).


> [Filtering] appears to look more like a bag of tricks than a unified subject - V. Peterka (1979)<sup>2</sup>

</v-click>

<!-- > It is the author's thesis that this approach offers a unifying methodology, at least conceptually, to the general problems of estimation and control. - https://ieeexplore.ieee.org/document/1105763 -->

<Footnotes separator>
  <Footnote :number=1>
    Ho, Yu-Chi, and R. C. K. A. Lee. "A Bayesian approach to problems in stochastic estimation and control." IEEE transactions on automatic control 9.4 (1964): 333-339.
    </Footnote>
  <Footnote :number=2>
    Peterka, V. "Bayesian system identification." IFAC Proceedings Volumes 12.8 (1979): 99-114.
  </Footnote>
</Footnotes>


---

## The explosion in applications

Shortly after the KF paper, people started to use (and still use) this method in different fields:
electrical engineering, acoustic engineering, quantitative finance, weather forecasting.

<v-clicks>

Main points of interest in filtering-type problems are

Robustness to outliers. <sup>1</sup>

Adaptivity to misspecified (changing) environments. <sup>2</sup>

Scalability to high-dimensional latent spaces. <sup>3</sup>

</v-clicks>

<Footnotes separator>
  <Footnote :number=1>
    West, Mike. "Robust sequential approximate Bayesian estimation." Journal of the Royal Statistical Society Series B: Statistical Methodology 43.2 (1981): 157-166.
    </Footnote>
  <Footnote :number=2>
    Mehra, Raman. "Approaches to adaptive filtering." IEEE Transactions on automatic control 17.5 (2003): 693-698.
    </Footnote>
  <Footnote :number=3>
  Evensen, Geir. "Sequential data assimilation with a nonlinear quasi‐geostrophic model using Monte Carlo methods to forecast error statistics." JGRO 99.C5 (1994).
  </Footnote>
</Footnotes>




---
zoom: 0.9
---


# Part II: Filtering? The dawn of AI/ML methods in the engineering sciences

<v-clicks>

* _Classical_ filtering methods are as good as their hardcoded inductive bias.
* More and more research shows that _with plenty of data and big models_, weak inductive biases are better.<sup>1</sup>

* Alternative Bayesian ML (as a way to perform model averaging / uncertainty quantification) is being disfavoured for point estimates. <sup>2</sup>

</v-clicks>

<Footnotes separator>
  <Footnote :number=1>
    Lam, Remi, et al. "Learning skillful medium-range global weather forecasting." Science 382.6677 (2023): 1416-1421.
    </Footnote>
  <Footnote :number=2>
    See e.g., Section 2.6.3 in 
    Bishop, Christopher M., and Hugh Bishop. Deep learning: Foundations and concepts. Springer Nature, 2023.
  </Footnote>
</Footnotes>


---

## An ML approach to forecasting

The recipe:

<v-click> 

1. Take your favourite neural network $h_\vtheta$ parametrized by parameters $\vtheta$ --- CNN, LSTM, Transformer, GNN, ...

</v-click>


<v-click> 

2. Define the target variable $\vy_{t} = h_\vtheta(\vy_{t-1},\,\vy_{t-2},\,\ldots,\,\vy_{t-k})$

</v-click> 

<v-click> 

3. Optimize using mini-batch GD (e.g., adamw):
$$
    \vtheta_* = \arg\min_\vtheta {\cal L}(\vtheta,\,{\cal D}_{1:N})
$$
with $\data_{t} = \{\vy_t, \vy_{t-1}, \vy_{t-2}, \ldots, \vy_{t-k}\}$, big $N$, and some loss function ${\cal L}$.

</v-click> 


<v-click>

4. Submit to your favourite conference!

</v-click> 

---

## Why Bayesian filtering?

<v-clicks>

**Q:** If scale is all we need (Sutton’s Bitter Lesson<sup>1</sup>), why bother with filtering or Bayes?

**A**: We still need to _learn_ $\vtheta_* = \arg\min_\vtheta {\cal L}(\vtheta,\,{\cal D}_{1:N})$

For some problems, the _best_ estimate $\vtheta_*$ changes over time: we call this **continual learning problems**.

<!-- True _learning_ is about adapting over time --- not just about bigger models and more data. -->

</v-clicks>

<v-click>

Fields in AI/ML facing continual learning challenges:

1. Fully-online reinforcement learning.
1. Contextual bandits / recommender systems.
1. Financial applications (Market making).
1. High-dimensional signal processing / state-estimation / system identification.
1. Test-time adaptation
1. Dataset shift — covariance shift, prior probability shift, domain shift, etc

</v-click>


<Footnotes separator>
  <Footnote :number=1>
    http://www.incompleteideas.net/IncIdeas/BitterLesson.html
  </Footnote>
</Footnotes>

---

## A (toy) continual learning example

Consider a two hidden-layer neural network that is trained sequentially using the following time-varying dataset
![moons dataset split](./figures/mooons-dataset-split.png) {style="max-width:70%;" class="horizontal-center"}

<v-click>

**Sequential updating and catastrophic forgetting / lack of plasticity**  
![C-static-non-adaptive](./figures/changes-moons-c-static.gif) {style="max-width:70%;" class="horizontal-center"}

</v-click>


---

# Part III: Filtering!

For time-dependent AI/ML problems, we seek to create continual learning methods that are

<v-clicks>

1. **Robust** to outliers and heavy-tailed noise,
1. **Adaptive** to non-stationarity and drift,
1. **Scalable** to large number of model parameters, and
1. **Unified**, i.e., expressible in a common language.

</v-clicks>

<v-click> 

**As we've seen Bayesian filtering methods have tried to tackle this challenge**

</v-click>

---

## Learning is Filtering
Let $h_\vtheta$ be a neural network parametrized by $\vtheta$,
$\data_t = (\vx_t, \vy_t)$ a datapoint, and
$\data_{1:t}$ a dataset.

Parameters are states: $\vtheta_t$ evolves over time. 

<v-click>

Learning $\equiv$ Bayesian filtering over $\vtheta_t$:

$$
\begin{aligned}
    p(\vtheta_{t} \mid \data_{1:t})
    &\propto
    p(\data_t \mid \vtheta_t)\,
    \underbrace{p(\vtheta_t \mid \data_{1:t-1})}_\text{prior predictive}\\
    % &=
    % \underbrace{
    % p(\data_t \mid \vtheta_t)
    % }_\text{new obs.}
    % \,\int p(\vtheta_t\mid \vtheta_{t-1})\,
    % \underbrace{
    % p(\vtheta_{t-1} \mid \data_{1:t-1})
    % }_\text{past belief}
    % \d\vtheta_{t-1}
\end{aligned}
$$

</v-click>


<v-click>

Examples:
* Regression: Gaussian likelihood,
* Classification: Bernoulli likelihood,
* Generative modelling.

</v-click>

---

### Example: Ridge regression as a filtering problem



Given
$\data_{1:t}$ with
$\data_t = (\vx_t, \vy_t)$,
$\vx_t \in \reals^M$,
$\vy_t \in \reals^D$,
and
$\alpha, \beta, \lambda > 0$.



$$
  {\color{red} \vmu_T}
  = \argmin_{\vtheta} \|\vy_t - \vx_t^\intercal\vtheta\|_2^2 + \lambda\,\|\vtheta\|_2^2
$$

<v-click>

$$
  \iff
$$

$$
  p(\vtheta \mid \data_{1:t})
  \propto p(\vtheta)\, p(\data_{1:t} \mid \vtheta)
  % &= {\cal N}(\vtheta \mid \vzero, \alpha\,\vI)\,\prod_{t=1}^T {\cal N}(\vy_t \mid \vtheta^\intercal\,\vx_t, \beta)\\
  = {\cal N}(\vtheta \mid {\color{red}\vmu_T}, \vSigma_T)
$$

</v-click>


<v-click>

$$
  \implies
$$

$$
  {\color{red} \vmu_T} = (\vX^\intercal\,\vX + \lambda\vI)^{-1}\,\vX^\intercal\,\vY,
$$
with
$\vX \in \reals^{T\times M}$,
$\vY \in \reals^{T}$.


</v-click>


<v-click>

**Takeaway:** Similar to the KF problem, solving the Ridge regression loss function
is equivalent to finding the posterior over model parameters (under the assumption of Gaussianity).


</v-click>

---

### Example: Ridge regression as a filtering problem (cont'd)

$$
  p(\vtheta \mid \data_{1:T})
  \propto p(\vtheta)\, p(\data_{1:T} \mid \vtheta)
  % &= {\cal N}(\vtheta \mid \vzero, \alpha\,\vI)\,\prod_{t=1}^T {\cal N}(\vy_t \mid \vtheta^\intercal\,\vx_t, \beta)\\
  = {\cal N}(\vtheta \mid {\color{red}\vmu_T}, \vSigma_T)
$$


<v-click>

$$
  \iff
$$

$$
  p(\vtheta \mid \data_{1:T})
  \propto p(\vtheta \mid \data_{1:T-1})\,p(\data_T \mid \vtheta)
  % &= {\cal N}(\vtheta \mid \vzero, \alpha\,\vI)\,\prod_{t=1}^T {\cal N}(\vy_t \mid \vtheta^\intercal\,\vx_t, \beta)\\
  = {\cal N}(\vtheta \mid {\color{red}\vmu_T}, \vSigma_T)
$$

</v-click>


<v-click>

$$
  \iff
$$

$$
\begin{array}{c c c c c c c}
p(\vtheta) &
\xrightarrow{\rm update} &
p(\vtheta \mid \data_1) &
\xrightarrow{\rm update} &
\ldots &
\xrightarrow{\rm update} &
p(\vtheta \mid \data_{1:T}) \\
& & \uparrow & & \uparrow & & \uparrow \\
& & \data_1 & & \ldots & & \data_T \\
\end{array}
$$

</v-click>


<v-click>

$$
  \iff
$$

$$
\text{Apply KF updates!}
$$


</v-click>

---

### Example: Ridge regression as a filtering problem (cont'd)

$$
\begin{aligned}
  \vS_t &= \vx_t^\intercal\,\vSigma_{t-1}\,\vx_t + \beta\\
  \vK_t &= \vSigma_{t-1}\,\vx_t^\intercal\,\vS_t^{-1}\\
  \vmu_t &= \vmu_{t-1} + \vK_t\,(\vy_t - \vx_t^\intercal\,\vmu_{t-1})\\
  \vSigma_t &= \vSigma_{t-1} - \vK_t\,\vS_t\,\vK_t
\end{aligned}
$$

* (Left) Sequential parameter estimation.
(Right) RMSE error on held-out test-set.

* Value of parameters and RMSE on test set  match at the final timestep $T$.


![](./figures/reclinreg-params.png){style="max-width:50%; float: left;"}
![](./figures/reclinreg-error.png){style="max-width:50%; float: right;"}

---

### Example: Bayesian filtering using a neural network?


Take
$$
    h_\vtheta(\vx) = \sigma(\phi_\vtheta(\vx)),
$$

with $\phi_\vtheta: \reals^M \to \reals$ a two-layered neural network with real-valued output unit

Take the model $\hat{p}(y_t \mid \theta, x_t) = {\rm Bern}(y_t \mid h_\vtheta(\vx_t))$.

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

### Example: Bayesian filtering using a neural network? (cont'd)
**Use the Extended Kalman filter!** <sup>1</sup>

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

<Footnotes separator>
  <Footnote :number=1>
  Singhal, Sharad, and Lance Wu. "Training multilayer perceptrons with the extended Kalman algorithm." Neurips (1988).
  </Footnote>
</Footnotes>

---

### Example: Bayesian filtering using a neural network? (cont'd)
 
![sequential classification with static dgp](./figures/moons-c-static.gif)




---
hideInToc: true
---

## Examples on filtering as learning

**On robustness**  
* Likelihood are misspecified and are sensitive to outliers.

**On adaptivity**  
* How do we create learning methods that adapt to changes in the data-generating process?

**On scalability**  
* How do we create scalable learning methods?


---

## Robustness<sup>1</sup>
Use of generalised-Bayes ideas to create robust online learner.


![online-forecast-outliers-nonrobust](./figures/online-learning-outliers-nonrobust.gif){style="max-width:45%; float:left"}

<v-click>

![online-forecast-outliers-robust](./figures/online-learning-outliers-robust.gif){style="max-width:45%; float:right;"}

</v-click>

<Footnotes separator>
  <Footnote :number=1>
  Duran-Martin, Gerardo, et al. "Outlier-robust kalman filtering through generalised bayes." ICML (2024).
  </Footnote>
</Footnotes>



---

## Adaptivity<sup>1</sup>

Make use of Bayesian hierarchical state-space models to embed inductive biases of non-stationarity (in parameter space).

![](./figures/changes-moons-rl-oupr.gif)


<Footnotes separator>
  <Footnote :number=1>
  Duran-Martin, Gerardo, et al. "A unifying framework for generalised Bayesian online learning in non-stationary environments." TMLR (2024).
  </Footnote>
</Footnotes>


--- 

## Scalability

For moderately-sized neural network, a _full-rank_ KF update step is infeasible because of $\vSigma_t \in \reals^{D\times D}$
whenever $\vtheta_t \in \reals^D$:

$$
\begin{aligned}
  \vSigma_t &= \vSigma_{t-1} - \vK_t\,\vS_t\,\vK_t
\end{aligned}
$$

Let $\vW_t \in \reals^{D\times d}$ and $D \ggg d$.

We can make various approximations to $\vSigma_t$:

<v-clicks>

1. Diagonal + low-rank $\vSigma_t = \vW_t\,\vW_t^\intercal + {\rm diag}(\sigma_1, \ldots, \sigma_D)$, <sup>1</sup>
2. Low-rank only $\vSigma_t = \vW_t\,\vW_t^\intercal$, <sup>2</sup>
3. Diagonal $\vSigma_t = {\rm diag}(\sigma_1^2, \ldots, \sigma_D^2)$ <sup>3</sup> (in some cases, recovers adamw).

</v-clicks>


<v-click>

There is a tight relationship between _low-rank_ Bayesian filters and current SOTA optimisers.
Still unresolved.

</v-click>

<Footnotes separator>
  <Footnote :number=1>
  Chang, Peter G., et al. "Low-rank extended Kalman filtering for online learning of neural networks from streaming data." CoLLAs (2023).
  </Footnote>
  <Footnote :number=2>
  Duran-Martin, Gerardo, et al. "Scalable Generalized Bayesian Online Neural Network Training for Sequential Decision Making." arXiv preprint (2025).
  </Footnote>
  <Footnote :number=3>
    Aitchison, Laurence. "Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods." Neurips (2020).
  </Footnote>
</Footnotes>


----

## A new look at Bayesian filtering

We can leverage existing filtering tools (and derive new filtering-like tools) to tackle these problems!

| Filtering | latent $\vtheta_t$ | example | inductive bias for predictive model |
| --- | --- | --- | --- |
| classical | Latent (explainable) process | Position of spaceship | included in SSM |
| ML | meaningless* | neural network parameters | separate from choice of SSM



---

# Conclusion
Many learning problems in AI/ML can be reformulated into a Bayesian filtering framework.

This allows:

<v-clicks>

1. Tackling continual learning problems (non-stationarity, dataset-shift, plasticity-stability tradeoff, ...)
1. Developing robust learning methods.
1. Thinking about scaling optimization methods in a principled way.
1. Viewing many open problems in AI/ML under a single lens.

</v-clicks>

---
layout: end
---


```
END | grdm.io
```


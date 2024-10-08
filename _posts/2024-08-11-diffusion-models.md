---
title: 'Diffusion Models: Sampling via Stochastic Differential Equations'
date: 2024-08-11
permalink: /posts/2024/08/diffusion-models/
tags:
  - cool posts
  - category1
  - category2
---

Generative Diffusion Processes
==============================

We are given a sample $X_1, \dots, X_N \sim P$ of size $N \in \mathbb{N}$ drawn from an unknown data distribution $P \in \mathcal{P}(\mathbb{R}^D)$. 

Our goal is to generate a new sample $X^*$ that is indistinguishable from the original samples $X_1, \dots, X_N$. A classic example of this is image generation, where the task is to build a sampler that can generate new images from the unknown data distribution of a given image dataset.

Diffusion models (DMs) provide a powerful framework for progressively transforming samples from a simple Gaussian distribution into the often highly complex and multimodal data distribution $ P $. The mathematics behind diffusion models are remarkably beautiful and represent, in my view, one of the most fascinating applications of stochastic calculus in recent times. This blog post will focus on the conceptual ideas underlying DMs and offer some mathematical intuition for their success.



The forward process
-----------------------------------------

Let’s begin with the straightforward task. Our objective is to gradually transform the samples $ X_1, \dots, X_N $ into Gaussian noise. To achieve this, we consider a very simple Stochastic Differential Equation (SDE) described as 
<br>
<br>
$$
\begin{align}
  X(0) &= x_0 \\
  dX(t) &= - \alpha(t) X(t) \, dt + \sigma(t) \, d B(t),
\end{align}
$$
<br>
where $x_0 \in \mathbb{R}^D$ is an arbitrary initial condition, $\alpha:[0,T] \to (0,\infty)$ is the shrinkage factor, $\sigma: [0, T] \to (0, \infty)$ is the diffusion coefficient, $T >0 $ the time-horizon and $\big(B(t)\big)$ a Brownian motion. This SDE is one of the simplest to encounter.
The drift term is linear in the space variable $X(t)$, and the diffusion coefficient is independent of $X(t)$. Consequently, it represents a special case of a linear SDE, and the solution, which we denote as $X_{x_0}(t)$, is readily available in closed form. Specifically, the marginal law for fixed $t$ is given as

$$
\begin{align}
    X_{x_0}(t) \sim \mathcal{N} \big( m(t), \Sigma(t) I_D \big) 
\end{align}    
$$ 
where 
$$
\begin{align}
    m(t) & := \exp \big(-\int_0^t \alpha(\tau) d\tau \big) x_0 \in \mathbb{R}^D  \\
    \Sigma(t) &:=  \int_0^t \sigma(\tau)^2 \exp\left(-2\int_\tau^t \alpha(u) \, du \right) \, d\tau.
\end{align}    
$$

We can leverage this fact to generate noisy versions of our samples $X_n$ by using (1) with the initial condition $x_0 = X_n$, provided we have access to $m(t)$ and $\Sigma(t)$ which we can ensure by choosing $\alpha(t)$ and $\sigma(t)$ appropriately.

Importantly, we can generate such a sample for any $ t > 0 $ without relying on an Euler-Maruyama discretization of the SDE. This approach avoids the need for expensive recursive function evaluations that would be necessary for more general SDEs, where closed-form solutions are not available.

The shrinkage factor $\alpha(t)$ and the diffusion coefficient $\sigma(t)$ are part of the model architecture. Typically, we aim to shrink only a little and add small amounts of noise at the beginning and increase $\alpha(t)$ and $\sigma(t)$ as $t$ grows. We will explore the reasons for this approach later. For now, it is sufficient to know that successful schedulers are well-established in the [literature](https://arxiv.org/abs/2011.13456) and that we do not need to learn $\alpha(t)$ and $\sigma(t)$ from data.

Another crucial point to note is that we can infer the long-term behavior of the SDE from (1). Specifically, since $m(t) \to 0$ as $t \to \infty$ (exponentially fast), we find that
<br>
<br>
$$
\begin{align}
  X_{x_0}(t) &\approx \mathcal{N}(0, \Sigma(t) I_D)
\end{align}
$$
<br>
for sufficiently large $t > 0$ and arbitrary $x_0 \in \mathbb{R}^D$. The right-hand side of (6) can again be easily sampled from, as long as we have access to $\Sigma(t)$, which is typically available by the construction of $\sigma(t)$.

So far, we have considered the SDE with a fixed initial condition $x_0 \in \mathbb{R}^D$. We now want to replace this fixed initial condition with an unknown data distribution $P \in \mathcal{P}(\mathbb{R}^D)$, leading to another (but very closely related) SDE:
<br>
<br>
$$
\begin{align}
    X(0) &\sim P \\
    dX(t) &= -  \alpha(t) X(t) \, dt + \sigma(t) \, dB(t)
\end{align}
$$
<br>
We denote the solution to this SDE as $\big(X(t)\big)$, suppressing the $P$ subscript for notational convenience. Notice that it is easy to produce samples from $X(t)$ via
<br>
<br>
$$
\begin{align}
    X(t) \sim \exp \big(-\int_0^t \alpha(\tau) d\tau \big) X_0 + \sqrt{\Sigma(t)} \epsilon
\end{align}
$$
<br>
where $\epsilon \sim \mathcal{N}(0,I_D)$ and $X_0 \sim P$.  However, we do not have a closed-form expression for its Lebesgue density $p_t$. More formally, we define $P(t) := \text{Law} [ X(t) ]$ and $P(t \mid x_0) := \text{Law} [ X_{x_0}(t)]$, and note that
<br>
<br>
$$
\begin{align}
    p_t(x) = \int p_t(x \mid x_0) \, dP(x_0)
\end{align}
$$
<br>
where $p_t$ and $p_t(\cdot \mid x_0)$ denote the Lebesgue densities of $P(t)$ and $P(t \mid x_0)$, respectively. Here, $p_t(x \mid x_0) = \mathcal{N}(x \mid x_0, \Sigma(t))$ is known, but $p_t(x)$ is unknown, since we do not know $P$ and therefore cannot calculate the integral in (10).


The time reversal
-----------------
So far, we have only explored how to evolve an initial data distribution $P$ over time into a zero-mean Gaussian with a known variance $\Sigma(t)$. The more interesting case is to reverse this process: turning Gaussian noise into a sample from $P$.

To this end, consider the SDE with a random initial condition $P(T)$ given by
<br>
<br>
$$
\begin{align}
    & \widehat{X}(0) \sim P(T) \\
    & d \widehat{X}(t) =  \left[ \alpha(T-t) \widehat{X}(t) + \sigma^2(T-t) s\big(T-t, \widehat{X}(t)\big) \right] dt + \sigma(T-t) \, d \widehat{B}(t),
\end{align}
$$
<br>
where $s(t, x) := \nabla \log p_t(x)$ for $t \in [0, T]$ and $x \in \mathbb{R}^D$ is the score function of the distribution $P_t$. [Anderson (1982)](https://www.sciencedirect.com/science/article/pii/0304414982900515) show that the solution to this SDE, denoted as $(\widehat{X}(t))$, has marginals $Q(t) := \text{Law}[\widehat{X}(t)]$ with
<br>
<br>
$$
\begin{align}
    Q(t) = P(T-t)
\end{align}
$$
<br>
for all $t \in [0, T]$. In particular, the marginal distribution $Q(T)$ coincides with the data distribution $P(0) = P$.

In principle, we can produce samples from $P$ by forward simulation of the reverse SDE (a standard approach would be to use an Euler-Maruyama discretization). However, we encounter two issues:
- We don't know the random initialization $P(T)$.
- We don't have access to the score function $s_t$. As noted in the previous section, $p_t$ is intractable because we cannot marginalize over $P$ in (10). Consequently, we cannot compute the gradient of $\log p_t$ to obtain the score function.

Fortunately, the first problem is less concerning. We established earlier that $P(t) \to \mathcal{N}(0, \Sigma(t))$ exponentially fast. Thus, our first approximation will be to initialize the forward simulation with $\mathcal{N}(0, \Sigma(T))$ instead of $P(T)$. This approximation will be quite accurate as long as $T$ is sufficiently large.

The second problem is more challenging but not insurmountable. We can address it by replacing $s(t, x)$ with a parameterized model $s_{\theta}(t, x)$ and then learn to approximate the score function from the data. Therefore, we need a differentiable loss function to find a parameter vector $\theta$ such that $s_{\theta}(t, x) \approx s(t, x)$.


A differentiable loss for score matching
----------------------------------------

Let $\big(\hat{X_\theta}(t)\big)$ be the solution to the SDE (12) with $s$ replaced by $s_\theta$, for a fixed $\theta$ and random initialization $\mathcal{N}(0, \Sigma(T))$.

Our goal is to choose a $\theta$ such that the sample paths of $\big(\hat{X_\theta} (t)\big)$ closely match those of $\big(\widehat{X}(t)\big)$. Mathematically, this can be formalized by stating that the path measures $\mathbf{Q}$ and $\mathbf{Q_\theta}$, which represent the probability measures associated with $\big(\widehat{X}(t)\big)$ and $\big(\hat{X_\theta}(t)\big)$ on the space of continuous paths $C\big([0,T],\mathbb{R}^D\big)$, respectively, should be close to each other, i.e., $\mathbf{Q_\theta} \approx \mathbf{Q}$. The closeness between two probability measures can be assessed using a metric or divergence, and the training process aims to improve this measure of closeness between $\mathbf{Q}$ and $\mathbf{Q_\theta}$. 

Typically, finding a measure of closeness that results in a tractable objective is challenging. However, in our special case, an application of [Girsanov's theorem](https://arxiv.org/abs/2101.09258) allows us to derive the Kullback-Leibler (KL) divergence between the path measures:
<br>
<br>
$$
\begin{align}
    \ell(\theta) &:= 2 \cdot \text{KL}(\mathbf{Q}, \mathbf{Q_\theta)} \\
                 &= - \int_{0}^T \sigma^2(T-t) \mathbb{E} \left[ \| s\big(T-t, \widehat{X}(t)\big) - s_\theta\big(T-t, \widehat{X}(t)\big) \|^2 \right] \, dt + \text{const.} \\
                 &= \int_{0}^T \sigma^2(t) \mathbb{E} \left[ \| s\big(t, X(t)\big) - s_\theta\big(t, X(t)\big) \|^2 \right] \, dt + \text{const.}
\end{align}
$$
<br>
Several methods exist for generating unbiased estimators of $\ell$: [Standard score matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) (very expensive), [sliced score matching](https://arxiv.org/pdf/1905.07088) (cheap but high variance), and [denoising score matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) (cheap and lower variance). The latter is by far the most common and may initially seem somewhat magical. We will first introduce the key lemma.

### Lemma. 
*Let $Y$ be an arbitrary random variable. Define $\widetilde{Y}:= Y + \sqrt{\Sigma} \xi$ where $\xi \sim \mathcal{N}(0,I_D)$ and let further $\widetilde{s}$ be the score function of $\widetilde{Y}$. Then
    $$
    \begin{align}
        \mathbb{E} \left[ \| h(\widetilde{Y}) - \widetilde{s}(\widetilde{Y}) \|_2^2 \right] = \mathbb{E} \left[ \| h(Y + \sqrt{\Sigma} \xi) - \Sigma^{-1/2} \xi \|_2^2 \right]
    \end{align}
    $$
holds for arbitrary $h: \mathbb{R}^D \to \mathbb{R}^D$.*

Notice that the left-hand side of (17) requires the analytical form of the score function $\widetilde{s}$ whereas the right-hand side can be easily approximated by jointly sampling $(Y, \xi)$, as long as $h$ is known. In our case, we can apply the lemma for fixed $t \in [0,T]$ with $Y :=   \exp \big(-\int_0^t \alpha(\tau) d\tau \big) X_0$, $\Sigma := \Sigma(t)$, and $h = s(t, \cdot)$, since we know that
<br>
<br>
$$
\begin{align}
    X(t)  \sim \exp \big(-\int_0^t \alpha(\tau) d\tau \big) X_0+ \sqrt{\Sigma(t)} \xi =: Y(t) +  \sqrt{\Sigma(t)} \xi
\end{align}
$$

where $\xi \sim \mathcal{N}(0, I_D)$. This results in
<br>
<br>
$$
\begin{align}
    \ell(\theta)&= \int_0^T \sigma^2(t) \mathbb{E} \left[ \left\| s_{\theta}\Big(t, Y(t)+ \sqrt{\Sigma(t)} \xi(t) \Big) - \Sigma(t)^{-1/2} \xi(t) \right\|^2 \right] dt \\
    &\approx \sum_{i=1}^I \sigma^2(t_i) \mathbb{E} \left[ \left\| s_{\theta}\Big(t_i, Y(t_i)+ \sqrt{\Sigma(t_i)} \xi(t_i) \Big) - \Sigma(t_i)^{-1/2} \xi(t_i) \right\|^2 \right] (t_i - t_{i-1}),
\end{align}
$$
<br>
where $0 = t_0 < t_1 < \dots < t_I = T$ is a partition of $[0, T]$, and $\xi(t)$ is a white noise Gaussian process. The expected value is now replaced by a Monte Carlo (MC) estimator, resulting in the final tractable objective:
<br>
<br>
$$
\begin{align}
    \ell(\theta) = \frac{1}{N} \sum_{i=1}^I \sum_{n=1}^N \sigma^2(t_i) (t_i - t_{i-1}) \left\| s_{\theta}\Big(t_i, \exp \big(-\int_0^t \alpha(\tau) d\tau \big) X_n + \sqrt{\Sigma(t_i)} \xi_n(t_i) \Big) - \Sigma(t_i)^{-1/2} \xi_n(t_i) \right\|^2,
\end{align}
$$
<br>
where $\xi_n(t_i) \sim \mathcal{N}(0, I_D)$ are independent for $n = 1, \dots, N$ and $i = 1, \dots, I$.

In most cases, it is too expensive to compute the double sum in Equation (22), which is why we typically sample a random time point $i\sim U({1,\dots,I})$ and then a mini-batch from $(X_n \mid n=1,\dots, N)$ for each gradient update.

After successful training, we can use the obtained minimizer to generate a sample trajectory of $\big(\hat{X_\theta}(t)\big)$ (e.g. via an Euler-Maruyama discretization). If all goes well, we will have $\hat{X_\theta}(T) \approx P$ and therefore achieved our goal of generating a new sample!

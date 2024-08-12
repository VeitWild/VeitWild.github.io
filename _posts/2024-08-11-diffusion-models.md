---
title: 'Diffusion Models: Sampling via Stochastic Differential Equations'
date: 2024-08-11
permalink: /posts/2024/08/diffusion-models/
tags:
  - cool posts
  - category1
  - category2
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

Generative Diffusion Processes
==============================

Let $X_1, \dots, X_N \sim P$ be a sample of size $N \in \mathbb{N}$ drawn from an unknown data distribution $P \in \mathcal{P}(\mathbb{R}^D)$. 

Our goal is to generate a new sample $X^*$ that is indistinguishable from the original samples $X_1, \dots, X_N$. A classic example of this is image generation, where the task is to build a sampler that can generate new images from the unknown data distribution of a given image dataset.

Diffusion models (DMs) provide a powerful framework for progressively transforming samples from a simple Gaussian distribution into the often highly complex and multimodal data distribution $ P $. The mathematics behind diffusion models are remarkably beautiful and represent, in my view, one of the most fascinating applications of stochastic calculus in recent times. This blog post will focus on the conceptual ideas underlying DMs and offer some mathematical intuition for their success.



The forward process
-----------------------------------------

Let us start with the easy part. We want to turn the samples $X_1,\dots,X_N$ gradually into Gaussian noise. In order to accomplish this we turn our attention to a a very simple Stochastic Differential Equation (SDE) given as 
$$
\begin{align}
  X(0) &= x_0 \\
  dX(t) &= - X(t) \, dt + \sigma(t) \, d B(t),
\end{align}
$$
where $x_0 \in \mathbb{R}^D$ is an arbitrary initial condition, $\sigma: [0, T] \to (0, \infty)$ is the diffusion coefficient, $T >0 $ the time-horizon and $\big(B(t)\big)$ a Brownian motion. This SDE is amongst the easiest one will ever encounter. The drift term is linear in the space-variable $X(t)$ and the diffusion coefficient does not depend on $X(t)$. It is therefore a special case of a linear SDE and the solution, which we denote as $X_{x_0}(t)$, is readily available in closed form 
$$
\begin{align}
    X_{x_0}(t) \sim \mathcal{N} \big( m(t), \Sigma(t) I_D \big) 
\end{align}    
$$ 
where 
$$
\begin{align}
    m(t) & := \exp(-t) x_0 \in \mathbb{R}^D  \\
    \Sigma(t) &:=  \int_0^t \exp(-2 (t-\tau) )\sigma^2(\tau) \, d \tau \, \in \mathbb{R}.
\end{align}
$$
We can use this fact and generate noisy versions of our samples $X_n$ via (1) by setting $x_0 = X_n$ as long as we have acces to $\Sigma(t)$ (which we usually do by chosing $\sigma(t)$ appropriately). Notice, in particular that we can generate such a sample for any $t>0$ without having to use an Euler-Maruyama discretization of the SDE. This avoids expensive recursive function evaluations that for more general SDEs (where we do not have closed form solutions) would be necessary. 

The diffusion coefficient $\sigma(t)$ determines the noise schedule and is part the model architecture. Typically, we want to add small amounts of noise in the beginnning and increase $\sigma(t)$ for larger $t$. We will see later why this is the case. For now it's sufficient to know that succesfull schedulers are well-established in the literatur [1] and we dont need to learn $\sigma(t)$ from data.

Another crucial point to notice is that we can infer the long term behaviour of the SDE from (1). Specifically since $m(t) \to 0$ as $t \to \infty$ (exponentially fast), we find that 
$$
\begin{align}
  X_{x_0}(t) \approx \mathcal{N} (0 , \Sigma(t) I_D )
\end{align}
$$
for large enough $t >0$ and arbitrary $x_0 \in \mathbb{R}^D$. The RHS of (6) can again be easily sampled from as long as we have access to $\Sigma(t)$, which is usually the case by construction of $\sigma(t)$.

So far we have looked at the SDE for a fixed initial condition $x_0 \in \mathbb{R}$. We now want to replace the fixed initial condition with the unknown data distribution $P \in \mathcal{P}(\mathbb{R}^D)$ which leads to another (but very closely related) SDE:
$$
\begin{align}
     X(0) &\sim P \\
    dX(t) &= - X(t) \, dt + \sigma(t) \, d B(t)
\end{align}
$$
We denote the solution to this SDE as $( X(t) )$ (suppressing the $P$ subscript for notational convenience). Notice that it is easy to produce samples from $X(t)$ via
$$
\begin{align}
    X(t) \sim \exp(-t) X_0 +  \sqrt{\Sigma(t)} \epsilon
\end{align}
$$
with $\epsilon \sim N(0,1)$ and $X_0 \sim P$. However, we don't have a closed form expression for it's Lesbegue density $p_t$. More formally, we define $P(t):= \text{Law} [ X(t) ]$ and $P(t|x_0) := \text{Law} [ X_{x_0}(t) ] $
and note that
$$ 
\begin{align}
    p_t(x) = \int p_t(x|x_0) \, dP(x_0)
\end{align}
$$
where $p_t$ and $p_t(\cdot|x_0)$ denote the Lebesgue densities of $P(t)$ and $P(t|x_0)$. Here $p_t(x|x_0)=\mathcal{N}( x | x_0, \Sigma(t))$ is known but $p_t(x)$ is unknown, since we do not know $P$ and therfore can not calculate the integral in (10).


The time reversal
-----------------

So far we have only learned how we can morph an inital data distribution $P$ over time into a zero-mean Gaussian with known variance $\Sigma(t)$. The more interesting case is to reverse this process. We want to turn Gaussian noise into a sample from $P$.

To this end, consider the SDE with random initial condition $P(T)$ given as
$$
\begin{align}
    & \widehat{X}(0) \sim P(T) \\
    &d \widehat{X}(t) =  \big[ \widehat{X}(t) + \sigma^2(T-t) s\big( T-t, \widehat{X}(t) \big)  \big] dt + \sigma(T-t) \, d \widehat{B}(t), 
\end{align}
$$
where $s(t,x) :=  \nabla \log p_t(x), \, t \in [0,T],  \, x \in \mathbb{R}^D$, is the score function of the distribution $P_t$. The solution to this SDE (cf. [2]), denoted as $\big(\widehat{X}(t)\big)$, has marginals $Q(t):=\text{Law}[\widehat{X}(t)]$ with  
$$
\begin{align}
Q(t) = P(T-t).
\end{align}
$$
for all $t \in [0,T]$. In particular, the marginal distribution $Q(T)$ coincides with the data distribution $P(0)=P$.

In principle, we can therefore produce samples from $P$ by forward simulation of the reverse SDE (the standard approach would be to use an Euler-Maruyama discretization of the SDE). However, we run into two problems:
- We don't know the random initilisaion $P(T)$. 
- The bigger concern is that we don't have access to the score function $s_t$. We have seen in the previous section, that $p_t$ this is intractable, since can not mariginale over $P$ in (10). Consquently, we can not calculate the gradient of $\log p_t$ to obtain the score function. 

Fortunately, the first problem is not a big concern. We already established in the previous section that $P(t) \to \mathcal{N}(0, \Sigma(t))$ exponentially fast. Our first approximation will therefore be to initialize the forward simulation with $\mathcal{N}(0, \Sigma(T))$ instaed of $P(T)$. This approximation will be extremely good as long as $T$ is large enough.

The second problem is a bit more challenging. However, we can replace $s(t,x)$ in with a parametrised model $s_{\theta}(t,x)$ and try to learn the score function from the data. We are therefore in need of a differentiable loss function that can be used to find a paramter vector $\theta$ such that $s_\theta(t,x) \approx s(t,x)$.

A differentiable loss for score matching
----------------------------------------
Let $\big( \hat{X_\theta} (t) \big)$ be the solution to the SDE (12) with $s$ replaced by $s_\theta$ for fixed $\theta \in \Theta$ and random initilisation $\mathcal{N}(0, \Sigma(T))$.

Our goal is to choose a $\theta$ such that the sample paths of $\big(\hat{X_\theta}(t)\big)$ coincide with $\big(\widehat{X}(t)\big)$. Mathematically, this means that $\mathbf{P}$ and $ \mathbf{Q_\theta} \in \mathcal{P}\Big( C\big([0,T],\mathbb{R}^D\big) \Big)$ which we define as the path measures associated with $\big(\widehat{X}(t)\big)$ and $\big(\hat{X_\theta}(t)\big)$ on the space of continuous paths $C\big([0,T],\mathbb{R}^D \big)$ are required to be close to each other $ \mathbf{Q}_\theta \approx \mathbf{P}$. The closeness between two probability measures can be assessed by means of a metric or divergence and so training can be based on improving some measure of clossness between $\mathbf{P}$ and $\mathbf{Q}_\theta$. Typically, it is extremely hard to find such a measure of closeness that leads to a tractable objective, but in our special case an application of Girsanov's theorem allows us to derive the Kullback-Leibler (KL) divergence between the path measures:
$$
\begin{align}
    \ell(\theta) &:= 2 \cdot \text{KL}( \mathbf{P}, \mathbf{Q}_\theta) \\
                 & = -  \int_{0}^T \sigma^2(T-t) \mathbb{E} \Big[ \| s\big(T-t,\widehat{X}(t)\big) - s_{\theta}\big(T-t,\widehat{X}(t)\big) \|^2 \Big] \, dt + \text{ const.} \\
                 &=   \int_{0}^T \sigma^2(t) \mathbb{E} \Big[ \| s\big(t,X(t)\big) - s_{\theta}\big(t,X(t)\big) \|^2 \Big] \, dt + \text{ const. } 
\end{align}
$$
There are now several ways to generate unbiased estimators of $\ell$: [Standard score-matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) (very expensive), [sliced score matching](https://arxiv.org/pdf/1905.07088) (cheap but high variance) and [denoising score matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) (cheap and lower variance). The latter is by far the most common and looks a bit like magic initially. We introduce the key lemma first.

### Lemma. 
Let $Y$ be an arbitrary random variable. Define $\widetilde{Y}:= Y + \xi$ where $\xi \sim \mathcal{N}(0,\Sigma)$ and let further $\widetilde{s}$ be the score function of $\widetilde{Y}$. Then
    $$
    \begin{align}
        \mathbb{E} \left[ \| h(\widetilde{Y}) - \widetilde{s}(\widetilde{Y}) \|_2^2 \right] = \mathbb{E} \left[ \| h(\widetilde{Y}) - \Sigma^{-1}(Y-\widetilde{Y}) \|_2^2 \right]
    \end{align}
    $$
holds for arbitrary $h: \mathbb{R}^D \to \mathbb{R}^D$.


Notice that the LHS of (17) requires us to know the analytical form of the score-function $\widehat{s}$ whereas the RHS of (17) can easily be approximated by jointly sampling $(Y,\widetilde{Y})$ as long as $h$ is known. We apply this Lemma for each fixed $t \in [0,T]$. Recall that
$$ 
\begin{equation}
    X(t) \sim e^{-t} X_0 + \xi
\end{equation}
$$
where $\xi \sim \mathcal{N}\big(0,\Sigma(t)\big)$ and we therefore apply the Lemma with $Y:=e^{-t} X_0$, $\Sigma:=\Sigma(t)$ and $h= s(t, \cdot)$ which results in
$$ 
\begin{align}
    \ell(\theta) &= \int_0^T \sigma^2(t) \mathbb{E}  \left[ \| s_{\theta}\big(t, X(t) \big) - \Sigma(t)^{-1} \big( X(t) - e^{-t} X_0 \big) \|^2 \right] dt \\
    &= \int_0^T \sigma^2(t) \mathbb{E}  \left[ \| s_{\theta}\Big(t, e^{-t} X_0 + \sqrt{\Sigma(t)} Z(t) \Big) - \Sigma(t)^{-1/2} Z(t) \|^2 \right] dt \\
    &\approx \sum_{i=1}^I \sigma^2(t_i) \mathbb{E}  \left[ \| s_{\theta}\Big(t_i, e^{-t_i} X_0 + \sqrt{\Sigma(t_i)} Z(t_i) \Big) - \Sigma(t_i)^{-1/2} Z(t_i) \|^2 \right] (t_i - t_{i-1})
\end{align}
$$
where $0=t_0  < t_1 < \dots < t_I = T$ is a partition of $[0,T]$ and $Z(t)$ is a white noise Gaussian process. The expected value is now replaced by the MC-estimator and we obtain the final tractable objective
$$
\begin{align}
    \ell(\theta)= \frac{1}{N}\sum_{i=1}^I  \sum_{n=1}^N \sigma^2(t_i) (t_i - t_{i-1}) \| s_{\theta}\Big(t_i, e^{-t_i} X_{n} + \sqrt{\Sigma(t_i)} Z_n(t_i) \Big) - \Sigma(t_i)^{-1/2} Z_n(t_i) \|^2,  
\end{align}
$$
where $Z_n(t_i) \sim \mathcal{N}(0,I_D)$ independently for $n=1,\dots,N$ and $i=1,\dots,I$.









References:

[1] Lim, Jae Hyun, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti et al. "Score-based diffusion models in function space." arXiv preprint arXiv:2302.07400 (2023).

[2] Anderson, B.D., 1982. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3), pp.313-326.

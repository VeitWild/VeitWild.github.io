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

Let $X_1, \dots, X_N \sim P$ be a sample of size $N \in \mathbb{N}$ from an unknown data distribution $P \in \mathcal{P}(\mathbb{R}^D)$. 

Our goal is to generate a new sample $X^*$ which looks indistinguishable from $X_1, \dots, X_N$. The paradigmatic case is image generation, where a dataset with images is given and we want to build a sampler generating a new image from the unknown image data distribution.

Diffusion models (DM) allow us to progressively morph samples from a Gaussian distribution into the often highly complex and multimodal data distribution $P$. The mathematics behind diffusion models are stunningly beatiful and to my eye the most fascinating application of stochastic calculus in a long time. This blog post will focus on conceptual ideas behind DM and maybe provide some mathematical intution for why they are so succesful.


The forward process
-----------------------------------------

Let us start with the easy part. We want to turn the samples $X_1,\dots,X_N$ gradually into Gaussian noise. In order to accomplish this we turn our attention to a a very simple Stochastic Differential Equation (SDE) given as 
$$
\begin{align}
  X(0) &= x_0 \\
  dX(t) &= - X(t) \, dt + \sigma(t) \, d B(t),
\end{align}
$$
where $x_0 \in \mathbb{R}^D$ is an arbitrary initial condition, $\sigma: [0, T] \to (0, \infty)$ is the diffusion coefficient, $T >0 $ the time-horizon and $ ( B(t) )_{t ≥ 0}$ a Brownian motion. This SDE is amongst the easiest one will ever encounter. The drift term is linear in the space-variable $X(t)$ and the diffusion coefficient does not depend on $X(t)$. It is therefore a special case of a linear SDE and the solution, which we denote as $X_{x_0}(t)$, is readily available in closed form 
$$
\begin{align}
    X_{x_0}(t) \sim \mathcal{N}\big( m(t), \Sigma(t) I_D \big) 
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

The diffusion coefficient $\sigma(t)$ determines the noise schedule and is part the model architecture. Typically, we want to add small amounts of noise in the beginnning and as $t$  gets larger increase $\sigma(t)$. We will see later why this is the case. For now it's sufficient to know that succesfull schedulers are well-established in the literatur XXX and we dont need to learn $\sigma(t)$ from data.

Another crucial point to notice is that we can infer the long term behaviour of the SDE from (1). Specifically since $m(t) \to 0$ as $t \to \infty$ (exponentially fast), we find that 
$$
\begin{align}
  X_{x_0}(t) \approx \mathcal{N}\big(0 , \Sigma(t) I_D \big)
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
We denote the solution to this SDE as $\big( X(t) \big)_{t=0}^T$ (suppressing the $P$ subscript for notational convenience). Notice that it is easy to produce samples from $X(t)$ via
$$
\begin{align}
    X(t) \sim \exp(-t) X_0 +  \sqrt{\Sigma(t)} \epsilon
\end{align}
$$
with $\epsilon \sim N(0,1)$ and $X_0 \sim P$. However, we don't have a closed form expression for it's Lesbegue density $p_t$. More formally, we define $P(t):= \text{Law} \big[ X(t) \big]$ and $P(t|x_0) := \text{Law} \big[ X_{x_0}(t) \big] $
and note that
$$ 
\begin{align}
    p_t(x) = \int p_t(x|x_0) \, dP(x_0)
\end{align}
$$
where $p_t$ and $p_t(\cdot|x_0)$ denote the Lebesgue densities of $P(t)$ and $P(t|x_0)$. Here $p_t(x|x_0)=\mathcal{N}( x | x_0, \Sigma(t))$ is known but $p_t(x)$ is unknown, since we do not know $P$ and therfore can not calculate the integral in (10).


The time reversal
-----------------

So far we have only learned how we can morph an inital data distribution $P$ over time into a zero-mean Gaussian with known variance $\Sigma(t)$. The more interesting case is to reverse this process, i.e. how to go from Gaussian noise to a data sample.

To this end we consider the SDE with random initial condition $P_T$ given as
$$
\begin{align}
    & \widehat{X}(0) \sim P(T) \\
    &d \widehat{X}(t) =  \big[ \widehat{X}(t) + \sigma^2(T-t) s\big( T-t, \widehat{X}(t) \big)  \big] dt + \sigma(T-t) \, d \widehat{B}(t), 
\end{align}
$$
where $s(t,x) :=  \nabla \log p_t(x), \, t \in [0,T],  \, x \in \mathbb{R}^D$, is the score function of the distribution $P_t$. Anderson (1982) show that the solution to this SDE, denoted as $\big(\widehat{X}(t)\big)_{t=0}^T$, has marginals $Q(t):=\text{Law}[\widehat{X}(t)]$ with  
$$
\begin{align}
Q(t) = P(T-t).
\end{align}
$$
for all $t \in [0,T]$. In particular, the marginal $Q(T)$ coincides with the data distribution $P(0)=P$.

In principle, we can therefore produce samples from $P$ by forward simulation of the reverse SDE (the standard approach would be to use an Euler-Maruyama). However, we run into two problems:
- We don't know the random initilisaion $P(T)$. Fortunately, this can be remedied quite easily. As discussed in the previous section $P(t) \to \mathcal{N}(0, \Sigma(t))$ exponentially fast. Our first approximation will therefore be to initialise the forward simulation with $\mathcal{N}(0, \Sigma(T))$. This approximation will be extremely good as long as $T$ is large enough.
- The bigger concern is that we don't have access to the score function $s_t$. We have seen in the previous section, that $p_t$ this is intractable, since can not mariginale over $P$ in (10). Consquently, we can not calculate the gradient of $\log p_t$ to obtain the score function.




References:

[1] Anderson, B.D., 1982. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3), pp.313-326.

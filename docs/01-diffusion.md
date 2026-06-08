# Chapter 01 — Diffusion from First Principles

> *Goal of this chapter:* derive, from scratch, why gradually adding noise to data
> and then learning to reverse that process gives you a generative model. We build
> the forward process, prove its closed form, derive the reverse process and the
> *score function*, connect the score to a *denoiser* via Tweedie's formula, and
> derive the training objective. By the end you will understand the machinery that
> FLUX's *flow matching* (Chapter 02) is a particular, streamlined instance of.
>
> *Prerequisites:* multivariable calculus, basic probability (Gaussians,
> conditional densities, expectation), and comfort with linear algebra. No prior
> ML. Math is written in LaTeX; if your viewer doesn't render it, the prose around
> each equation states it in words.

> *Notation.* $x_0$ is a clean data point (an image, or for us a latent).
> $\mathbf{I}$ is the identity matrix. $\mathcal{N}(\mu,\Sigma)$ is a Gaussian.
> $p_{\text{data}}$ is the (unknown) distribution we want to sample from.
> $\nabla_x$ is the gradient with respect to $x$. We work in $\mathbb{R}^d$ where
> $d$ is the dimensionality of a latent (for us, tens of thousands).

---

## 1.1 The generative modeling problem

We have a dataset of samples $x_0 \sim p_{\text{data}}$ — millions of images. We
want a procedure that produces *new* samples from the same distribution. We never
get to see $p_{\text{data}}$ itself; we only have samples.

The difficulty is dimensionality. A $1024\times1024\times3$ image lives in
$\mathbb{R}^{3{,}145{,}728}$. The set of "plausible images" is a vanishingly thin,
intricately curved manifold inside that enormous space. Sampling from it directly
is hopeless: pick a random point and with overwhelming probability you get noise,
not a picture.

Historically there were several attacks on this problem — GANs (learn a generator
network adversarially), VAEs (learn an encoder/decoder with a latent prior),
autoregressive models (factorize $p(x) = \prod_i p(x_i \mid x_{<i})$). Diffusion
models took over image generation around 2020–2022 because they are **stable to
train** and produce **high sample quality and diversity**. Their core idea is
almost suspiciously simple, and the rest of this chapter is its derivation.

> **The idea in one line.** It is hard to jump from noise to an image in one step,
> but it is *easy* to take a tiny step from a slightly-noisier image toward a
> slightly-cleaner one. So: learn the tiny step, and take many of them.

## 1.2 The forward process: destroying structure on purpose

Define a sequence that progressively corrupts a clean sample $x_0$ with Gaussian
noise over "time" $t$ running from $0$ (clean) to $T$ (pure noise). In discrete
form (the DDPM formulation), pick a *noise schedule* $\beta_1,\dots,\beta_T \in
(0,1)$ and define each step as

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0,\mathbf{I}).
$$

Equivalently, $q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(\sqrt{1-\beta_t}\,x_{t-1},\,
\beta_t \mathbf{I}\big)$. Each step scales the previous value down slightly (the
$\sqrt{1-\beta_t}$ factor) and injects a little fresh noise. The down-scaling is
what keeps the variance from exploding; we will see in a moment that it keeps the
total variance bounded at $1$.

```
 x0           x1            x2                         x_T
 ┌────┐  +β1  ┌────┐  +β2   ┌────┐         ...         ┌────┐
 │cat │ ────▶ │cat'│ ─────▶ │blur│ ──────────────────▶│noise│
 └────┘       └────┘        └────┘                     └────┘
 clean      slightly       noticeably               indistinguishable
            noisy          corrupted                from N(0,I)
```

This **forward process is fixed** — there is nothing to learn here. It is a known
Markov chain that turns data into noise. The art is in *reversing* it.

### 1.2.1 The closed form: jumping to any time $t$ in one shot

Applying the recursion $t$ times looks expensive, but Gaussians compose, and the
result telescopes into a clean closed form. Define

$$
\alpha_t := 1-\beta_t, \qquad \bar\alpha_t := \prod_{s=1}^{t} \alpha_s .
$$

**Claim.** $\;x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$,
with $\epsilon \sim \mathcal{N}(0,\mathbf{I})$. Equivalently

$$
q(x_t \mid x_0) = \mathcal{N}\!\big(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)\,\mathbf{I}\big).
$$

**Proof (by unrolling two steps, then induction).** Start from
$x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$ and substitute
$x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}$:

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2}
      + \underbrace{\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}
      + \sqrt{1-\alpha_t}\,\epsilon_t}_{\text{sum of two independent Gaussians}} .
$$

The two noise terms are independent zero-mean Gaussians, so their sum is a single
zero-mean Gaussian whose variance is the sum of the variances:

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t
= 1 - \alpha_t\alpha_{t-1}.
$$

So $x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon$.
This has exactly the same shape as the one-step rule with $\alpha_t \to
\alpha_t\alpha_{t-1}$. Repeating down to $x_0$ replaces the product by
$\bar\alpha_t = \prod_{s\le t}\alpha_s$, giving the claim. $\;\blacksquare$

Two consequences worth pausing on:

- **The variance is bounded.** $\operatorname{Var}[x_t \mid x_0] = (1-\bar\alpha_t)\mathbf{I}$,
  and the signal coefficient is $\sqrt{\bar\alpha_t}$. As $t$ grows $\bar\alpha_t
  \to 0$, so $x_T \approx \mathcal{N}(0,\mathbf{I})$ — pure noise, with no memory of
  $x_0$. This is the *variance-preserving* (VP) schedule: $(\text{signal})^2 +
  (\text{noise})^2 = \bar\alpha_t + (1-\bar\alpha_t) = 1$ at every $t$.

- **Training is cheap.** To get a training pair at any noise level we do not
  simulate the chain; we sample $x_0$ from the data, sample one $\epsilon$, pick a
  $t$, and compute $x_t$ in closed form. One draw, any time level. This is the
  property that makes diffusion practical.

### 1.2.2 The signal-to-noise reparameterization

It is often cleaner to forget the discrete index and describe the corruption by a
single scalar. Write $x_t = a_t x_0 + \sigma_t \epsilon$ with $a_t =
\sqrt{\bar\alpha_t}$ and $\sigma_t = \sqrt{1-\bar\alpha_t}$. The **signal-to-noise
ratio** $\text{SNR}(t) = a_t^2/\sigma_t^2$ decreases monotonically from $\infty$
(clean) to $0$ (noise). Everything that matters about a schedule is *how SNR
sweeps from high to low*; the specific $\beta_t$ values are just one way to
parameterize that sweep. Flow matching (Chapter 02) chooses a different, simpler
sweep — keep this reparameterization in mind, it is the bridge.

## 1.3 The reverse process: where the learning lives

We want to go backward: start from $x_T \sim \mathcal{N}(0,\mathbf{I})$ and undo
the noise step by step until we reach a clean $x_0$. The reverse of a Markov
chain is also Markov, so we want $q(x_{t-1}\mid x_t)$. The problem: this density
depends on the data distribution and is intractable.

But there is a beautiful fact. **Conditioned on the original $x_0$**, the reverse
step *is* a tractable Gaussian. Using Bayes' rule,

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)},
$$

and all three factors on the right are the Gaussians we already derived.
Multiplying them and completing the square (a mechanical but tedious computation)
gives another Gaussian:

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\big(\tilde\mu_t(x_t,x_0),\, \tilde\beta_t\mathbf{I}\big),
$$

$$
\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0
+ \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t .
$$

The catch is the dependence on $x_0$, which we do not have at sampling time — that
is the whole point, we are *trying* to produce $x_0$. So the model's job is to
**estimate the quantity that lets us recover $\tilde\mu_t$**. There are three
equivalent things it could predict; they are reparameterizations of each other:

1. the original $x_0$ directly,
2. the noise $\epsilon$ that was added (the "$\epsilon$-prediction" of DDPM),
3. the **score** $\nabla_{x_t}\log q(x_t)$ — the gradient of the log-density.

The score is the most theoretically illuminating, so we derive it next; FLUX ends
up predicting a fourth, closely related quantity (a *velocity*), and Chapter 02
shows it is the same idea in a tidier coordinate system.

## 1.4 The score function and Tweedie's formula

The **score** of a distribution is $\nabla_x \log p(x)$ — a vector field pointing
in the direction of increasing log-probability, i.e. "toward more plausible data."
If you knew the score everywhere, you could sample by starting anywhere and
walking uphill (with the right amount of noise — §1.6).

For the *conditional* forward density we can compute the score exactly. Since
$q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf I)$, its
log is $-\tfrac{1}{2(1-\bar\alpha_t)}\lVert x_t - \sqrt{\bar\alpha_t}x_0\rVert^2 +
\text{const}$, and so

$$
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}
= -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}} = -\frac{\epsilon}{\sigma_t},
$$

using $x_t - \sqrt{\bar\alpha_t}x_0 = \sigma_t \epsilon$. **The score is the added
noise, negated and scaled.** This is the precise sense in which "estimating the
score" and "estimating the noise" are the same task — predicting $\epsilon$ and
predicting $\nabla\log q$ differ only by the known factor $-1/\sigma_t$.

### Tweedie's formula: the denoiser *is* the score

There is a deeper identity connecting the score of the *marginal* $p_t(x_t) =
\int q(x_t\mid x_0)p_{\text{data}}(x_0)\,dx_0$ to the best possible denoiser. For a
Gaussian-corrupted observation, **Tweedie's formula** states the posterior mean of
the clean signal is

$$
\mathbb{E}[x_0 \mid x_t] = \frac{1}{\sqrt{\bar\alpha_t}}
\Big(x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)\Big).
$$

*Derivation sketch.* Write $p_t(x_t) = \int \mathcal N(x_t; a_t x_0, \sigma_t^2
\mathbf I)\,p_{\text{data}}(x_0)\,dx_0$. Differentiate under the integral:
$\nabla_{x_t} p_t = \int \frac{a_t x_0 - x_t}{\sigma_t^2}\,\mathcal N(\cdot)\,
p_{\text{data}}\,dx_0$. Divide by $p_t$ to turn the integrand's $\mathcal N\,
p_{\text{data}}/p_t$ into the posterior $p(x_0\mid x_t)$, and you get
$\nabla_{x_t}\log p_t = \frac{a_t\,\mathbb E[x_0\mid x_t]-x_t}{\sigma_t^2}$.
Solve for $\mathbb E[x_0\mid x_t]$. $\;\blacksquare$

Read it slowly: **the optimal denoiser — the conditional mean of the clean image
given the noisy one — is a simple algebraic function of the score.** A network
trained to denoise is, automatically, a network that has learned the score. This
is why the three prediction targets in §1.3 are interchangeable: $x_0$-prediction,
$\epsilon$-prediction, and score-estimation are linear reparameterizations of one
another. The model architecture (the MMDiT of Part II) is the *function
approximator* for this denoiser/score; the choice of *what exactly it outputs* is
a coordinate convention, and FLUX's choice is Chapter 02.

## 1.5 The training objective

We want a network $s_\theta(x_t, t)$ that approximates the marginal score
$\nabla_{x_t}\log p_t(x_t)$. We cannot evaluate that target — $p_t$ is the
intractable marginal. **Denoising score matching** (Vincent, 2011) rescues us: it
proves that matching the *conditional* score (which we *can* evaluate, §1.4)
yields the same minimizer as matching the marginal score. Concretely, the two
objectives

$$
\mathbb{E}_{x_t\sim p_t}\big\lVert s_\theta(x_t,t) - \nabla_{x_t}\log p_t(x_t)\big\rVert^2
\quad\text{and}\quad
\mathbb{E}_{x_0,\,x_t\sim q(\cdot\mid x_0)}\big\lVert s_\theta(x_t,t) - \nabla_{x_t}\log q(x_t\mid x_0)\big\rVert^2
$$

differ only by a constant independent of $\theta$. *Why:* expand the marginal-score
objective, and the cross term $\mathbb E[\langle s_\theta, \nabla\log p_t\rangle]$
can be rewritten — using $\nabla p_t = \int \nabla q(\cdot\mid x_0)p_{\text{data}}$
— as $\mathbb E_{x_0,x_t}[\langle s_\theta, \nabla\log q(x_t\mid x_0)\rangle]$,
which is the cross term of the conditional objective. The squared-norm-of-target
terms differ by a $\theta$-independent constant. Same gradient, same minimizer.

Substituting the conditional score $\nabla\log q(x_t\mid x_0) = -\epsilon/\sigma_t$
and reparameterizing the network to predict the noise directly,
$\epsilon_\theta(x_t,t) := -\sigma_t\, s_\theta(x_t,t)$, the objective collapses to
the famous, almost trivial-looking DDPM loss:

$$
\boxed{\;\mathcal{L}(\theta) = \mathbb{E}_{x_0\sim p_{\text{data}},\;
\epsilon\sim\mathcal N(0,\mathbf I),\; t\sim\mathcal U\{1,\dots,T\}}
\Big[\,\big\lVert \epsilon - \epsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon}_{x_t},\, t)\big\rVert^2\,\Big]\;}
$$

The entire training procedure is: *take a real image, noise it to a random level,
ask the network to guess the noise, penalize the squared error.* That is all.
There is no adversary, no sampling during training, no instability. This
robustness is why diffusion won.

> **Conditioning.** For a text-to-image model the network also receives the prompt
> conditioning $c$ (the Qwen3 embedding, Chapter 05): $\epsilon_\theta(x_t,t,c)$.
> The derivation is unchanged; $c$ is just an extra input. The model learns
> $\nabla_{x_t}\log p_t(x_t\mid c)$ — the score of images *given the prompt*.

## 1.6 Sampling: turning a score into images

With a trained score model, how do we generate? Two views, both important.

**Stochastic (ancestral) sampling.** Plug $\epsilon_\theta$ into the reverse
Gaussian of §1.3: start at $x_T\sim\mathcal N(0,\mathbf I)$, and for $t=T,\dots,1$
draw $x_{t-1}\sim\mathcal N(\tilde\mu_t,\tilde\beta_t\mathbf I)$ with
$\tilde\mu_t$ reconstructed from the predicted noise. This is the original DDPM
sampler; it takes many (hundreds of) steps.

**The continuous view and the probability-flow ODE.** Take the noising process to
the continuous-time limit and it becomes a stochastic differential equation
$dx = f(x,t)\,dt + g(t)\,dW$. Anderson's theorem gives the time-reversed SDE,
which involves the score:

$$
dx = \big[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar W .
$$

Remarkably, there is a **deterministic** ODE with the *same marginal densities*
$p_t$ at every time — the *probability-flow ODE*:

$$
\frac{dx}{dt} = f(x,t) - \tfrac{1}{2}g(t)^2\,\nabla_x\log p_t(x).
$$

Sampling is then just **numerically integrating an ODE** from $t=T$ to $t=0$,
using the learned score as the velocity field. Fewer, larger steps become possible
with good integrators. This deterministic view is the doorway to flow matching:
if all we need is a good velocity field whose flow carries $\mathcal N(0,\mathbf
I)$ onto $p_{\text{data}}$, maybe we can *learn that velocity field directly* and
choose the straightest possible paths. That is exactly what FLUX does, and it is
Chapter 02.

## 1.7 Where this lives in the code

This chapter is pure theory, but two of its objects appear directly in the runtime
and are worth locating now so Part II/VII feel grounded:

- **The starting noise** $x_T \sim \mathcal N(0,\mathbf I)$. In
  `tools/generate.cu` the initial latent is drawn from a standard normal (not a
  uniform — rectified-flow models are trained against Gaussian noise, a subtlety
  that was once a bug; see Appendix B). The seed makes it reproducible.

- **The sampler / the reverse update.** `src/backend/cuda/sampler.{h,cu}`
  implements the scheduler and the per-step update `latent += dt · v`. In the
  score language of this chapter that update is one integration step of the
  reverse process; in the flow-matching language of the next chapter it is one
  Euler step of an ODE whose velocity field is the transformer's output. Same
  arithmetic, two derivations.

- **The network $\epsilon_\theta$ / $v_\theta$.** The entire FLUX transformer
  (`flux_transformer.cu`, Chapters 04–10) *is* the function approximator for the
  score/velocity. Everything in Parts II–IV is about evaluating that one function
  quickly.

## 1.8 Summary and what to carry forward

- A **fixed** forward process turns data into noise; its closed form
  $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ makes training a
  one-draw operation at any noise level.
- The **reverse** process is where learning happens. The optimal reverse step is
  governed by the **score** $\nabla\log p_t$, which (Tweedie) is equivalent to the
  optimal **denoiser**, which is equivalent to **predicting the noise**.
- The **training loss** is a trivial denoising regression — predict the noise you
  added — and that simplicity is diffusion's superpower.
- **Sampling** is integrating the reverse SDE, or equivalently its deterministic
  **probability-flow ODE**. The deterministic view motivates flow matching.

The mental model to keep: *the network is a learned vector field that points noisy
latents toward the data manifold (conditioned on the prompt); generation is
following that field from noise to image.* Chapter 02 replaces the noise-schedule
machinery with the cleanest possible version of this field — straight-line paths —
and shows that this is precisely what FLUX.2-klein computes, four steps at a time.

---

### Exercises

1. **Variance preservation.** Verify $(\sqrt{\bar\alpha_t})^2 + (\sqrt{1-\bar\alpha_t})^2 = 1$
   and explain in words why down-scaling the signal at each step is necessary for
   $x_T$ to be standard normal regardless of $x_0$.
2. **Score ↔ noise.** Starting from $q(x_t\mid x_0)$, re-derive $\nabla_{x_t}\log
   q(x_t\mid x_0) = -\epsilon/\sigma_t$ and state the constant relating
   $s_\theta$ and $\epsilon_\theta$.
3. **Tweedie in 1-D.** Let $p_{\text{data}}$ be a single point mass at $\mu$.
   Compute $p_t$, its score, and check Tweedie's formula returns
   $\mathbb E[x_0\mid x_t]=\mu$.
4. **ODE vs SDE.** Explain why the probability-flow ODE and the reverse SDE can
   share marginals yet produce different individual sample trajectories. Which one
   is reproducible given a fixed start, and which is not?

*Next: [Chapter 02 — Flow matching & rectified flow](02-flow-matching.md), where
the noise schedule disappears and the velocity field becomes a straight line.*

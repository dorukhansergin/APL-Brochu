We have a set of $N$ items that can be described in the two dimensional space with their associated vectors.

$$ 
\mathcal{X} = \{x_1, x_2, \dots, x_N \}
$$

Assume the real preference of the user can be described by a utility function $f(x)$.

Given two inputs to compare, $x_1$ and $x_2$, the user will prefer the first item if $f(x_1) > f(x_2)$ and vice versa.

Our goal is to find an item that is good enough for the user, as quickly as possible. We do this by presenting the user
pairwise comparisons, and better these comparisons each time the user picks one, through active learning. In mathematical terms,
we are trying to find a point where $f(x)$ yields a large enough value and we want to this in less than $T$ queries.

The user, however, will not be a perfect decision maker, and there will be a slight noise in her judgement at comparison.
We can assume a zero-centered Gaussian noise to model this, which is injected to the utilities of the two items at each comparison.
We also assume this noise is iid, so the noise is independent from one query to another.

$$ 
u(x) = f(x) + \epsilon, \hspace{10px} \epsilon \sim \mathcal{N}(0, \sigma)
$$

We model $f(x)$ through a zero-mean Gaussian Process with a kernel function $K(.,.)$:

$$ 
f(x) \sim GP(0, K(.,.)) 
$$

How do we update our belief about $f(x)$. The prior above assumes all items are equally likable by the user, because of the constant mean assumption.
Well, we present the user two items at each iteration, indexed by $m$. Let's denote $c_m$ as the inferior and $r_m$ as the superior pick at each iteration.

The item $r_k$ will be preferred over $c_k$ with the following probability:
$$
P(u(r_m) > u(c_m)) = P(\epsilon_{cm} - \epsilon_{rm} > f(r_m) - f(c_m)) = \Phi[\frac{f(r_m) - f(c_m)}{\sqrt{2}\sigma}]
$$
where $\Phi$ is the cdf of a standard normal distribution.

Thus, the data we provide to the algorithm is the preferences of the user at each iteration.
$$
D = {r_m \succ c_m: k \in 1,2,\dots,M}
$$

All we have to do is to infer the posterior distribution of the utility function given this data.

$$
p(f|D) \propto p(f)p(D|f)
$$

After each iteration $k$, we feed back this posterior as prior to the inference in the next iterations, thus active learning.

The query is selected using the upper confidence bound method. 
This is one difference between the original paper, in which expected improvement is used.
Both are implemented, though, should you want to switch.
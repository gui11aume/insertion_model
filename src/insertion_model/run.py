"""Train the insertion model."""

import chr1_SRR12322252
import matplotlib.pyplot as plt
import pyro
import torch
import torch.nn.functional as f
from torch.distributions import constraints
from tqdm import tqdm


# With `alpha` set to 1, the distribution of `beta` is uniform.
# This means that the remaining part of the "stick" is broken
# uniformly every time we need to create a cluster. If `alpha`
# is less than 1, the distribution of `beta` is biased to
# values closer to 1. This means that the first cluster will
# tend to have a large weight, leaving little for further
# clusters. This corresponds to a situation where the data can
# be fitted into one major cluster. If `alpha` is higher than
# 1, the distribution of `beta is biased to values closer to 0.
# This means that the first cluster will be small, leaving a
# lot for further clusters. This corresponds to a case where
# multiple clusters are needed to fit the data.
DEVICE = "cuda"
T = 25  # max number of breaks; needs a relatively high number but 25 is for exploration.
alpha = 1 * torch.ones(1).to(DEVICE)


def mix_weights(beta):
    """Helper function to convert beta coeffs to probabilities.

    Those are the breakpoints on the uniform "stick" (0,1).
    """
    beta1m_cumprod = (1.0 - beta).cumprod(-1)
    return f.pad(beta, (0, 1), value=1) * f.pad(beta1m_cumprod, (1, 0), value=1)


def model(x):
    """The model."""
    # Sample beta, the stick-breaking parameters. They determine
    # the locations of the breakpoints on the "stick". The "stick"
    # is an abstract object that measures the number of clusters
    # (the number of fragments in the broken stick) and their
    # probabilities (the length of the fragments). There are `T-1`
    # breakpoints, so there are `T` clusters.
    with pyro.plate("T-1", T - 1):
        beta = pyro.sample(name="beta", fn=pyro.distributions.Beta(1.0, alpha))

    # NOTE: I keep beta as the name of the variable above, but I
    # would like to change it. I find it very confusing because
    # there are two Beta distributions: the first describes the
    # breaking of the stick, the second describes the positions
    # of the insertion sites.

    with pyro.plate("T", T):
        # Each cluster is a Beta(a1, a0) distribution, that is
        # characterized by typical position on the chromosome and
        # a spread. The typical position is a1 / (a0 + a1) and the
        # spread decreases as a0 and a1 increase. We sample `a0_`
        # and `a1_` from a half Cauchy so that very large values
        # are plausible (large values correspond to narrow peaks,
        # i.e., to hotspots), we define a1 = a1_ + 1, a0 = a0_ + 1
        # so that the values are greater than 1 (when a1 or a0 is
        # less than 1, the Beta distribution is u-shaped instead of
        # n-shaped, which has no biological interpretation here).
        a1_ = pyro.sample(
            name="a1_",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.ones(1).to(DEVICE),
            ),
        )
        a0_ = pyro.sample(
            name="a0_",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.ones(1).to(DEVICE),
            ),
        )

    with pyro.plate("obs", x.shape[0]):
        # Convert `beta` to probabilities (weight of each cluster).
        probs = mix_weights(beta).unsqueeze(-2)
        # Sample a cluster for every observation (insertion site).
        z = pyro.sample(
            name="z",
            fn=pyro.distributions.Categorical(probs),
        )

        # Finally, the observations are drawn as a Beta distribution
        # from the cluster that was assigned to them. If a1 and a0 are
        # large, the observation will be close to a1 / (a0 + a1),
        # otherwise, it can be quite far from the typical location.
        a1 = a1_.gather(dim=-1, index=z) + 1.0
        a0 = a0_.gather(dim=-1, index=z) + 1.0
        pyro.sample(name="x", fn=pyro.distributions.Beta(a1, a0), obs=x)


def guide(x):
    """The guide."""
    # The parameter `kappa` controls the variational posterior of
    # `beta`. The variational posterior of `beta` is assumed to
    # be Beta(1., kappa), so large values of `kappa` correspond
    # to smaller `beta`, and therefore clusters of smaller weight.
    # Reciprocally, small values of `kappa` correspond to bigger
    # `beta` and therefore clusters of bigger weight.
    kappa = pyro.param(
        "kappa", lambda: torch.ones(T - 1).to(DEVICE), constraint=constraints.positive
    )

    # The variational posterior of `a1` and `a0` is log-normal.
    # Level 1: (2,6) (4,4), (6,2)
    # Level 2: (2,14) (4,12) (6,10) ...
    # Level 3: (2,30) (4,28) (6,26) ...

    # fmt: off
    a1_mu = pyro.param(
        "a1_mu",
        lambda: torch.tensor([
            2,4,6, # Level 1
            2,4,6,8,10,12,14, # Level 2
            2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 # Level 3
        ]).log().to(DEVICE),
    )
    a0_mu = pyro.param(
        "a0_mu",
        lambda: torch.tensor([
            6,4,2, # Level 1
            14,12,10,8,6,4,2, # Level 2
            30,28,26,24,22,20,18,16,14,12,10,8,6,4,2 # Level 3
        ]).log().to(DEVICE),
    )
    # fmt: on
    a1_sd = pyro.param(
        "a1_sd", lambda: 0.01 * torch.ones(T).to(DEVICE), constraint=constraints.positive
    )
    a0_sd = pyro.param(
        "a0_sd", lambda: 0.01 * torch.ones(T).to(DEVICE), constraint=constraints.positive
    )

    # `phi` is the posterior assignment of each
    # observation to a cluster. For now we use a
    # naive initial value, but we should initialize
    # it to a more meaningful value.
    phi = pyro.param(
        "phi", 1.0 / T * torch.ones(x.shape[0], T).to(DEVICE), constraint=constraints.simplex
    )

    # This is the variational posterior distribution
    # of the `beta` coefficients.
    with pyro.plate("T-1", T - 1):
        pyro.sample(name="beta", fn=pyro.distributions.Beta(torch.ones(1).to(DEVICE), kappa))

    # This is the variational posterior distribution
    # of `a1_` and `a0_`. The log-normal distribution
    # allows us to interpret the parameters in a
    # relatively easy way.
    with pyro.plate("T", T):
        pyro.sample(name="a1_", fn=pyro.distributions.LogNormal(a1_mu, a1_sd))
        pyro.sample(name="a0_", fn=pyro.distributions.LogNormal(a0_mu, a0_sd))

    # This is the variational posterior of the cluster
    # assignment. This is a discrete distribution so
    # we should use enumeration to speed up the inference
    # but we are going to leave it as is for now.
    with pyro.plate("obs", x.shape[0]):
        pyro.sample(
            name="z",
            fn=pyro.distributions.Categorical(phi),
        )


x = torch.tensor(chr1_SRR12322252.x).to(DEVICE)
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(
    model, guide, optim, loss=pyro.infer.Trace_ELBO(vectorize_particles=True, num_particles=24)
)
losses = []


def train(num_iterations):
    """Train the model."""
    pyro.clear_param_store()
    for _ in tqdm(range(num_iterations)):
        loss = svi.step(x)
        losses.append(loss)


train(5000)

print(pyro.param("kappa"))
print(pyro.param("phi"))
print(pyro.param("a1_mu"))
print(pyro.param("a1_sd"))
print(pyro.param("a0_mu"))
print(pyro.param("a0_sd"))


plt.plot(range(len(losses)), losses, color="skyblue")
plt.xlabel("Iteration")
plt.ylabel("ELBO loss")
plt.title("Loss curve")
plt.show()

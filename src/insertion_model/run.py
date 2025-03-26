"""Train the insertion model."""

import chr1_exp2_bat1_SRR12322252
import chr1_exp2_bat1_SRR12322265
import matplotlib.pyplot as plt
import pyro
import torch
import torch.nn.functional as f
from torch.distributions import constraints
from tqdm import tqdm


# With `alpha` set to 1.5, the distribution of `beta` is
# approximately uniform. This means that the remaining part
# of the "stick" is broken uniformly every time we need to
# create a cluster. If `alpha` is less than 1.5, the distribution
# of `beta` is biased to values closer to 0.5, so this is not a
# good method to parametrize `beta`. In order to make a better
# parametrization, one needs to introduce a new hyperparameter
# that shifts the distribution of `beta` to the left or right,
# i.e., closer to 0 or closer to 1.
DEVICE = "cuda"
alpha = 1.5 * torch.ones(1).to(DEVICE)

T = 25  # Max number of breaks; needs a relatively high number but 25 is for exploration.
N = 2  # Number of replicates.


def mix_weights(beta):
    """Helper function to convert beta coeffs to probabilities.

    Those are the breakpoints on the uniform "stick" (0,1).
    """
    beta1m_cumprod = (1.0 - beta).cumprod(-1)
    return f.pad(beta, (0, 1), value=1) * f.pad(beta1m_cumprod, (1, 0), value=1)


# The shapes below are described in functions of `P` the number of
# Pyro particles, which is a runtime parameter, `N` the number of
# replicates and `T` the maximum number of clusters.


def model(x_pos, x_batch):
    """The model."""
    # Define `n` as the number of observations.
    n = x_pos.shape[0]

    # Sample beta, the stick-breaking parameters. They determine
    # the locations of the breakpoints on the "stick". The "stick"
    # is an abstract object that measures the number of clusters
    # (the number of fragments in the broken stick) and their
    # probabilities (the length of the fragments). There are `T-1`
    # breakpoints, so there are `T` clusters.
    with pyro.plate("T-1", T - 1):
        # Instead of sampling beta directly, we sample eta and
        # then use the sigmoid function to get beta. This is a
        # trick to ensure that beta is always between 0 and 1.
        # Shape(eta): [P, 1, T-1]
        eta = pyro.sample(
            "eta",
            pyro.distributions.Normal(
                loc=0.0 * torch.ones(T - 1, device=DEVICE),
                scale=alpha * torch.ones(T - 1, device=DEVICE),
            ),
        )

        with pyro.plate("NxT", N):
            # We "break the stick" `T-1` times, each time creating
            # a new cluster of a given size. The parameters `eta`
            # specify how we break the stick and `epsilon` are
            # little shifts that depend on the batch. This means
            # that the cluster size depends on the batch.
            # Shape(epsilon): [P, N, T-1]
            epsilon = pyro.sample(
                "epsilon",
                pyro.distributions.Normal(
                    loc=0.0 * torch.ones(1, 1, device=DEVICE),
                    scale=0.05 * torch.ones(1, 1, device=DEVICE),
                ),
            )

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
        # Shape (a1_): [P, T]
        a1_ = pyro.sample(
            name="a1_",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.ones(1).to(DEVICE),
            ),
        )
        # Shape (a0_): [P, T]
        a0_ = pyro.sample(
            name="a0_",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.ones(1).to(DEVICE),
            ),
        )

    # NOTE: I keep beta as the name of the variable above, but I
    # would like to change it. I find it very confusing because
    # there are two Beta distributions: the first describes the
    # breaking of the stick, the second describes the positions
    # of the insertion sites.
    beta = torch.sigmoid(eta + epsilon)  # Shape: [P, N, T-1]

    with pyro.plate("obs", n):
        # Convert `beta` to probabilities (weight of each cluster).
        probs = mix_weights(beta)  # Shape: [P, N, T]
        # Create indices for gathering along N dimension
        batch_indices = x_batch.unsqueeze(0).unsqueeze(-1).expand(probs.shape[0], -1, 1)
        # Gather probabilities for each observation's batch across all T clusters
        probs_for_each_x = torch.gather(
            probs, dim=1, index=batch_indices.expand(-1, -1, probs.shape[2])
        )  # Shape: [P, n, T]
        # Reshape `probs_for_each_x` because the last dimention (T) does not
        # contribute to the event shape: it is only used for the parameter
        # so it is consumed during generation. The implicit "particle"
        # outer plate has to be at index -3, so we compensate by adding a
        # dimension now. The tensor has four dimensions, but if we disregard
        # the last (T) as parameter-only, the numbers coincide with the size
        # of the plates.
        # Shape(probs_for_each_x): [P, 1, n, T]
        probs_for_each_x = probs_for_each_x.unsqueeze(-3)
        # The implementation below is probably equivalent to the above.
        # I need to check which works in the context of multiple particles.
        # batch_indices = x_batch.unsqueeze(0).unsqueeze(-1).expand(probs.shape[0], -1, 1)
        # batch_indices = batch_indices.expand(-1, -1, probs.shape[2])
        # # Gather probabilities based on batch indices
        # probs_per_x = torch.gather(probs, dim=1, index=batch_indices)
        # Sample a cluster for every observation (insertion site).
        z = pyro.sample(
            # Shape: [P, n]
            name="z",
            fn=pyro.distributions.Categorical(probs_for_each_x),
        )

        # Finally, the observations are drawn as a Beta distribution
        # from the cluster that was assigned to them. If a1 and a0 are
        # large, the observation will be close to a1 / (a0 + a1),
        # otherwise, it can be quite far from the typical location.
        a1 = a1_.gather(dim=-1, index=z) + 1.0
        a0 = a0_.gather(dim=-1, index=z) + 1.0
        pyro.sample(name="x_pos", fn=pyro.distributions.Beta(a1, a0), obs=x_pos)


def guide(x_pos, x_batch):  # noqa: ARG001
    """The guide."""
    eta_loc = pyro.param("eta_loc", torch.zeros(T - 1, device=DEVICE))
    eta_scale = pyro.param(
        "eta_scale", torch.ones(T - 1, device=DEVICE), constraint=constraints.positive
    )

    epsilon_loc = pyro.param("epsilon_loc", torch.zeros(N, T - 1, device=DEVICE))
    epsilon_scale = pyro.param(
        "epsilon_scale", torch.ones(N, T - 1, device=DEVICE), constraint=constraints.positive
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
        "phi", 1.0 / T * torch.ones(x_pos.shape[0], T).to(DEVICE), constraint=constraints.simplex
    )

    # This is the variational posterior distribution
    # of the `beta` coefficients.
    with pyro.plate("T-1", T - 1):
        _ = pyro.sample("eta", pyro.distributions.Normal(eta_loc, eta_scale))

        with pyro.plate("NxT", N):
            _ = pyro.sample("epsilon", pyro.distributions.Normal(epsilon_loc, epsilon_scale))

    # This is the variational posterior distribution
    # of `a1_` and `a0_`. The log-normal distribution
    # allows us to interpret the parameters in a
    # relatively easy way.
    with pyro.plate("T", T):
        _ = pyro.sample(name="a1_", fn=pyro.distributions.LogNormal(a1_mu, a1_sd))
        _ = pyro.sample(name="a0_", fn=pyro.distributions.LogNormal(a0_mu, a0_sd))

    # This is the variational posterior of the cluster
    # assignment. This is a discrete distribution so
    # we should use enumeration to speed up the inference
    # but we are going to leave it as is for now.
    with pyro.plate("obs", x_pos.shape[0]):
        _ = pyro.sample(
            name="z",
            fn=pyro.distributions.Categorical(phi),
        )


x_pos_1 = torch.tensor(chr1_exp2_bat1_SRR12322252.x)
x_pos_2 = torch.tensor(chr1_exp2_bat1_SRR12322265.x)

x_pos = torch.cat([x_pos_1, x_pos_2], dim=0).to(DEVICE)
x_batch = (
    torch.cat([torch.zeros(x_pos_1.shape[0]), torch.ones(x_pos_2.shape[0])], dim=0)
    .long()
    .to(DEVICE)
)

optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(
    model,
    guide,
    optim,
    loss=pyro.infer.Trace_ELBO(
        vectorize_particles=True,
        num_particles=24,
        max_plate_nesting=2,
    ),
)
losses = []


def train(num_iterations):
    """Train the model."""
    pyro.clear_param_store()
    for _ in tqdm(range(num_iterations)):
        loss = svi.step(x_pos, x_batch)
        losses.append(loss)


train(5000)

print(pyro.param("eta_loc"))
print(pyro.param("eta_scale"))
print(pyro.param("epsilon_loc"))
print(pyro.param("epsilon_scale"))
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

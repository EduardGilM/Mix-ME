import jax
import jax.numpy as jnp
from jax.random import PRNGKey as KeyArray
from qdax.core.neuroevolution.networks.networks import MLP
from typing import List
import optax

def distill_knowledge(
    transitions: List[dict],
    observation_size: int = 376,
    action_size: int = 17,
    hidden_layer_size: int = 128,
    batch_size: int = 64,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    random_key: KeyArray = jax.random.PRNGKey(0)
) -> MLP:
    # Initialize the distillation network
    distillation_network = MLP(
        layer_sizes=(hidden_layer_size, hidden_layer_size, action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Initialize the parameters
    sample_input = jnp.zeros((1, observation_size))
    params = distillation_network.init(random_key, sample_input)

    # Prepare the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Prepare the data
    observations = []
    actions = []

    for transition in transitions:
        obs = transition['obs']
        action = transition['actions']
        
        if isinstance(action, dict):
            action = jnp.concatenate([action[key] for key in sorted(action.keys())])
        
        observations.append(obs)
        actions.append(action)

    observations = jnp.array(observations)
    actions = jnp.array(actions)

    # Training step
    @jax.jit
    def train_step(params, opt_state, batch_obs, batch_actions):
        def loss_fn(params):
            pred_actions = distillation_network.apply(params, batch_obs)
            return jnp.mean((pred_actions - batch_actions) ** 2)

        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Training loop
    num_batches = len(observations) // batch_size
    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_obs = observations[i * batch_size:(i + 1) * batch_size]
            batch_actions = actions[i * batch_size:(i + 1) * batch_size]
            params, opt_state = train_step(params, opt_state, batch_obs, batch_actions)

    return distillation_network

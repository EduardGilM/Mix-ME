import jax
import jax.numpy as jnp
from jax.random import PRNGKey as KeyArray
from qdax.core.neuroevolution.networks.networks import MLP
from typing import List, Tuple
import optax

def distill_knowledge(
    transitions: List[dict],
    observation_size: int = 376,
    action_size: int = 17,
    hidden_layer_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    random_key: KeyArray = jax.random.PRNGKey(0)
) -> Tuple[MLP, dict]:
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
        
        # Ensure the observation has the correct size by padding with zeros if necessary
        obs = jnp.pad(obs, (0, max(0, observation_size - obs.size)), constant_values=0)
        # Ensure the action has the correct size by padding with zeros if necessary
        action = jnp.pad(action, (0, max(0, action_size - action.size)), constant_values=0)
        
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
    num_batches = len(transitions)
    for epoch in range(num_epochs):
        for i in range(num_batches):
            # Use observations and actions from a single transition
            batch_obs = observations[i:i + 1]
            batch_actions = actions[i:i + 1]
            params, opt_state = train_step(params, opt_state, batch_obs, batch_actions)

    return distillation_network, params

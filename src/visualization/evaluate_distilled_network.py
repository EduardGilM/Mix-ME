import jax.numpy as jnp
import jax
import wandb
from brax.v1.io import html
from brax.v1.io.file import File
from qdax import environments

def evaluate_distilled_network(config, distilled_network):
    # Create a new environment instance for evaluation
    env = environments.create(config['env_name'], episode_length=config['episode_length'])

    # Initialize the parameters
    sample_input = jnp.zeros((1, env.observation_size))
    params = distilled_network.init(jax.random.PRNGKey(config['seed']), sample_input)

    # JIT compile the functions
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(distilled_network.apply)

    # Run the distilled network in the environment
    rollout = []
    rng = jax.random.PRNGKey(seed=config['seed'])
    state = jit_env_reset(rng)
    total_reward = 0.0

    while not state.done:
        rollout.append(state)
        action = jit_inference_fn(params, state.obs)
        state = jit_env_step(state, action)
        total_reward += state.reward

    print(f"Total reward obtained by the distilled network: {total_reward:.2f}")
    print(f"The trajectory of this individual contains {len(rollout)} transitions.")

    # Save the trajectory as an HTML file
    frames = html.render(env.sys, [s.qp for s in rollout[:500]])
    with File("trajectories/distilled_network_trajectory.html", "w") as f:
        f.write(frames)

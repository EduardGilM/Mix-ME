import jax.numpy as jnp
import jax
import wandb
from brax.v1.io import html
from brax.v1.io.file import File
from qdax import environments

def evaluate_distilled_network(config, distilled_network, pretrained_params):
    # Create a new environment instance for evaluation
    env = environments.create(config['env_name'], episode_length=config['episode_length'])

    # Initialize the parameters (use pretrained parameters)
    params = pretrained_params

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
        # Ensure the observation has the correct size by padding with zeros if necessary
        obs = jnp.pad(state.obs, (0, max(0, 376 - state.obs.size)), constant_values=0)
        action = jit_inference_fn(params, obs)
        state = jit_env_step(state, action)
        total_reward += state.reward

    print(f"Total reward obtained by the distilled network: {total_reward:.2f}")
    print(f"The trajectory of this individual contains {len(rollout)} transitions.")

    # Save the trajectory as an HTML file
    frames = html.render(env.sys, [s.qp for s in rollout[:500]])
    with File("trajectories/distilled_network_trajectory.html", "w") as f:
        f.write(frames)

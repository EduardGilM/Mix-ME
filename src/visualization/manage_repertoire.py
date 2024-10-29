import jax.numpy as jnp
import jax
import wandb
from brax.v1.io import html
from brax.v1.io.file import File
def manage_repertoire(config, repertoire, env, policy_network):
    if config['parameter_sharing'] or not config['multiagent']:
        best_idx = jnp.argmax(repertoire.fitnesses)
        best_fitness = jnp.max(repertoire.fitnesses)
        best_bd = repertoire.descriptors[best_idx]

        print(
            f"Best fitness in the repertoire: {best_fitness:.2f}\n"
            f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n"
            f"Index in the repertoire of this individual: {best_idx}\n"
        )
    else:
        best_idx = jnp.argmax(repertoire.fitnesses)
        best_fitness = jnp.max(repertoire.fitnesses)
        best_bd = repertoire.descriptors[best_idx]

        if config['low_spread']:
            best_spread = repertoire.spreads[best_idx]

            print(
                f"Best fitness in the repertoire: {best_fitness:.2f}\n"
                f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n"
                f"Spread of the best individual in the repertoire: {best_spread}\n"
                f"Index in the repertoire of this individual: {best_idx}\n"
            )
        else:
            print(
                f"Best fitness in the repertoire: {best_fitness:.2f}\n"
                f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n"
                f"Index in the repertoire of this individual: {best_idx}\n"
            )

    def get_index_of_closest_bd(repertoire, bd):
        #Convert bd to a jax.numpy array
        bd = jnp.array(bd)
        distances = jnp.linalg.norm(repertoire.descriptors - bd, axis=1)
        return jnp.argmin(distances)

    # Get the closest behavior descriptor in the repertoire
    closest_idx = get_index_of_closest_bd(repertoire, [config['min_bd'], config['max_bd']])

    my_params = jax.tree_util.tree_map(
        lambda x: x[best_idx],
        repertoire.genotypes
    )

    if config['multiagent']:
        if config['parameter_sharing']:
          # PARAMETER SHARING == TRUE

          jit_env_reset = jax.jit(env.reset)
          jit_env_step = jax.jit(env.step)
          jit_inference_fn = jax.jit(policy_network.apply)

          rollout = []
          rng = jax.random.PRNGKey(seed=config['seed'])
          state = jit_env_reset(rng)
          while not state.done:
            rollout.append(state)
          # Apply the policy to get the action
            action = jit_inference_fn(my_params, state.obs)
          # Check if the action is a scalar, and if so, convert it to an array
            if jnp.isscalar(action):
                action = jnp.array([action])

          # Assuming env.step expects a dictionary of actions for multi-agent settings
          # Get the correct number of agents from the environment
            num_agents = 2

          #Provide actions for all agents.
            agent_actions = {i: action for i in range(num_agents)}

            state = jit_env_step(state, agent_actions)
          
          print(f"This trajectory is MULTI-AGENT - PARAMETER SHARING == TRUE")
          print(f"The trajectory of this individual contains {len(rollout)} transitions.")

        else:
            # PARAMETER SHARING == FALSE

            jit_env_reset = jax.jit(env.reset)
            jit_env_step = jax.jit(env.step)
            jit_inference_fns = {
                agent_id: jax.jit(policy.apply)
                for agent_id, policy in policy_network.items()
            }

            rollout = []
            rng = jax.random.PRNGKey(seed=config['seed'])
            state = jit_env_reset(rng)

            while not state.done:
                rollout.append(state)

             # Get observations for all agents (assuming state.obs is a dictionary)
                obs = env.obs(state)

            # Get actions for each agent using zip
                agent_actions = {
                    agent_idx: jit_inference_fns[agent_idx](my_params[agent_idx], agent_obs)
                    for (agent_idx, _), agent_obs in zip(policy_network.items(), obs.values())
              }
            # Avanzar al siguiente estado
                state = jit_env_step(state, agent_actions)
            
            print(f"This trajectory is MULTI-AGENT - PARAMETER SHARING == FALSE {config['emitter_type']}")
            print(f"La trayectoria de este individuo contiene {len(rollout)} transiciones.")
    else:
      # MONO AGENTE

      jit_env_reset = jax.jit(env.reset)
      jit_env_step = jax.jit(env.step)
      jit_inference_fn = jax.jit(policy_network.apply)

      rollout = []
      rng = jax.random.PRNGKey(seed=config['seed'])
      state = jit_env_reset(rng=rng)
      while not state.done:
          rollout.append(state)
          action = jit_inference_fn(my_params, state.obs)
          state = jit_env_step(state, action)

      print(f"This trajectory is INDIVIDUAL AGENT")
      print(f"The trajectory of this individual contains {len(rollout)} transitions.")

    frames = html.render(env.sys, [s.qp for s in rollout[:500]])
    with File("trajectories/last_trajectory.html", "w") as f:
        f.write(frames)
    
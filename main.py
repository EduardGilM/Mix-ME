import jax
import os
from src.utils.arguments import (
    load_config,
    parse_arguments,
    merge_configs,
    check_config,
)
from src.utils.logging import init_wandb
from qdax.utils.plotting import (
    plot_map_elites_results,
    plot_multidimensional_map_elites_grid
)
from src.visualization.visualize import plot_2d_map
from src.visualization.manage_repertoire import manage_repertoire
from src.training.map_elites import (
    prepare_map_elites_multiagent,
    prepare_map_elites,
    run_training,
    evaluate_adaptation,
)
import wandb
import jax.numpy as jnp
from src.training.distillation import distill_knowledge, save_distilled_network, load_distilled_network
from src.visualization.evaluate_distilled_network import evaluate_distilled_network

def main():
    args = parse_arguments()

    # Load the configuration file and merge it with the CLI arguments
    config = load_config("config.yaml.default")
    config = merge_configs(config, args)

    # Check the configuration
    check_config(config)

    # Initialise WandB
    init_wandb(config)

    dispositivos = jax.devices()
    print(dispositivos)

    config = wandb.config

    if config["disable_jit"]:
        jax.config.update("jax_disable_jit", True)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    if config["multiagent"]:
        preparation_fun = prepare_map_elites_multiagent
    else:
        preparation_fun = prepare_map_elites

    # Init the MAP-Elites algorithm for multi agent
    map_elites, repertoire, emitter_state, random_key, env, policy_network = preparation_fun(
        random_key=random_key, **config
    )

    # Run the training
    repertoire, emitter_state, random_key, all_metrics = run_training(
        map_elites, repertoire, emitter_state, random_key=random_key, **config
    )

    transition_list = manage_repertoire(config, repertoire, env, policy_network)

    print(f"Number of transitions collected: {len(transition_list)}")

    # Distill knowledge from the transitions
    total_transitions = len(transition_list)
    max_transitions = int(total_transitions)
    sampled_transitions = transition_list[:max_transitions]
    distilled_network, params, network_config = distill_knowledge(sampled_transitions, random_key=random_key)

    # Save the distilled network and its parameters
    #save_distilled_network(params, network_config, "distilled_network_params.npy", "distilled_network_config.json")

    # Evaluate the performance of the distilled network in a new environment
    evaluate_distilled_network(config, distilled_network, params)

    # Load the distilled network and its parameters
    #loaded_network, loaded_params = load_distilled_network("distilled_network_params.npy", "distilled_network_config.json")

    # Evaluate the performance of the loaded network in a new environment
    evaluate_distilled_network(config, distilled_network, params)

    if config['env_name'] == 'ant_uni':
      fig, _ = plot_multidimensional_map_elites_grid(
      repertoire=repertoire,
      minval=jnp.array([config['min_bd']]),
      maxval=jnp.array([config['max_bd']]),
      grid_shape=(6,6,6,6),
      )
      wandb.log({"2d_map": wandb.Image(fig)})
    else:
      plot_2d_map(
        repertoire=repertoire,
        **config,
      )

    # create the x-axis array
    env_steps = jnp.arange(config['num_iterations']) * config['episode_length'] * config['batch_size']

    fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=config['min_bd'], max_bd=config['max_bd'])
    wandb.log({"map_elites_results": wandb.Image(fig)})

    # Save the repertoire
    if config.get("save_repertoire"):
        repertoire_path = os.path.join(
            config["output_dir"],
            "saved_repertoires",
            wandb.run.name + "-" + wandb.run.id + "/",
        )
        os.makedirs(repertoire_path, exist_ok=True)
        repertoire.save(path=repertoire_path)

    # Evaluate adaptation
    if config.get("adaptation"):
        evaluate_adaptation(
            repertoire=repertoire,
            random_key=random_key,
            **config,
        )


if __name__ == "__main__":
    # Set the current working directory to the directory of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

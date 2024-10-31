import functools
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jax.random import PRNGKey as KeyArray
from brax.envs import Env
from qdax.core.mels import MELS
from qdax.core.emitters.emitter import EmitterState
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.containers.mels_repertoire import MELSRepertoire
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax import environments
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    scoring_function_brax_envs as scoring_function,
    reset_based_scoring_function_brax_envs
)
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.core.emitters.ma_standard_emitters import (
    NaiveMultiAgentMixingEmitter,
    MultiAgentEmitter,
)
from qdax.custom_types import (
    EnvState,
    Params,
    RNGKey,
)

from ..utils.generalisation_constants import ADAPTATION_CONSTANTS
from ..visualization.visualize import plot_adaptation_fitness

from qdax.environments.multi_agent_wrappers import MultiAgentBraxWrapper

from qdax.utils.metrics import default_qd_metrics

import wandb

def init_multiple_policy_networks(
    env: MultiAgentBraxWrapper,
    policy_hidden_layer_size: int,
) -> dict[int, MLP]:
    action_sizes = env.get_action_sizes()

    policy_networks = {
        agent_idx: MLP(
            layer_sizes=(policy_hidden_layer_size, policy_hidden_layer_size)
            + (action_size,),
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
        for agent_idx, action_size in action_sizes.items()
    }
    return policy_networks


def init_policy_network(
    policy_hidden_layer_size: int,
    action_size: int,
) -> MLP:
    layer_sizes = (policy_hidden_layer_size, policy_hidden_layer_size) + (action_size,)
    policy_network = MLP(
        layer_sizes=layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    return policy_network


def init_controller_population_multiple_networks(
    env: MultiAgentBraxWrapper,
    policy_networks: dict[int, MLP],
    batch_size: int,
    random_key: KeyArray,
):
    global num_agents
    num_agents = len(policy_networks)
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_agents * batch_size)

    init_variables = []
    for (agent_idx, agent_policy), agent_keys in zip(
        policy_networks.items(), jnp.split(keys, num_agents, axis=0)
    ):
        fake_batch = jnp.zeros(shape=(batch_size, env.get_obs_sizes()[agent_idx]))
        init_variables.append(jax.vmap(agent_policy.init)(agent_keys, fake_batch))

    return init_variables


def init_controller_population_single_network(
    policy_network: MLP,
    batch_size: int,
    observation_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    return init_variables


def init_environment_states(
    env: Env,
    batch_size: int,
    random_key: KeyArray,
):
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)
    return init_states

def make_policy_network_play_step_fn(
    env: MultiAgentBraxWrapper,
    policy_network: dict[int, MLP] | MLP,
    parameter_sharing: bool,
) -> Callable[[EnvState, Params, RNGKey], tuple[EnvState, Params, RNGKey, QDTransition]]:

    def play_step_fn(env_state: EnvState, policy_params: list[Params] | Params, random_key: KeyArray):
        obs = env.obs(env_state)
        if not parameter_sharing:
            agent_actions = {
                agent_idx: network.apply(params, agent_obs)
                for (agent_idx, network), params, agent_obs in zip(
                    policy_network.items(), policy_params, obs.values()
                )
            }
        else:
            agent_actions = {
                agent_idx: policy_network.apply(policy_params, agent_obs)
                for agent_idx, agent_obs in obs.items()
            }

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, agent_actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=agent_actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    return play_step_fn

def prepare_map_elites_multiagent(
    env_name: str,
    low_spread: bool,
    batch_size: int,
    episode_length: int,
    policy_hidden_layer_size: int,
    parameter_sharing: bool,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    k_mutations: int,
    emitter_type: str,
    homogenisation_method: str,
    eta: float,
    mut_val_bound: float,
    proportion_to_mutate: float,
    variation_percentage: float,
    crossplay_percentage: float,
    random_key: KeyArray,
    num_samples: float,
    **kwargs,
):
    # Create environment
    global env
    base_env_name = env_name.split("_")[0]
    env = environments.create(env_name, episode_length=episode_length)
    env = MultiAgentBraxWrapper(
        env,
        env_name=base_env_name,
        parameter_sharing=parameter_sharing,
        emitter_type=emitter_type,
        homogenisation_method=homogenisation_method,
    )
    num_agents = len(env.get_action_sizes())
    global policy_network

    # Init policy network/s
    if parameter_sharing:
        if homogenisation_method == "concat":
            policy_network = init_policy_network(
                policy_hidden_layer_size, env.action_size
            )
        else:
            policy_network = init_policy_network(
                policy_hidden_layer_size, env.get_action_sizes()[0]
            )
        init_variables = init_controller_population_multiple_networks(
            env, {0: policy_network}, batch_size, random_key
        )[0]
    else:
        policy_network = {
            agent_idx: init_policy_network(policy_hidden_layer_size, action_size)
            for agent_idx, action_size in env.get_action_sizes().items()
        }
        init_variables = init_controller_population_multiple_networks(
            env, policy_network, batch_size, random_key
        )

    # Create the initial environment states
    init_states = init_environment_states(env, batch_size, random_key)

    # Create the play step function
    play_step_fn = make_policy_network_play_step_fn(
        env, policy_network, parameter_sharing
    )

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

    if low_spread:
      scoring_fn = functools.partial(
          reset_based_scoring_function_brax_envs,
          episode_length=episode_length,
          play_reset_fn=env.reset,
          play_step_fn=play_step_fn,
          behavior_descriptor_extractor=bd_extraction_fn,
      )
    else:
      scoring_fn = functools.partial(
          scoring_function,
          init_states=init_states,
          episode_length=episode_length,
          play_step_fn=play_step_fn,
          behavior_descriptor_extractor=bd_extraction_fn,
      )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mutation_function = functools.partial(
        polynomial_mutation,
        eta=eta,
        minval=-mut_val_bound,
        maxval=mut_val_bound,
        proportion_to_mutate=proportion_to_mutate,
    )

    if parameter_sharing:
        emitter = MixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
        )
    elif emitter_type == "naive":
        emitter = NaiveMultiAgentMixingEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            agents_to_mutate=k_mutations,
        )
    else:
        emitter = MultiAgentEmitter(
            mutation_fn=mutation_function,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            crossplay_percentage=crossplay_percentage,
            batch_size=batch_size,
            num_agents=num_agents,
            role_preserving=emitter_type == "role_preserving",
            agents_to_mutate=k_mutations,
        )

    # Instantiate MAP-Elites

    if low_spread:
      map_elites = MELS(
          scoring_function=scoring_fn,
          emitter=emitter,
          metrics_function=metrics_function,
          num_samples=num_samples,
      )
    else:
      map_elites = MAPElites(
          scoring_function=scoring_fn,
          emitter=emitter,
          metrics_function=metrics_function,
          #qd_offset=reward_offset,
      )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    return map_elites, repertoire, emitter_state, random_key, env, policy_network


def prepare_map_elites(
    env_name: str,
    batch_size: int,
    sample_size: int,
    episode_length: int,
    policy_hidden_layer_size: int,
    iso_sigma: float,
    line_sigma: float,
    num_init_cvt_samples: int,
    num_centroids: int,
    min_bd: float,
    max_bd: float,
    eta: float,
    mut_val_bound: float,
    proportion_to_mutate: float,
    variation_percentage: float,
    random_key: KeyArray,
    **kwargs,
):
    global env
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    action_size = (
        env.action_size if env_name != "hanabi" else env.action_space(env.agents[0]).n
    )
    global policy_network
    policy_network = init_policy_network(policy_hidden_layer_size, action_size)

    # Init population of controllers
    observation_size = (
        env.observation_size
        if env_name != "hanabi"
        else env.observation_space(env.agents[0]).n
    )
    init_variables = init_controller_population_single_network(
        policy_network, batch_size, observation_size, random_key
    )

    # Create the initial environment states
    init_states = init_environment_states(env, batch_size, random_key)

    play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        init_states=init_states,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name] if env_name != "hanabi" else 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )
    mutation_function = functools.partial(
        polynomial_mutation,
        eta=eta,
        minval=-mut_val_bound,
        maxval=mut_val_bound,
        proportion_to_mutate=proportion_to_mutate,
    )

    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_function,
        variation_fn=variation_fn,
        variation_percentage=variation_percentage,
        batch_size=batch_size,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        #qd_offset=reward_offset,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length if env_name != "hanabi" else 2,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    return map_elites, repertoire, emitter_state, random_key, env, policy_network


def run_training(
    map_elites: MAPElites,
    repertoire: MapElitesRepertoire,
    emitter_state: EmitterState,
    num_iterations: int,
    log_period: int,
    random_key: KeyArray,
    **kwargs,
):
    num_loops = int(num_iterations / log_period)

    # Prepare the logger
    global all_metrics
    all_metrics = {}

    # main
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
        start_time = time.time()
        (
            (repertoire, emitter_state, random_key),
            metrics
        ) = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
        logged_metrics = {
            "time": timelapse,
        }
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]
            
            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        wandb.log(logged_metrics, step=1 + i * log_period)

    return repertoire, emitter_state, random_key, all_metrics


def evaluate_adaptation(
    repertoire: MapElitesRepertoire,
    adaptation_eval_num: int,
    env_name: str,
    episode_length: int,
    policy_hidden_layer_size: int,
    parameter_sharing: bool,
    emitter_type: str,
    homogenisation_method: str,
    multiagent: bool,
    random_key: KeyArray,
    **kwargs,
):
    base_env_name = env_name.split("_")[0]

    fitnesses = []

    for adaptation_name in ADAPTATION_CONSTANTS:
        adaptation_constants = ADAPTATION_CONSTANTS[adaptation_name]
        adaptation_constants_env = adaptation_constants[env_name]

        for adaptation_idx in range(10):
            env_kwargs = {}
            env_kwargs[adaptation_name] = jax.tree.map(
                lambda x: x[adaptation_idx], adaptation_constants_env
            )

            eval_env = environments.create(
                env_name=env_name,
                batch_size=None,
                episode_length=episode_length,
                auto_reset=True,
                eval_metrics=True,
                **env_kwargs,
            )
            if multiagent:
                eval_env = MultiAgentBraxWrapper(
                    eval_env,
                    env_name=base_env_name,
                    parameter_sharing=parameter_sharing,
                    emitter_type=emitter_type,
                    homogenisation_method=homogenisation_method,
                )
                policy_network = init_policy_network(
                  policy_hidden_layer_size, eval_env.action_size
            )
                play_step_fn = make_policy_network_play_step_fn(
                    eval_env, policy_network, parameter_sharing
                )
            else:
                policy_network = init_policy_network(
                    policy_hidden_layer_size, eval_env.action_size
                )
                play_step_fn = make_policy_network_play_step_fn_brax(
                    eval_env, policy_network
                )

            bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
            scoring_fn = functools.partial(
                scoring_function,
                episode_length=episode_length,
                play_step_fn=play_step_fn,
                behavior_descriptor_extractor=bd_extraction_fn,
            )

            scoring_fn = jax.jit(scoring_fn)
            reset_fn = jax.jit(jax.vmap(eval_env.reset))

            # Extract the policies
            policies = jax.tree.map(
                lambda x: x[repertoire.fitnesses != -jnp.inf], repertoire.genotypes
            )
            num_policies = jax.tree_util.tree_leaves(policies)[0].shape[0]

            # Define a helper function to evaluate policies
            @jax.jit
            def evaluate_policies_helper(random_key):
                keys = jax.random.split(random_key, num=num_policies)
                init_states = reset_fn(keys)
                eval_fitnesses, descriptors, extra_scores, random_key = scoring_fn(
                    policies, random_key, init_states
                )
                return eval_fitnesses

            # Generate random keys for each evaluation
            random_key, subkey = jax.random.split(random_key)
            keys = jax.random.split(subkey, num=adaptation_eval_num)

            # Parallelize the evaluation
            eval_fitnesses = jax.vmap(evaluate_policies_helper)(keys)

            # Compute the median fitness for each policy over its states
            median_fitnesses = jnp.median(eval_fitnesses, axis=0)

            # Report the highest median fitness
            fitnesses.append(
                (adaptation_name, adaptation_idx, jnp.max(median_fitnesses).item())
            )

    global table
    table = wandb.Table(
        columns=["adaptation_name", "adaptation_idx", "adaptation_fitness"],
        data=fitnesses,
    )
    wandb.log({"adaptation_fitness": table})
    plot_adaptation_fitness(table)

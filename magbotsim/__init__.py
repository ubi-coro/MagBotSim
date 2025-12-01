__version__ = '1.0.2a1'

from gymnasium.envs.registration import register

from magbotsim.basic_magbot_env import BasicMagBotEnv
from magbotsim.rl_envs.basic_multi_agent_env import BasicMagBotMultiAgentEnv
from magbotsim.rl_envs.basic_single_agent_env import BasicMagBotSingleAgentEnv
from magbotsim.utils.benchmark_utils import BENCHMARK_PLANNING_LAYOUTS
from magbotsim.utils.impedance_control import MoverImpedanceController
from magbotsim.utils.rendering import Matplotlib2DViewer, MujocoViewerCollection

__all__ = [
    'BasicMagBotEnv',
    'BasicMagBotMultiAgentEnv',
    'BasicMagBotSingleAgentEnv',
    'Matplotlib2DViewer',
    'MoverImpedanceController',
    'MujocoViewerCollection',
]


def register_gymnasium_envs():
    #######################
    # Trajectory Planning #
    #######################
    register(
        id='LongHorizonGlobalTrajectoryPlanningEnv-v0',
        entry_point='magbotsim.rl_envs.trajectory_planning.long_horizon_global_trajectory_planning_env:LongHorizonGlobalTrajectoryPlanningEnv',
    )

    # Trajectory Planning Benchmarks
    for i in range(len(BENCHMARK_PLANNING_LAYOUTS)):
        register(
            id=f'LongHorizonGlobalTrajectoryPlanningEnvB{i}-v0',
            entry_point=f'magbotsim.rl_envs.trajectory_planning.long_horizon_global_trajectory_planning_env:LongHorizonGlobalTrajectoryPlanningEnvB{i}',
        )

    #######################
    # Object Manipulation #
    #######################
    register(
        id='StateBasedStaticObstaclePushingEnv-v0',
        entry_point='magbotsim.rl_envs.object_manipulation.pushing.state_based_static_obstacle_env:StateBasedStaticObstaclePushingEnv',
        max_episode_steps=25,
    )

    register(
        id='StateBasedGlobalPushingEnv-v0',
        entry_point='magbotsim.rl_envs.object_manipulation.pushing.state_based_global_pushing_env:StateBasedGlobalPushingEnv',
        max_episode_steps=25,
    )

    register(
        id='StateBasedPushBoxEnv-v0',
        entry_point='magbotsim.rl_envs.object_manipulation.pushing.state_based_push_box_env:StateBasedPushBoxEnv',
        max_episode_steps=25,
    )

    # Push Box Benchmarks
    for i in range(len(BENCHMARK_PLANNING_LAYOUTS)):
        register(
            id=f'StateBasedPushBoxEnvB{i}-v0',
            entry_point=f'magbotsim.rl_envs.object_manipulation.pushing.state_based_push_box_env:StateBasedPushBoxEnvB{i}',
        )

    # Letter-Pushing Envs
    for letter in ['t', 'x', 'l']:
        env_name = f'StateBasedPush{letter.upper()}Env'
        path = f'magbotsim.rl_envs.object_manipulation.pushing.state_based_push_{letter}_env'

        # Base Environment
        register(
            id=f'{env_name}-v0',
            entry_point=f'{path}:{env_name}',
            max_episode_steps=50,
        )

        # Benchmark Environments
        for i in range(len(BENCHMARK_PLANNING_LAYOUTS)):
            register(
                id=f'{env_name}B{i}-v0',
                entry_point=f'{path}:{env_name}B{i}',
            )


register_gymnasium_envs()

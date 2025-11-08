.. _lh_global_trajectory_planning_env:

Long-Horizon Global Trajectory Planning Environment
===================================================

.. raw:: html

   <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">

      <iframe width="320" height="180"
            src="https://www.youtube.com/embed/pJvkNNimoTo?autoplay=1&mute=1&loop=1&playlist=pJvkNNimoTo"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

      <iframe width="320" height="180"
            src="https://www.youtube.com/embed/o_LCuxNir2w?autoplay=1&mute=1&loop=1&playlist=o_LCuxNir2w"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

   </div>

The ``LongHorizonGlobalTrajectoryPlanningEnv`` is a simple trajectory planning
environment that should be understood as an example of how trajectory planning with
MagLev systems can look like. This environment is therefore intended for
parameter or algorithm tests.

The aim is to learn to control multiple movers that start from random (x,y)
positions and continuously receive new goal positions as they complete their
current tasks. When a mover reaches its goal, it is immediately assigned a new
goal position, allowing it to continue working without waiting for other movers.
The environment runs indefinitely until either a collision occurs or no mover
reaches any goal within a specified timeout period. A collision is detected,
if any two movers collide or if at least one mover leaves the tiles (think of
this as a collision with a wall). Control is achieved by specifying either the
jerk or the acceleration. In this environment, positions, velocities, accelerations,
and jerks have the units m, m/s, m/s² and m/s³, respectively.

Observation Space
-----------------

The observation space of this environment is a dictionary containing the following keys and values:

================ =============================================================================================================
Key              Value
================ =============================================================================================================
observation      - if ``learn_jerk=True``:
                   a numpy array of shape (num_movers*2*2,) containing the (x,y)-velocities and (x,y)-accelerations of
                   each mover ((x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...)
                 - if ``learn_jerk=False``:
                   a numpy array of shape (num_movers*2,) containing the (x,y)-velocities and of each mover
                   ((x,y)-velo mover 1, (x,y)-velo mover 2, ...)
 achieved_goal   a numpy array of shape (num_movers*2,) containing the current (x,y)-positions of all movers w.r.t. the
                 frame ((x,y)-pos mover 1, (x,y)-pos mover 2, ...)
 desired_goal    a numpy array of shape (num_movers*2,) containing the desired (x,y)-positions of all movers w.r.t the
                 base frame ((x,y) goal pos mover 1, (x,y) goal pos mover 2, ...)
================ =============================================================================================================

Action Space
------------

The action space is continuous. If ``learn_jerk=True``, an action

.. math::
    a_j := [j_{1x}, j_{1y}, ..., j_{nx}, j_{ny}]^T

represents the desired jerks for each mover in x and y direction of the base frame (unit: m/s³), where

.. math::
    j_{1x}, j_{1y}, ..., j_{nx}, j_{ny} \in [-j_{max},j_{max}]

``j_max`` is the maximum possible jerk (see environment parameters) and n denotes the number of movers.

Accordingly, if ``learn_jerk=False``, an action

.. math::
    a_a := [a_{1x}, a_{1y}, ..., a_{nx}, a_{ny}]^T

represents the accelerations for each mover in x and y direction of the base frame (unit: m/s²), where

.. math::
    a_{1x}, a_{1y}, ..., a_{nx}, a_{ny} \in [-a_{max},a_{max}]

``a_max`` is the maximum possible acceleration (see environment parameters) and n denotes the number of movers.

Immediate Rewards
-----------------

The agent receives a reward similar to number of movers that reached their goal positions without 
collisions within this timestep. In case of a collision, the agent receives a reward of -10. 
If in a timestep no mover has reached its goal and no collisions have been detected, a reward of -1 is given.

Episode Termination and Truncation
----------------------------------

Episodes are designed to run indefinitely with goals being continuously
resampled. An episode only terminates under two conditions:

1. **Collision**:
    If there is a collision, the episode terminates immediately.

2. **Goal timeout**:
    If no mover reaches any goal within a specified time limit (defined by the ``timeout_steps`` parameter, which defaults to 50 steps), the episode terminates.
    The timeout counter resets each time any mover reaches a goal.

When any individual mover reaches its current goal, a new goal is automatically
generated for that specific mover and the episode continues indefinitely, unless
one of the termination conditions above is met.

Environment Reset
-----------------

When the environment is reset, all movers are randomly positioned within the
defined workspace boundaries, and their corresponding goal positions are also
randomly sampled within the same workspace.

Basic Usage
-----------
The following example shows how to train an agent using
`Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_.
To use the example, please install Stable-Baselines3 as described in the
`documentation <https://stable-baselines3.readthedocs.io/en/master/guide/install.html>`_.

.. note::
    This is a simplified example that is not guaranteed to converge, as the default parameters are used. However, it is important to note that
    the parameter ``copy_info_dict`` is set to ``True``. This way, it is not necessary to check for collision again to compute the reward when a
    transition is relabeled by HER, since the information is already available in the ``info``-dict.


.. code-block:: python

    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import SAC, HerReplayBuffer
    import magbotsim

    render_mode = None
    mover_params = {'size': np.array([0.113 / 2, 0.113 / 2, 0.012 / 2]), 'mass': 0.628}
    collision_params = {'shape': 'box', 'size': np.array([0.113 / 2 + 1e-6, 0.113 / 2 + 1e-6]), 'offset': 0.0, 'offset_wall': 0.0}
    env_params = {
        'layout_tiles_list': [np.ones((5,5))],
        'num_movers': 5,
        'show_2D_plot': False,
        'mover_params': mover_params,
        'collision_params': collision_params,
        'render_mode': render_mode
    }

    env = gym.make('LongHorizonGlobalTrajectoryPlanningEnv-v0', **env_params)
    # copy_info_dict=True, as information about collisions is stored in the info dictionary to avoid
    # computationally expensive collision checking calculations when the data is relabeled (HER)
    model = SAC(
        policy='MultiInputPolicy',
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={'copy_info_dict': True},
        verbose=1
    )
    model.learn(total_timesteps=int(1e6))

Version History
---------------
- v0: initial version of the environment

Parameters
----------

.. automodule:: magbotsim.rl_envs.trajectory_planning.long_horizon_global_trajectory_planning_env
  :members:
  :no-index:
  :show-inheritance:

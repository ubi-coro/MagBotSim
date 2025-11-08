Restoring the MuJoCo State
==========================

Since MagBotSim environments are based on the MuJoCo physics engine, you can
use MuJoCo's :func:`mj_setState` and :func:`mj_getState` to save and restore
the complete simulation state. This can be useful to reset an environment
to a specific timestep, e.g. to compare different trajectory planning algorithms
starting from a specific configuration.

Basic Usage
-----------

The following example shows how to save the current simulation state and restore
it later:

.. code-block:: python

    import gymnasium as gym
    import mujoco
    import numpy as np
    import magbotsim
    gym.register_envs(magbotsim)

    env = gym.make(
        'LongHorizonGlobalTrajectoryPlanningEnv-v0',
        layout_tiles=np.ones((4, 3)),
        mover_params={
            'mass': 0.639 - 0.034,
            'shape': 'mesh',
            'mesh': {
                'mover_stl_path': 'beckhoff_apm4220_mover',
                'bumper_stl_path': 'beckhoff_apm4220_bumper',
                'bumper_mass': 0.034,
            },
        },
        num_movers=1,
        show_2D_plot=False,
        render_mode='rgb_array',
    )
    env.reset()

    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(env.unwrapped.model, state_type)

    saved_state = np.zeros(state_size)
    mujoco.mj_getState(
        env.unwrapped.model,
        env.unwrapped.data,
        saved_state,
        state_type
    )

    # Run some simulation steps
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)

    # Restore the saved state
    mujoco.mj_setState(
        env.unwrapped.model,
        env.unwrapped.data,
        saved_state,
        state_type,
    )

    # Forward the simulation to ensure consistency
    mujoco.mj_forward(
        env.unwrapped.model,
        env.unwrapped.data,
    )

Verifying the Restored State
----------------------------

You can verify that the restored state matches the original by re-reading it
and comparing arrays:

.. code-block:: python

    before = saved_state.copy()

    # Run and restore
    for _ in range(5):
        env.step(env.action_space.sample())

    mujoco.mj_setState(env.unwrapped.model, env.unwrapped.data, before, state_type)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

    after_restore = np.zeros_like(before)
    mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, after_restore, state_type)

    print("States identical after restore:", np.allclose(before, after_restore))

Choosing a State Type
---------------------

MuJoCo supports multiple state types, such as ``mjSTATE_INTEGRATION`` and
``mjSTATE_FULLPHYSICS``, which determine the level of detail stored (positions,
velocities, etc.).

See the `MuJoCo documentation <https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate>`_
for details.

Important Notes
---------------

* Always call :func:`mj_forward()` after :func:`mj_setState()` to ensure simulation consistency
* State arrays are tied to specific model configurations. Changing the model invalidates saved states
* For deterministic replay, ensure any random number generators are also reset appropriately
* Consider memory usage when saving many states in long-running simulations

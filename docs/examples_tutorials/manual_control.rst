.. _manual_mover_control:

Manual Mover Control
====================

MagBotSim environments can be manually controlled using Gymnasium's ``play``
utility function. This is useful for testing, debugging, and getting a feel for
the physics of the environment. For more details about the play function and its parameters, see the
`Gymnasium documentation <https://gymnasium.farama.org/api/utils/#gymnasium.utils.play.play>`_.

.. note::
    Gymnasium's ``play`` utility function can only be used, if the environment follows the Gymnasium API.
    In order to use a custom MagBotSim environment with this utility, the :ref:`basic_magbot_single_agent_env`
    should be used as a parent class for the custom environment. This ensures that the custom environment
    follows the correct API.

Requirements
------------

Manual control requires pygame for keyboard input handling. You can install it either by:

- Installing MagBotSim with the optional extra: ``pip install magbotsim[manual_control]``
- Installing Gymnasium with the optional extra: ``pip install gymnasium[classic-control]``
- Installing pygame directly: ``pip install pygame``

.. note::
    Due to a bug with MuJoCo and Pygame on Linux, manual control currently does not work on Linux systems.
    See the tracking issue: https://github.com/Farama-Foundation/Gymnasium/issues/920

Basic Usage
-----------

Here is a complete example showing how to set up manual control for a single mover:

.. code-block:: python

    import numpy as np
    import gymnasium as gym
    from gymnasium.utils.play import play
    import magbotsim

    m = 0.4

    play(
        gym.make(
            'LongHorizonGlobalTrajectoryPlanningEnv-v0',
            layout_tiles=np.ones((4, 3)),
            mover_params={
                'size': np.array([0.113 / 2, 0.113 / 2, 0.012 / 2]),
                'mass': 0.63 - 0.1,
                'shape': 'mesh',
                'mesh': {
                    'mover_stl_path': 'beckhoff_apm4220_mover',
                    'bumper_stl_path': 'beckhoff_apm4220_bumper',
                    'bumper_mass': 0.1,
                },
            },
            num_movers=1,
            show_2D_plot=False,
            mover_colors_2D_plot=['red'],
            render_mode='rgb_array',
        ),
        keys_to_action={
            'w': np.array([0, m], dtype=np.float32),
            'a': np.array([-m, 0], dtype=np.float32),
            's': np.array([0, -m], dtype=np.float32),
            'd': np.array([m, 0], dtype=np.float32),
            'wa': np.array([-m, m], dtype=np.float32),
            'dw': np.array([m, m], dtype=np.float32),
            'ds': np.array([m, -m], dtype=np.float32),
            'as': np.array([-m, -m], dtype=np.float32),
        },
        noop=np.array([0, 0], dtype=np.float32),
    )

Controls
--------

The example above implements the following keyboard controls:

- **W, A, S, D**: Basic directional movement (up, left, down, right)
- **WA, DW, DS, AS**: Diagonal movement combinations
- **No key pressed**: No movement (noop action)

The parameter `m = 0.4` controls the magnitude of forces applied to the mover.
Adjust this value to make movements stronger or weaker according to your needs.

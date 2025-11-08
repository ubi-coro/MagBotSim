.. _custom_mover_mesh:

Using a Custom Mover Mesh
=========================

By default, movers are a simple box geometry. For more complex shapes, you can
specify a custom mesh file or use one of the mesh files supplied by MagBotSim, 
which were generated from the STEP files provided by the manufacturers of 
well-known MagLev systems:

.. include:: ../stl_list.rst

You can optionally specify an STL file for the bumper, which allows the mover
body to be rendered in the designated mover color while the bumper remains
consistently colored in black.

The example below demonstrates using a predefined mesh. To use your own mesh
files, replace the ``mover_stl_path`` and (optionally) ``bumper_stl_path``
with paths to your local mesh files. If you are not using a separate bumper
mesh, you must explicitly set it to ``None``, otherwise the default
``beckhoff_apm4330_bumper`` will be used.

.. code-block:: python

    import gymnasium as gym
    import numpy as np
    import magbotsim
    gym.register_envs(magbotsim)

    env = gym.make(
        'LongHorizonGlobalTrajectoryPlanningEnv-v0',
        layout_tiles=np.ones((3, 3)),
        mover_params={
            'shape': 'mesh',
            'mesh': {
                'mover_stl_path': 'beckhoff_apm4220_mover',
                'bumper_stl_path': 'beckhoff_apm4220_bumper',
                'mass': 0.639 - 0.034, # mover mass (including bumper) - bumper mass
                'bumper_mass': 0.034,
            }
        },
        num_movers=1,
        render_mode='human',
        show_2D_plot=False
    )
    # render env
    env.reset(seed=42)
    try:
        while True:
            env.render()
    except KeyboardInterrupt:
        pass
    env.close()

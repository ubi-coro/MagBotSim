.. _rotation_limits_tutorial:

Rotation Limits
===============
Some MagLev systems may limit the rotation of a mover around the z-axis (yaw), e.g. a full
360° rotation may only be possible at certain positions. To take such limitations into account, MagBotSim provides the
``check_mover_rotation()`` method in the :ref:`basic_magbot_env`. It models a rotation limitation by a set of rotation points: a full rotation is
possible if the (x,y) distance between the center of a mover and one of the rotation points is at most ``tol``. At all
other positions, the yaw of a mover may only deviate by ``max_yaw_deviation`` (in rad) from the nearest multiple of 90°.
The method returns a numpy array of shape (num_movers,), where an element is True if the rotation limit of the
corresponding mover is violated:

.. code-block:: python

    rotation_violation = self.check_mover_rotation(
        mover_names=self.mover_names,
        mover_qpos=None,
        add_qpos_noise=True,  # would also occur in a real system
        tol=0.001,  # [m]
        max_yaw_deviation=np.deg2rad(10.0),
    ).any()

By default, the rotation points model one possible rotation limitation (see ``get_tile_rotation_points()``): a full
rotation is only possible at the center, the four corners, and the four edge midpoints of a tile, and only if the mover
fully covers existing tiles there, i.e. all tiles touching the point must exist (center: 1 tile, edge midpoint: 2 tiles,
corner: 4 tiles). Thus, a full rotation is not possible at the corners and edge midpoints at the border of the system or
next to missing tiles.

If your system has a different rotation limitation, you can specify arbitrary rotation points as a numpy array of shape
(num_points,2), either using the ``rotation_points`` parameter of the ``BasicMagBotEnv`` or by assigning the
``rotation_points`` attribute of an existing environment:

.. code-block:: python

    # full rotation only possible at two specific positions
    env.rotation_points = np.array([[0.36, 0.36], [0.6, 0.12]])

    # default rotation points plus an additional point
    env.rotation_points = np.vstack([env.get_tile_rotation_points(), [[0.42, 0.3]]])

Limitations that cannot be described by points, e.g. rotation areas, can be modeled by overriding
``full_rotation_possible()``, which is used by ``check_mover_rotation()``. If you only want to know whether a full
rotation is possible at certain positions, e.g. to plan where a mover should rotate, you can also use
``full_rotation_possible()`` directly. It ignores the orientation of the movers.

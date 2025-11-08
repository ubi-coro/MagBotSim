.. _apply_controls_and_collision_checking_tutorial:

Applying Controls and Collision Checking in Custom Environments
===============================================================

In this tutorial, we show how to apply controls and check collisions. To this end, 
we further modify the ``PandaBoxExampleEnv`` that we created in the previous 
tutorials :ref:`creating_env` and :ref:`customizing_mujoco_model`. We already added 
actuators for the movers and the Panda robot to the MuJoCo model, but the environment
does not include any method to set actuator controls. To apply controls, 
add the following code to your ``PandaBoxExampleEnv``:

.. code-block:: python

    def apply_controls(
        self, mover_x_ctrls: np.ndarray, panda_ctrls: np.ndarray, num_cycles: int = 40, render_every_cycle: bool = False
    ) -> bool:
        """Apply controls for both mover and Panda actuators, integrate and check for collisions.

        :param mover_x_ctrls: the mover controls (accelerations in x-direction) to be applied. Shape: (num_movers, )
        :param panda_ctrls: the controls for the joints of the Panda robot to be applied: Shape: (9,)
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``.
        :return: a bool value indicating whether there has been a collision
        """
        for _ in range(0, num_cycles):
            # set mover controls
            for idx_mover in range(0, self.num_movers):
                mujoco_utils.set_actuator_ctrl(
                    model=self.model,
                    data=self.data,
                    actuator_name=self.mover_actuator_x_names[idx_mover],
                    value=mover_x_ctrls[idx_mover],
                )
                # ensure that the initial pose (except x pos) is maintained
                # (x position is ignored by the impedance controller, since joint_mask[0]=0)
                self.impedance_controllers[idx_mover].update(
                    model=self.model,
                    data=self.data,
                    pos_d=self.initial_mover_xyz_pos[idx_mover,:],
                    quat_d=np.array([1,0,0,0])
                )
            # set controls Panda
            for idx_pa, actuator_name in enumerate(self.panda_actuator_names):
                mujoco_utils.set_actuator_ctrl(
                    model=self.model, data=self.data, actuator_name=actuator_name, value=panda_ctrls[idx_pa]
                )
            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            # render every cycle for a smooth visualization of the movement
            if render_every_cycle:
                self.render()
            # check wall and mover collision every cycle to ensure that the collisions are detected and
            # all intermediate mover positions are valid and without collisions
            wall_collision = self.check_wall_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            ).any()
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            )
            if mover_collision or wall_collision:
                break
        self.render()

        return mover_collision or wall_collision

This method allows us to apply control for the movers and the Panda robot, but it also checks for 
collisions and renders the environment. Let's take a closer look at the individual parts.

Cycle Times and Actuator Controls
---------------------------------
Actuator controls can be easily set independently of the robot using MagBotSim's :ref:`mujoco_utils`,
if the actuator names are known. Similar to a real system, where actuators expect new setpoints in 
each control cycle, actuator controls are set in every cycle.  We apply the same actuator 
controls for ``num_cycles``. In every control cycle, one MuJoCo step is executed after setting the 
actuator controls, which corresponds to executing one cycle in a real system. 

.. note::
    The cycle time is determined by the ``model.opt.timestep`` parameter.

Since 1 ms is a frequently used cycle time for high-level control tasks, MagBotSim is configured by 
default to use this cycle time.

In this example, the movers can be moved along the x-axis by specifying accelerations. All other 
DoFs of the movers are controlled by the mover-specific impedance controllers (see :ref:`impedance_controller`) 
to ensure that the movers maintain their initial y- and z-positions and orientations.

You can now apply actuator controls to move the Panda robot and the movers. Add a new file 
``apply_controls_panda_box_env.py`` to your directory where ``panda_box_env.py`` is located,  
write the following code and run the script to apply controls:

.. code-block:: python

    import numpy as np

    from panda_box_env import PandaBoxExampleEnv

    # init environment
    env = PandaBoxExampleEnv()
    # set mover controls (accelerations) - the movers should now move in x-direction 
    # until the first mover leaves  the tiles and MagBotSim detects a collision
    mover_x_ctrls = 0.5 * np.ones((env.num_movers,))
    panda_ctrls = np.zeros((9,))
    # apply controls
    collision = False
    try:
        while env.window_viewer_is_running() and not collision:
            collision = env.apply_controls(mover_x_ctrls=mover_x_ctrls, panda_ctrls=panda_ctrls, num_cycles=10, render_every_cycle=True)
    except KeyboardInterrupt:
        pass
    env.close()

.. note::
    On macOS, MuJoCo's passive viewer requires that the script is executed using the ``mjpython`` launcher, i.e. run 
    ``mjpython apply_controls_panda_box_env.py``.


Collision Checking
------------------
After setting the actuator controls and doing one MuJoCo integrator step, it is checked whether 
any movers collided or if any mover left the tiles. The latter is referred to as *wall collision*, 
since typically real systems have a boundary as a safety mechanism so that leaving the tiles 
would result in a crash. In case of a collision, no further simulation steps are performed, as a  
real system would typically stop as well due to position lag errors. 

A typical strategy to encourage a motion planning algorithm to use a sufficiently large distance 
between any two movers or between a mover and the system’s boundaries is to add a safety margin 
to the actual size of a mover. In these or similar scenarios, using MuJoCo’s collision detection 
would necessitate changing the size of the MuJoCo mover object. This requires building and compiling 
a completely new MuJoCo model to ensure that the simulation outputs physically correct results. 
Instead of using MuJoCo’s collision detection mechanism, MagBotSim checks for collisions without 
modifying the actual MuJoCo objects. This is particularly efficient for MagLev systems, as the 
movers and tiles have simple shapes, such as cuboids or cylinders, which significantly simplifies the 
collision checking problem. In addition, safety margins around a mover can be easily changed without 
requiring building and compiling a new MuJoCo model. The collision shapes, sizes, and additional margins
can be configured using the ``collision_params``. Please refer to the :ref:`basic_magbot_env` 
documentation for more information about the ``collision_params``, ``check_wall_collision()``, and 
``check_mover_collision()`` methods.

The complete code for the ``PandaBoxExampleEnv`` can be found `here <https://github.com/ubi-coro/MagBotSim/blob/main/magbotsim/envs_without_rl_api/object_manipulation/panda_box_env.py>`_.
You now know all the basics you need to create your own research-specific environment. If you are interested in more extensive examples of how to use MagBotSim, especially with a RL API, please
refer to one of our included :ref:`environments`.

.. note::
    We only showed a basic example of how to use MagBotSim, which is not specifically optimized. To increase performance, 
    information about MuJoCo objects, e.g. actuators, may be cached. However, this requires a deeper understanding of MuJoCo. 
    Please refer to one of our included :ref:`environments` for an example of how to cache information about MuJoCo objects.



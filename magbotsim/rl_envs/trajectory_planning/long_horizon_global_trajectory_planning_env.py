############################################################################
# Copyright (c) 2024 Cedric Grothues & Lara Bergmann, Bielefeld University #
############################################################################

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import logger

from magbotsim import (
    BasicMagBotSingleAgentEnv,
    Matplotlib2DViewer,
    MoverImpedanceController,
)
from magbotsim.utils import mujoco_utils
from magbotsim.utils.benchmark_utils import BENCHMARK_PLANNING_LAYOUTS, BENCHMARK_PLANNING_NUM_MOVERS, EnergyEfficiencyMeasurement


class LongHorizonGlobalTrajectoryPlanningEnv(BasicMagBotSingleAgentEnv):
    """A simple planning environment.

    :param layout_tiles: a numpy arrays of shape (num_tiles_x, num_tiles_y) indicating where to add a tile (use 1 to add a tile
        and 0 to leave cell empty). The x-axis and y-axis correspond to the axes of the numpy array, so the origin of the base frame is in
        the upper left corner.
    :param num_movers: the number of movers in the environment
    :param show_2D_plot: whether to show a 2D matplotlib plot (useful for debugging)
    :param mover_colors_2D_plot: a list of matplotlib colors, one for each mover (only used if ``show_2D_plot=True``), defaults to
        None. None is only accepted if ``show_2D_plot = False``.
    :param tile_params: a dictionary that can be used to specify the mass and size of a tile using the keys 'mass' or 'size',
        defaults to None. Since one MagLev system usually only contains tiles of one type, i.e. with the same mass and size,
        the mass is a single float value and the size must be specified as a numpy array of shape (3,). If set to None or only one
        key is specified, both mass and size or the missing value are set to the following default values:

        - mass: 5.6 [kg]
        - size: [0.24/2, 0.24/2, 0.0352/2] (x,y,z) [m] (note: half-size)
    :param mover_params: a dictionary that can be used to specify the mass and size of each mover using the keys 'mass' or 'size',
        defaults to None. To use the same mass and size for each mover, the mass can be specified as a single float value and the
        size as a numpy array of shape (3,). However, the movers can also be of different types, i.e. different masses and sizes.
        In this case, the mass and size should be specified as numpy arrays of shapes (num_movers,) and (num_movers,3),
        respectively. If set to None or only one key is specified, both mass and size or the missing value are set to the following
        default values:

        - mass: 1.24 [kg]
        - size: [0.155/2, 0.155/2, 0.012/2] (x,y,z) [m] (note: half-size)
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.003 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after ``num_cycles`` have been executed if ``render_mode != None``.
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: a dictionary that can be used to specify the following collision parameters, defaults to None:

        - collision shape (key: 'shape'): can be 'box' or 'circle', defaults to 'circle'
        - size of the collision shape (key: 'size'), defaults to 0.11 [m]:
            - collision shape 'circle':
                a single float value which corresponds to the radius of the circle, or a numpy array of shape (num_movers,) to specify
                individual values for each mover
            - collision shape 'box':
                a numpy array of shape (2,) to specify x and y half-size of the box, or a numpy array of shape (num_movers, 2) to
                specify individual sizes for each mover

        - additional size offset (key: 'offset'), defaults to 0.0 [m]: an additional safety offset that is added to the size of the
            collision shape. Think of this offset as increasing the size of a mover by a safety margin.
        - additional wall offset (key: 'offset_wall'), defaults to 0.0 [m]: an additional safety offset that is added to the size
            of the collision shape to detect wall collisions. Think of this offset as moving the wall, i.e. the edge of a tile
            without an adjacent tile, closer to the center of the tile.

    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk (only used if ``learn_jerk=True``), defaults to 100.0 [m/s³]
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param threshold_pos: the position threshold used to determine whether a mover has reached its goal position, defaults
        to 0.1 [m]
    :param reward_success: reward in case of success multiplied with the number of goals reached in the current step, defaults to 20.0
    :param reward_collision: reward in case of collision, defaults to -20.0
    :param reward_per_step: reward per step (no collision and no goal reached), defaults to -1.0
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    :param timeout_steps: the number of steps after which the episode is terminated if no goal is reached by any mover within that
        specified time limit, defaults to 50
    :param enable_energy_tracking: whether to track energy metrics for benchmarking purposes, defaults to False
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        reward_success: float = 20.0,
        reward_collision: float = -20.0,
        reward_per_step: float = -1.0,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        self.learn_jerk = learn_jerk
        self.enable_energy_tracking = enable_energy_tracking

        # impedance controllers
        self.impedance_controllers = None

        super().__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=self._compute_default_cam_config(layout_tiles, tile_params),
            render_every_cycle=render_every_cycle,
            num_cycles=num_cycles,
            collision_params=collision_params,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        self.num_steps = 0
        self.timeout_steps = timeout_steps
        self.last_goal_reached = 0
        self.collected_goals = np.zeros((num_movers,), dtype=np.uint32)

        # maximum velocity, acceleration and jerk
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max

        # position threshold in m
        self.threshold_pos = threshold_pos
        # reward in case of success multiplied with the number of goals reached in the current step
        self.reward_success = reward_success
        # reward in case of collision
        self.reward_collision = reward_collision
        # reward per step (no collision and no goal reached)
        self.reward_per_step = reward_per_step
        # whether to show a 2D matplotlib plot
        self.show_2D_plot = show_2D_plot

        # impedance contoller
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                joint_mask=np.array([0, 0, 1, 1, 1, 1]),
                translational_stiffness=np.array([1.0, 1.0, 100.0]),
                rotational_stiffness=np.array([0.1, 0.1, 1]),
            )
            for mover_idx in range(self.num_movers)
        ]
        self.reload_model(None, None)

        # observation space
        # observation:
        #   - velocities in x and y direction of each mover
        #   - accelerations in x and y direction of each mover if learn_jerk = True
        # achieved_goal:
        #   the current (x,y)-position of each mover
        # desired_goal:
        #   the (x,y) goal position of each mover
        low_goals = np.zeros((self.num_movers * 2,))
        high_goals = np.array(
            [
                np.max(self.x_pos_tiles) + (self.tile_size[0] / 2),
                np.max(self.y_pos_tiles) + (self.tile_size[1] / 2),
            ]
            * self.num_movers
        )
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_movers * (1 + int(self.learn_jerk)) * 2,),
                    dtype=np.float64,
                ),
                'achieved_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
                'desired_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
            }
        )

        # action space
        as_low = -self.j_max if self.learn_jerk else -self.a_max
        as_high = self.j_max if self.learn_jerk else self.a_max
        self.action_space = gym.spaces.Box(
            low=as_low,
            high=as_high,
            shape=(self.num_movers * 2,),
            dtype=np.float64,
        )

        # minimum and maximum possible mover (x,y)-positions
        safety_margin = self.c_size + self.c_size_offset_wall + self.c_size_offset
        self.min_xy_pos = np.zeros(2) + safety_margin
        self.max_xy_pos = (
            np.array(
                [
                    np.max(self.x_pos_tiles) + (self.tile_size[0] / 2),
                    np.max(self.y_pos_tiles) + (self.tile_size[1] / 2),
                ]
            )
            - safety_margin
        )

        # minimum distance between any two goals
        if self.c_shape == 'circle':
            self.min_goal_dist = 2 * (self.c_size + self.c_size_offset)
        else:
            # self.c_shape == 'box'
            self.min_goal_dist = 2 * np.linalg.norm(self.c_size + self.c_size_offset, ord=2)

        # 2D plot
        if self.show_2D_plot:
            if not mover_colors_2D_plot:
                raise ValueError('Please specify the colors of the movers for the 2D plot.')

            self.matplotlib_2D_viewer = Matplotlib2DViewer(
                layout_tiles=self.layout_tiles,
                num_movers=self.num_movers,
                mover_sizes=self.mover_size,
                mover_colors=mover_colors_2D_plot,
                tile_size=self.tile_size,
                x_pos_tiles=self.x_pos_tiles,
                y_pos_tiles=self.y_pos_tiles,
                c_shape=self.c_shape,
                c_size=self.c_size,
                c_size_offset=self.c_size_offset,
                arrow_scale=0.2,
                figure_size=(9, 9),
            )

        if self.enable_energy_tracking:
            self.energy_tracker = EnergyEfficiencyMeasurement(
                num_movers=self.num_movers,
                dt=self.cycle_time,
            )

    def _compute_default_cam_config(
        self,
        layout_tiles: np.ndarray,
        tile_params: dict[str, Any] | None = None,
        factor: float = 1.1,
    ) -> dict[str, Any]:
        if tile_params and 'size' in tile_params:
            tile_size = tile_params['size'][:2]
        else:
            tile_size = np.array([0.12, 0.12])

        x, y = layout_tiles.shape * tile_size

        diag = np.sqrt(x**2 + y**2)
        distance = (diag * factor) / np.cos(np.radians(-65.0))

        return {
            'distance': distance,
            'azimuth': 90.0,
            'elevation': -65.0,
            'lookat': np.array([x, y, 0.067]),
        }

    def update_cached_actuator_mujoco_data(self) -> None:
        """Update all cached information about MuJoCo actuators, such as names and ids."""
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='actuator', name_pattern='mover_actuator_x')
        self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='actuator', name_pattern='mover_actuator_y')

        self.mover_actuator_x_ids = np.zeros((len(self.mover_actuator_x_names),), dtype=np.int32)
        self.mover_actuator_y_ids = np.zeros((len(self.mover_actuator_x_names),), dtype=np.int32)
        for idx_a, actuator_x_name in enumerate(self.mover_actuator_x_names):
            self.mover_actuator_x_ids[idx_a] = self.model.actuator(actuator_x_name).id
            self.mover_actuator_y_ids[idx_a] = self.model.actuator(self.mover_actuator_y_names[idx_a]).id

        for mover_idx in range(self.num_movers):
            self.impedance_controllers[mover_idx].update_cached_actuator_mujoco_data(self.model)

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """For each mover, this callback adds actuators to the ``custom_model_xml_strings``-dict, depending on whether the jerk or
        acceleration is the output of the policy.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified ``custom_model_xml_strings``-dict
        """
        mover_actuator_list = ['\n\t<actuator>', '\t\t<!-- mover actuators -->']
        for idx_mover in range(0, self.num_movers):
            joint_name = f'mover_joint_{idx_mover}'
            mover_mass = self.mover_mass if isinstance(self.mover_mass, float) else self.mover_mass[idx_mover]

            if self.learn_jerk:
                mover_actuator_list.extend(
                    [
                        f'\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" '
                        f'dyntype="integrator" gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>',
                        f'\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                        f'dyntype="integrator" gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none" actearly="true"/>',
                    ]
                )
            else:
                # learn acceleration
                mover_actuator_list.extend(
                    [
                        f'\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none" '
                        f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>',
                        f'\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" dyntype="none" '
                        f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>',
                    ]
                )
            if self.impedance_controllers is not None:
                impedance_controller = self.impedance_controllers[idx_mover]
                mover_actuator_list.append(impedance_controller.generate_actuator_xml_string(idx_mover=idx_mover))

        mover_actuator_list.append('\t</actuator>')

        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
        custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
        mover_actuator_xml_str = '\n'.join(mover_actuator_list)
        if custom_outworldbody_xml_str is not None:
            custom_outworldbody_xml_str += mover_actuator_xml_str
        else:
            custom_outworldbody_xml_str = mover_actuator_xml_str
        custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray, mover_goal_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start and goal positions and reload the model. In this environment, it is necessary
        to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover.
        :param mover_goal_xy_pos: a numpy array of shape (num_movers_with_goals,2) containing the (x,y) goal positions of the
            movers (num_movers_with_goals <= num_movers)
        """
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos,
            mover_goal_xy_pos=mover_goal_xy_pos,
            custom_xml_strings=custom_model_xml_strings,
        )
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # update cached mujoco data
        self.update_cached_mover_mujoco_data()
        self.update_cached_actuator_mujoco_data()

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the start and goal positions of all movers and reload the model. It is also checked whether the start positions are
        collision-free (mover and wall collisions) and whether the new goals can be reached without mover or wall collisions.

        :param options: not used in this environment
        """
        if self.enable_energy_tracking:
            self.energy_tracker.reset()

        options = options or {}

        self.num_steps = 0
        self.last_goal_reached = 0
        self.collected_goals = np.zeros((self.num_movers,), dtype=np.uint32)

        # sample new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        # sample movers one after the other to ensure collision-free start positions
        counter = 0
        for mover_idx in range(self.num_movers):
            mover_pos_valid = False
            while not mover_pos_valid:
                counter += 1
                if counter > 0 and counter % 100 == 0:
                    logger.warn(
                        f'Trying to find a collision-free start position for mover {mover_idx}. '
                        + f'No valid position found within {counter} trails. Consider choosing fewer movers or more tiles.'
                    )

                # sample position for current mover
                start_qpos[mover_idx, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(2,))

                # check wall collision for current mover
                pos_is_valid = self.qpos_is_valid(
                    qpos=start_qpos[mover_idx : mover_idx + 1],
                    c_size=self.c_size,
                    add_safety_offset=True,
                )

                # check collision with previously placed movers
                mover_collision = False
                if mover_idx > 0:
                    # only check collision with movers that have already been placed
                    placed_movers_qpos = start_qpos[: mover_idx + 1]
                    mover_collision = self.check_mover_collision(
                        mover_names=self.mover_names[: mover_idx + 1],
                        c_size=self.c_size,
                        add_safety_offset=True,
                        mover_qpos=placed_movers_qpos,
                    )

                if not mover_collision and pos_is_valid.all():
                    mover_pos_valid = True

        self._sample_goals()

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2], mover_goal_xy_pos=self.goals)

    def _before_mujoco_step_callback(self, action: np.ndarray) -> None:
        """Apply the next action, i.e. it sets the jerk or acceleration, ensuring the minimum and maximum velocity and acceleration
        (for one cycle).

        :param action: a numpy array of shape (num_movers * 2,), which specifies the next action (jerk or acceleration)
        """
        action = action.reshape((self.num_movers, 2))

        vel = self.get_mover_qvel(mover_names=self.mover_names, add_noise=True)[:, :2]
        if self.learn_jerk:
            acc = self.get_mover_qacc(mover_names=self.mover_names, add_noise=False)[:, :2]
            next_acc_tmp, next_jerk = self.ensure_max_dyn_val(current_values=acc, max_value=self.a_max, next_derivs=action)
            _, next_acc = self.ensure_max_dyn_val(
                current_values=vel,
                max_value=self.v_max,
                next_derivs=next_acc_tmp,
            )
            if (next_acc_tmp != next_acc).any():
                next_jerk = (next_acc - acc) / self.cycle_time
            ctrl = next_jerk.copy()
        else:
            _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=action)
            ctrl = next_acc.copy()

        self.data.ctrl[self.mover_actuator_x_ids] = ctrl[:, 0]
        self.data.ctrl[self.mover_actuator_y_ids] = ctrl[:, 1]

        assert self.impedance_controllers is not None

        for mover_idx in range(self.num_movers):
            self.impedance_controllers[mover_idx].update(
                model=self.model,
                data=self.data,
                pos_d=np.array(
                    [
                        0,
                        0,
                        self.initial_mover_zpos + self.mover_size[mover_idx, 2] if self.mover_size.ndim == 2 else self.mover_size[2],
                    ]
                ),
                quat_d=np.array([1, 0, 0, 0]),
            )

    def _sample_goals(self, mover_idxs: np.ndarray | None = None, current_goals: np.ndarray | None = None) -> None:
        """Sample new goal positions for specified movers.

        This method generates new goal positions for the movers specified by their indices,
        ensuring that all goals are collision-free and satisfy minimum distance constraints.
        The sampling process continues until valid goal positions are found for all movers.

        :param mover_idxs: a numpy array containing the indices of movers that need new goal positions
        :param current_goals: a numpy array containing the current (x,y)-positions of all goals
        """
        # sample new goal positions
        goal_qpos = np.zeros((self.num_movers, 7))
        goal_qpos[:, 2] = self.initial_mover_zpos
        goal_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        if current_goals is not None:
            goal_qpos[:, :2] = current_goals

        # ensure that all goal positions can be reached without wall collisions
        counter = 0
        all_goal_pos_reachable = False
        while not all_goal_pos_reachable:
            counter += 1
            if counter > 0 and counter % 100 == 0:
                logger.warn(
                    'Trying to find valid goal positions for all movers. '
                    + f'No valid configuration found within {counter} trails. Consider choosing fewer movers or more tiles.'
                )

            if mover_idxs is not None:
                goal_qpos[mover_idxs, :2] = self.np_random.uniform(
                    low=self.min_xy_pos,
                    high=self.max_xy_pos,
                    size=(mover_idxs.size, 2),
                )
            else:
                goal_qpos[:, :2] = self.np_random.uniform(
                    low=self.min_xy_pos,
                    high=self.max_xy_pos,
                    size=(self.num_movers, 2),
                )

            goal_pos_reachable = self.qpos_is_valid(qpos=goal_qpos, c_size=self.c_size, add_safety_offset=True)
            all_goal_pos_reachable = np.sum(goal_pos_reachable) == self.num_movers

        self.goals = goal_qpos[:, :2].copy()

        if mover_idxs is not None:
            for mover_idx in mover_idxs:
                self.model.site(f'goal_site_mover_{mover_idx}').pos[:2] = self.goals[mover_idx]

    def _on_step_end_callback(self, observation: dict[str, np.ndarray] | np.ndarray) -> None:
        """Callback executed at the end of every step to handle goal completion and resampling.

        This callback increments the step counter and checks if any mover has reached its goal.
        When a mover reaches its goal (distance <= threshold_pos), new goal positions are
        automatically sampled for those movers and the episode continues indefinitely.
        The callback also updates the timestamp of when the last goal was reached, which
        is used for timeout detection.

        :param observation: the current observation dictionary containing 'achieved_goal'
            and 'desired_goal' arrays
        """
        if self.enable_energy_tracking:
            # Get current velocities and accelerations for all movers
            velocities = np.array(
                [self.get_mover_qvel(mover_name=mover_name, add_noise=False)[:2].flatten() for mover_name in self.mover_names]
            )
            accelerations = np.array(
                [self.get_mover_qacc(mover_name=mover_name, add_noise=False)[:2].flatten() for mover_name in self.mover_names]
            )

            self.energy_tracker.update(velocities, accelerations)

        self.num_steps += 1

        achieved_goal = observation['achieved_goal']
        desired_goal = observation['desired_goal']

        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        dist_goal = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal)
        assert dist_goal.shape == (batch_size, self.num_movers)
        goal_reached = dist_goal <= self.threshold_pos

        if np.any(goal_reached):
            self.collected_goals += goal_reached.squeeze()
            self.last_goal_reached = self.num_steps
            self._sample_goals(np.where(goal_reached.squeeze(axis=0))[0], self.goals.copy())

    def _render_callback(self) -> None:
        """Update the Matplotlib2DViewer if ``show_2D_plot=True``."""
        if self.show_2D_plot:
            mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            mover_qvel = self.get_mover_qvel(mover_names=self.mover_names, add_noise=False)
            self.matplotlib_2D_viewer.render(
                mover_qpos=mover_qpos,
                mover_qvel=mover_qvel,
                mover_goals=self.goals,
            )

    def compute_terminated(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. A state is terminal when there is a collision between two movers or between a
        mover and a wall.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,). An entry is True if the state is terminal, False otherwise
            - if batch_size = 1:
                True if the state is terminal, False otherwise
        """
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        return reward == self.reward_collision

    def compute_truncated(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. Since the environment continues indefinitely, there is no
        truncation, unless the ``timeout_steps`` parameter is set, in which case an episode is truncated if no mover has
        reached any goal within the last ``timeout_steps`` steps.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,) indicating whether truncation should occur
            - if batch_size = 1:
                True if truncation should occur due to timeout, False otherwise
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

        if not self.timeout_steps:
            return np.array([False] * batch_size) if batch_size > 1 else False

        if info is None or 'steps_since_goal' not in info:
            steps_since_goal = np.array([0] * batch_size) if batch_size > 1 else 0
        else:
            steps_since_goal = info['steps_since_goal']

        return steps_since_goal > self.timeout_steps

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray | float:
        """Compute the immediate reward.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        batch_size, mover_collisions, wall_collisions = self._preprocess_info_dict(info=info)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        mask_collision = np.bitwise_or(mover_collisions, wall_collisions)
        mask_no_collision = np.bitwise_not(mask_collision)

        dist_goal = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal)
        assert dist_goal.shape == (batch_size, self.num_movers)
        goal_reached = dist_goal <= self.threshold_pos
        num_goals_reached = np.sum(goal_reached, axis=1)
        assert num_goals_reached.shape == (batch_size,)

        reward = self.reward_collision * mask_collision.astype(np.float64)
        reward += self.reward_per_step * mask_no_collision.astype(np.float64)
        mask_success = np.bitwise_and(num_goals_reached > 0, mask_no_collision)
        reward[mask_success] = self.reward_success * num_goals_reached[mask_success]

        assert reward.shape == (batch_size,)
        return reward if batch_size > 1 else reward[0]

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a dictionary containing the following keys and values:
            - 'observation':

                - if ``learn_jerk=True``: a numpy array of shape (num_movers*2*2,) containing the (x,y)-velocities and
                                          (x,y)-accelerations of each mover
                                          ((x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...)
                - if ``learn_jerk=False``: a numpy array of shape (num_movers*2,) containing the (x,y)-velocities and of each mover
                                           ((x,y)-velo mover 1, (x,y)-velo mover 2, ...)
            - 'achieved_goal':
                a numpy array of shape (num_movers*2,) containing the current (x,y)-positions of all movers
                ((x,y)-pos mover 1, (x,y)-pos mover 2, ...)
            - 'desired_goal':
                a numpy array of shape (num_movers*2,) containing the desired (x,y)-positions of all movers
                ((x,y) goal pos mover 1, (x,y) goal pos mover 2, ...)
        """
        mover_xy_pos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=True)[:, :2]
        mover_xy_velos = self.get_mover_qvel(mover_names=self.mover_names, add_noise=True)[:, :2]
        if self.learn_jerk:
            # no noise, because only SetAcc is available in a real system
            mover_xy_accs = self.get_mover_qacc(mover_names=self.mover_names, add_noise=False)[:, :2]

        observation = np.concatenate((mover_xy_velos, mover_xy_accs), axis=0) if self.learn_jerk else mover_xy_velos.copy()
        achieved_goal = mover_xy_pos.copy()
        desired_goal = self.goals.copy()

        return OrderedDict(
            [
                # (x,y)-velo mover 1, (x,y)-velo mover 2, ..., (x,y)-acc mover 1, (x,y)-acc mover 2, ...
                ('observation', observation.flatten()),
                # (x,y)-pos mover 1, (x,y)-pos mover 2, ...
                ('achieved_goal', achieved_goal.flatten()),
                # (x,y) goal pos mover 1, (x,y) goal pos mover 2, ...
                ('desired_goal', desired_goal.flatten()),
            ]
        )

    def _get_info(
        self,
        mover_collision: bool,
        wall_collision: bool,
        other_collision: bool,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        collision_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a dictionary that contains auxiliary information.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param other_collision: whether there are other collisions besides wall or mover collisions, e.g. collisions with an obstacle
            (not used in this environment)
        :param achieved_goal: a numpy array of shape (length achieved_goal,) containing the (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (length achieved_goal,) containing the desired (x,y)-positions
        :param collision_info: a dictionary that is intended to contain additional information about collisions, e.g.
            collisions with obstacles. Defaults to None (not used in this environment)
        :return: the info dictionary with keys 'collected_goals', 'mover_collision', 'wall_collision', and 'steps_since_goal'
        """
        assert not isinstance(mover_collision, np.ndarray)
        assert not isinstance(wall_collision, np.ndarray)
        info = {
            'collected_goals': self.collected_goals,
            'mover_collision': mover_collision,
            'wall_collision': wall_collision,
            'steps_since_goal': self.num_steps - self.last_goal_reached,
        }

        if self.enable_energy_tracking:
            info['energy_efficiency'] = {
                'cumulative_energy_metric': self.energy_tracker.cumulative_energy_metric,
                'average_energy_metric': self.energy_tracker.average_energy_metric,
                'min_energy_metric': self.energy_tracker.min_energy_metric,
                'max_energy_metric': self.energy_tracker.max_energy_metric,
            }

        return info

    def close(self) -> None:
        """Close the environment."""
        super().close()
        if self.show_2D_plot:
            self.matplotlib_2D_viewer.close()

    def ensure_max_dyn_val(
        self,
        current_values: np.ndarray,
        max_value: float,
        next_derivs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ensure the minimum and maximum dynamic values.

        :param current_values: the current velocity or acceleration specified as a numpy array of shape (2,) or
            (num_checks,2)
        :param max_value: the maximum velocity or acceleration (float)
        :param next_derivs: the next derivative (acceleration or jerk) used for one integrator step specified as a numpy array of
            shape (2,) or (num_checks,2)
        :return: the next velocity or acceleration and the next derivative (acceleration or jerk) corresponding to the next action
            that must be applied to ensure the minimum and maximum dynamics (each of shape (num_checks,2))
        """
        if len(current_values.shape) == 1:
            current_values = current_values.reshape((1, -1))
        if len(next_derivs.shape) == 1:
            next_derivs = next_derivs.reshape((1, -1))

        next_values = np.zeros((current_values.shape[0], 2))
        next_derivs_new = np.zeros((current_values.shape[0], 2))

        next_values_tmp = self.cycle_time * next_derivs + current_values

        norm_next_values_tmp = np.linalg.norm(next_values_tmp, ord=2, axis=1)
        mask_norm = norm_next_values_tmp >= max_value

        next_values[np.bitwise_not(mask_norm), :] = next_values_tmp[np.bitwise_not(mask_norm), :]
        next_derivs_new[np.bitwise_not(mask_norm), :] = next_derivs[np.bitwise_not(mask_norm), :]

        if mask_norm.any():
            next_values[mask_norm] = max_value * np.divide(
                next_values_tmp[mask_norm],
                np.broadcast_to(
                    norm_next_values_tmp[mask_norm, np.newaxis],
                    (np.sum(mask_norm), 2),
                ),
            )
            next_derivs_new[mask_norm] = (next_values[mask_norm] - current_values[mask_norm]) / self.cycle_time

        return next_values, next_derivs_new

    def _calc_eucl_dist_xy(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """Calculate the Euclidean distance.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            (x,y)-positions already achieved
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of all movers
        :return: a numpy array of shape (batch_size,num_movers), which contains the distances between the achieved and the desired goals
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        achieved_goal_tmp = achieved_goal.reshape((batch_size, self.num_movers, 2))
        desired_goal_tmp = desired_goal.reshape((batch_size, self.num_movers, 2))

        return np.linalg.norm(achieved_goal_tmp - desired_goal_tmp, ord=2, axis=2)

    def _preprocess_info_dict(self, info: np.ndarray | dict[str, Any] | None) -> tuple[int, np.ndarray, np.ndarray]:
        """Extract information about mover collisions, wall collisions and the batch size from the info dictionary.

        :param info: the info dictionary or an array of info dictionary to be preprocessed. All dictionaries must contain the keys
            'mover_collision' and 'wall_collision'.
        :return: the batch_size (int), a numpy array of shape (batch_size,) containing the mover collision values (bool),
            a numpy array of shape (batch_size,) containing the wall collision values (bool)
        """
        if isinstance(info, np.ndarray):
            batch_size = info.shape[0]
            mover_collisions = np.zeros(batch_size).astype(bool)
            wall_collisions = np.zeros(batch_size).astype(bool)

            for i in range(0, batch_size):
                mover_collisions[i] = info[i]['mover_collision']
                wall_collisions[i] = info[i]['wall_collision']
        else:
            assert isinstance(info, dict)
            batch_size = 1
            mover_collisions = np.array([info['mover_collision']])
            wall_collisions = np.array([info['wall_collision']])

        return batch_size, mover_collisions, wall_collisions


class LongHorizonGlobalTrajectoryPlanningEnvB0(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[0],
            BENCHMARK_PLANNING_NUM_MOVERS[0],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB1(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[1],
            BENCHMARK_PLANNING_NUM_MOVERS[1],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB2(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[2],
            BENCHMARK_PLANNING_NUM_MOVERS[2],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB3(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[3],
            BENCHMARK_PLANNING_NUM_MOVERS[3],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB4(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[4],
            BENCHMARK_PLANNING_NUM_MOVERS[4],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB5(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[5],
            BENCHMARK_PLANNING_NUM_MOVERS[5],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB6(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[6],
            BENCHMARK_PLANNING_NUM_MOVERS[6],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB7(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[7],
            BENCHMARK_PLANNING_NUM_MOVERS[7],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB8(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[8],
            BENCHMARK_PLANNING_NUM_MOVERS[8],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )


class LongHorizonGlobalTrajectoryPlanningEnvB9(LongHorizonGlobalTrajectoryPlanningEnv):
    def __init__(
        self,
        show_2D_plot: bool,
        mover_colors_2D_plot: list[str] | None = None,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2,
        a_max: float = 10,
        j_max: float = 100,
        learn_jerk: bool = False,
        threshold_pos: float = 0.1,
        use_mj_passive_viewer: bool = False,
        timeout_steps: int | None = 50,
        enable_energy_tracking: bool = False,
    ) -> None:
        super().__init__(
            BENCHMARK_PLANNING_LAYOUTS[9],
            BENCHMARK_PLANNING_NUM_MOVERS[9],
            show_2D_plot,
            mover_colors_2D_plot,
            tile_params,
            mover_params,
            initial_mover_zpos,
            std_noise,
            render_mode,
            render_every_cycle,
            num_cycles,
            collision_params,
            v_max,
            a_max,
            j_max,
            learn_jerk,
            threshold_pos,
            use_mj_passive_viewer,
            timeout_steps,
            enable_energy_tracking,
        )

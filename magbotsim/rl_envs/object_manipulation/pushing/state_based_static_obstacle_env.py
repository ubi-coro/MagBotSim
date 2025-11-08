##########################################################
# Copyright (c) 2025 Lara Bergmann, Bielefeld University #
##########################################################

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import logger

from magbotsim import BasicMagBotSingleAgentEnv, MoverImpedanceController
from magbotsim.utils import mujoco_utils, benchmark_utils


class StateBasedStaticObstaclePushingEnv(BasicMagBotSingleAgentEnv):
    """A simple object pushing environment with two static obstacles.

    :param obstacle_mode: the obstacle mode of this environment which determines the size of the obstacles and the difficulty of the
        task, defaults to 'simple'. Can be 'hard' (large obstacles), 'medium' (medium-sized obstacles), 'simple' (small obstacles),
        'random' (choose size of the obstacles at random), or 'curriculum' (increase the size of the obstacle using a curriculum).
    :param mover_params: Dictionary specifying mover properties. If None, default values are used. Supported keys:

        - mass (float | numpy.ndarray): Mass in kilograms. Options:
            - Single float: Same mass for all movers
            - 1D array (num_movers,): Individual masses per mover

        Default: 1.24 [kg]

        - shape (str | list[str]): Mover shape type. Must be one of:
            - 'box': Rectangular cuboid
            - 'cylinder': Cylindrical shape
            - 'mesh': Custom 3D mesh

            Default: 'box'

        - size (numpy.ndarray): Shape dimensions in meters. Format depends on shape:
            - For 'box': Half-sizes (x, y, z)
            - For 'cylinder': (radius, height, _)
            - For 'mesh': Computed from mesh dimensions multiplied by scale factors in mesh['scale']

            Specification options:
            - 1D array (3,): Same size for all movers
            - 2D array (num_movers, 3): Individual sizes per mover

            Default: [0.155/2, 0.155/2, 0.012/2] [m]

        - mesh (dict): Configuration for mesh-based shapes. Required when shape='mesh'. Contains:
            - mover_stl_path (str): Path to mover mesh STL file or one of the predefined meshes:
                - 'beckhoff_apm4330_mover': Beckhoff APM4220 mover mesh (default)
                - 'beckhoff_apm4220_mover': Beckhoff APM4220 mover mesh
                - 'beckhoff_apm4550_mover': Beckhoff APM4550 mover mesh
                - 'planar_motor_M3-06': Planar Motor M3-06 mover mesh
                - 'planar_motor_M3-15': Planar Motor M3-15 mover mesh
                - 'planar_motor_M3-25': Planar Motor M3-25 mover mesh
                - 'planar_motor_M4-11': Planar Motor M4-11 mover mesh
                - 'planar_motor_M4-18': Planar Motor M4-18 mover mesh
            - bumper_stl_path (str | None): Path to bumper mesh STL file or one of the predefined meshes:
                - 'beckhoff_apm4330_bumper': Beckhoff APM4330 bumper mesh (default)
                - 'beckhoff_apm4220_bumper': Beckhoff APM4220 bumper mesh
                - 'beckhoff_apm4550_bumper': Beckhoff APM4550 bumper mesh
            - bumper_mass (float | numpy.ndarray): Bumper mass in kilograms. Can be specified as:
                - Single float: Same mass applied to all bumpers
                - 1D array (num_movers,): Individual masses for each bumper

                Default: 0.1 [kg]
            - scale (numpy.ndarray): Scale factors for mesh dimensions (x, y, z). Multiplied with the
                mesh geometry. Specification options:
                - 1D array (3,): Same scale factors applied to all movers
                - 2D array (num_movers, 3): Individual scale factors for each mover

                Default: [1.0, 1.0, 1.0] (no scaling)

        - material (str | list[str]): Material name to apply to the mover. Can be specified as:
            - Single string: Same material for all movers
            - List of strings: Individual materials for each mover

            Default: "gray" for movers without goals, color-coded materials for movers with goals

        Note: Custom mesh STL files must have their origin at the mover's center.
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.002 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle:  whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
        False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
        Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
        is always rendered after ``num_cycles`` have been executed if ``render_mode != None``.
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: _description_, defaults to None
    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk (only used if ``learn_jerk=True``), defaults to 100.0 [m/s³]
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param threshold_pos: the position threshold used to determine whether a mover has reached its goal position, defaults
        to 0.05 [m]
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the
        Gymnasium MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        obstacle_mode: str = 'simple',
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.002,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        learn_jerk: bool = False,
        threshold_pos: float = 0.05,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.learn_jerk = learn_jerk

        # object parameters, object type: box
        self.object_length_xy = 0.07 / 2  # [m] (half-size)
        self.object_height = 0.04 / 2  # [m] (half-size)
        self.object_mass = 0.03  # [kg]
        self.object_xy_start_pos = np.array([0.12, 0.36])
        self.object_xy_goal_pos = np.array([0.36, 0.36])
        self.object_noise_xy_pos = 1e-5

        # obstacles
        self.obstacle_mode = obstacle_mode
        self.valid_obstacle_modes = ['simple', 'medium', 'hard', 'curriculum', 'random']
        assert self.obstacle_mode in self.valid_obstacle_modes, f"Unkown obstacle mode: '{self.obstacle_mode}'"
        self.tmp_obstacle_mode = 'simple'  # curriculum learning
        self.ep_counter = 0  # curriculum learning
        self.cl_num_ep_before_incr = 800  # curriculum

        self.obstacle_x_pos = 0.48
        self.obstacle_size_x = 0.03  # [m] (half-size)
        self.max_obstacle_size_y = 0.13  # [m] (half-size)
        self.medium_obstacle_size_y = 0.115  # [m] (half-size)
        self.min_obstacle_size_y = 0.1  # [m] (half-size)
        self.max_obstacle_1_y_pos = 0.62
        self.medium_obstacle_1_y_pos = 0.605
        self.min_obstacle_1_y_pos = 0.59
        self._update_obstacle_size(
            obstacle_mode=self.tmp_obstacle_mode if self.obstacle_mode in ['random', 'curriculum'] else self.obstacle_mode
        )

        # there is only one mover in this environment -> remember index
        self.idx_mover = 0

        # impedance controller
        self.impedance_controller = None

        # cam config
        default_cam_config = {
            'distance': 0.8,
            'azimuth': 160.0,
            'elevation': -55.0,
            'lookat': np.array([0.8, 0.2, 0.4]),
        }

        # init basic env
        super().__init__(
            layout_tiles=np.ones((4, 3)),
            num_movers=1,
            tile_params=None,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            render_every_cycle=render_every_cycle,
            num_cycles=num_cycles,
            collision_params=collision_params,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        # maximum velocity, acceleration and jerk
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max

        # position threshold in m
        self.threshold_pos = threshold_pos
        # reward for a collision between the mover and a wall or between mover and obstacle or between object and obstacle
        self.reward_collision = -50

        # observation space
        low_goals = np.zeros(2)
        high_goals = np.array([np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)])
        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_movers * (2 + int(self.learn_jerk) + 4 * 2) * 2,), dtype=np.float64
                ),
                'achieved_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
                'desired_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
            }
        )

        # action space
        as_low = -self.j_max if self.learn_jerk else -self.a_max
        as_high = self.j_max if self.learn_jerk else self.a_max
        self.action_space = gym.spaces.Box(low=as_low, high=as_high, shape=(self.num_movers * 2,), dtype='float64')

        # minimum and maximum possible mover (x,y)-positions
        safety_margin_small = 0.1  # [m]
        self.min_object_start_xy_pos = np.zeros(2) + self.c_size + self.c_size_offset + self.c_size_offset_wall + safety_margin_small + 0.05
        max_object_start_y_pos = (
            np.max(self.y_pos_tiles)
            + self.tile_size[1]
            - (self.c_size + self.c_size_offset + self.c_size_offset_wall + safety_margin_small + 0.05)
        )
        if isinstance(max_object_start_y_pos, np.ndarray):
            max_object_start_y_pos = max_object_start_y_pos[1]
        self.max_object_start_xy_pos = np.array(
            [self.obstacle_x_pos - self.obstacle_size_x - self.object_length_xy - 0.05, max_object_start_y_pos]
        )

        self.min_mover_start_xy_pos = np.zeros(2) + self.c_size + self.c_size_offset + self.c_size_offset_wall + 0.05

        self.max_mover_start_xy_pos = (
            np.array([self.obstacle_x_pos - self.obstacle_size_x, np.max(self.y_pos_tiles) + self.tile_size[1]])
            - self.c_size
            - self.c_size_offset
            - self.c_size_offset_wall
            - 0.05
        )
        # minimum and maximum possible goal (x,y)-positions
        self.min_goal_xy_pos = (
            np.array([self.obstacle_x_pos + np.max(np.array([self.obstacle_size_x, self.object_length_xy])), 0.0]) + safety_margin_small
        )
        self.max_goal_xy_pos = (
            np.array([np.max(self.x_pos_tiles) + self.tile_size[0], np.max(self.y_pos_tiles) + self.tile_size[1]]) - safety_margin_small
        )

        # impedance contoller
        self.impedance_controller = MoverImpedanceController(
            model=self.model,
            mover_joint_name=self.mover_joint_names[0],
            joint_mask=np.array([0, 0, 1, 1, 1, 1]),
            translational_stiffness=1.0,
            rotational_stiffness=0.1,
        )
        self.reload_model()

        # minimum distance between object and mover after env reset
        if self.c_shape == 'circle':
            self.min_mo_dist = max(
                np.linalg.norm(self.object_length_xy + self.mover_size.flatten()[:2], ord=2), self.c_size + self.c_size_offset
            )
        else:
            # self.c_shape == 'box'
            self.min_mo_dist = max(
                np.linalg.norm(self.object_length_xy + self.mover_size.flatten()[:2], ord=2),
                np.linalg.norm(self.c_size + self.c_size_offset, ord=2),
            )

        # corrective movements
        self.cm_measurement = benchmark_utils.CorrectiveMovementMeasurement(
            distance_func=self._calc_eucl_dist_xy, threshold=self.threshold_pos
        )

        # throughput
        self.num_ms_success_first_time = -1
        self.num_elapsed_cycles = 0

    def update_cached_actuator_mujoco_data(self) -> None:
        """Update all cached information about MuJoCo actuators, such as names and ids."""
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='actuator', name_pattern='mover_actuator_x')
        self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='actuator', name_pattern='mover_actuator_y')

        self.mover_actuator_x_ids = np.zeros((len(self.mover_actuator_x_names),), dtype=np.int32)
        self.mover_actuator_y_ids = np.zeros((len(self.mover_actuator_x_names),), dtype=np.int32)
        for idx_a, actuator_x_name in enumerate(self.mover_actuator_x_names):
            self.mover_actuator_x_ids[idx_a] = self.model.actuator(actuator_x_name).id
            self.mover_actuator_y_ids[idx_a] = self.model.actuator(self.mover_actuator_y_names[idx_a]).id

        self.impedance_controller.update_cached_actuator_mujoco_data(self.model)

        # object
        object_joint_name = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='object')[0]
        self.object_joint_qpos_adr, _, self.object_joint_qpos_ndim, _ = mujoco_utils.get_joint_addrs_and_ndims(
            self.model, object_joint_name
        )

        # geom ids for collision check between mover and obstacle, and object and obstacle
        self.id_obstacle_0_geom = self.model.geom('obstacle_geom_0').id
        self.id_obstacle_1_geom = self.model.geom('obstacle_geom_1').id
        geom_names = mujoco_utils.MujocoModelNames(self.model).geom_names
        if 'bumper_geom_0' in geom_names:
            self.id_mover_geom = self.model.geom('bumper_geom_0').id
        else:
            self.id_mover_geom = self.model.geom('mover_geom_0').id
        self.id_object_geom = self.model.geom('object_geom').id

    def _update_obstacle_size(self, obstacle_mode: str | None = None) -> None:
        """Update the size of the obstacles, i.e. the vertices of the obstacles, depending on the current obstacle mode.

        :param obstacle_mode: the obstacle mode of this environment which determines the size of the obstacles and the difficulty of
            the task, defaults to None. Can be 'hard', 'medium', 'simple', 'random', or None. If None, ``self.obstacle_mode`` is used.
        :raises ValueError: if the obstacle mode is not 'hard', 'medium', 'simple', or 'random'
        """
        if obstacle_mode is None:
            om = self.obstacle_mode
        else:
            om = obstacle_mode

        if om == 'hard':
            self.obstacle_size_y = self.max_obstacle_size_y
            self.obstacle_0_y_pos = self.max_obstacle_size_y
            self.obstacle_1_y_pos = self.min_obstacle_1_y_pos
        elif om == 'medium':
            self.obstacle_size_y = self.medium_obstacle_size_y
            self.obstacle_0_y_pos = self.medium_obstacle_size_y
            self.obstacle_1_y_pos = self.medium_obstacle_1_y_pos
        elif om == 'simple':
            self.obstacle_size_y = self.min_obstacle_size_y
            self.obstacle_0_y_pos = self.min_obstacle_size_y
            self.obstacle_1_y_pos = self.max_obstacle_1_y_pos
        elif om == 'random':
            self.obstacle_size_y = self.np_random.uniform(low=self.min_obstacle_size_y, high=self.max_obstacle_size_y, size=1)[0]
            self.obstacle_0_y_pos = self.obstacle_size_y
            self.obstacle_1_y_pos = np.max(self.y_pos_tiles) + self.tile_size[1] - self.obstacle_size_y
        else:
            raise ValueError(f'Unkown obstacle mode: {om}')

        # obstacle vertices
        self.obstacle_0_vert_xy = np.array(
            [
                [self.obstacle_x_pos + self.obstacle_size_x, self.obstacle_0_y_pos + self.obstacle_size_y],
                [self.obstacle_x_pos + self.obstacle_size_x, self.obstacle_0_y_pos - self.obstacle_size_y],
                [self.obstacle_x_pos - self.obstacle_size_x, self.obstacle_0_y_pos - self.obstacle_size_y],
                [self.obstacle_x_pos - self.obstacle_size_x, self.obstacle_0_y_pos + self.obstacle_size_y],
            ]
        )

        self.obstacle_1_vert_xy = np.array(
            [
                [self.obstacle_x_pos + self.obstacle_size_x, self.obstacle_1_y_pos + self.obstacle_size_y],
                [self.obstacle_x_pos + self.obstacle_size_x, self.obstacle_1_y_pos - self.obstacle_size_y],
                [self.obstacle_x_pos - self.obstacle_size_x, self.obstacle_1_y_pos - self.obstacle_size_y],
                [self.obstacle_x_pos - self.obstacle_size_x, self.obstacle_1_y_pos + self.obstacle_size_y],
            ]
        )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """For each mover, this callback adds actuators to the ``custom_model_xml_strings``-dict, depending on whether the jerk or
        acceleration is the output of the policy.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
        # actuators
        if self.impedance_controller is not None:
            mover_actuator_list = ['\n\t<actuator>', '\t\t<!-- mover actuators -->']

            joint_name = self.mover_joint_names[self.idx_mover]
            if self.learn_jerk:
                mover_actuator_list.extend(
                    [
                        f'\t\t<general name="mover_actuator_x_{self.idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" '
                        f'dyntype="integrator" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none" actearly="true"/>',
                        f'\t\t<general name="mover_actuator_y_{self.idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                        f'dyntype="integrator" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none" actearly="true"/>',
                    ]
                )
            else:
                # learn acceleration
                mover_actuator_list.extend(
                    [
                        f'\t\t<general name="mover_actuator_x_{self.idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none"'
                        f' gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none"/>',
                        f'\t\t<general name="mover_actuator_y_{self.idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                        f'dyntype="none" gaintype="fixed" gainprm="{self.mover_mass} 0 0" biastype="none"/>',
                    ]
                )

            mover_actuator_list.append(self.impedance_controller.generate_actuator_xml_string(idx_mover=self.idx_mover))
            mover_actuator_list.append('\t</actuator>')

            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            mover_actuator_xml_str = '\n'.join(mover_actuator_list)
            if custom_outworldbody_xml_str is not None:
                custom_outworldbody_xml_str += mover_actuator_xml_str
            else:
                custom_outworldbody_xml_str = mover_actuator_xml_str
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        # object and goal
        custom_object_xml_str = (
            '\n\t\t<!-- object -->'
            + f'\n\t\t<body name="object" pos="{self.object_xy_start_pos[0]} {self.object_xy_start_pos[1]} {self.object_height}">'
            + '\n\t\t\t<joint name="object_joint" type="free" damping="0.01"/>'
            + '\n\t\t\t<geom name="object_geom" '
            + f'type="box" size="{self.object_length_xy} {self.object_length_xy} {self.object_height}" '
            + f'mass="{self.object_mass}" material="light_green" friction="0.6 0.0001 0.0001"/>'
            + '\n\t\t</body>'
            + '\n\t\t<site name="object_goal_site" type="sphere" material="light_green" size="0.02" '
            + f'pos="{self.object_xy_goal_pos[0]} {self.object_xy_goal_pos[1]} {self.object_height}"/>'
        )

        # obstacles
        custom_object_xml_str += (
            '\n\n\t\t<!-- obstacles -->'
            + f'\n\t\t<body name="obstacle_0" pos="{self.obstacle_x_pos} {self.obstacle_0_y_pos} 0.03" gravcomp="1">'
            + f'\n\t\t\t<geom name="obstacle_geom_0" type="box" size="{self.obstacle_size_x} {self.obstacle_size_y} 0.03" '
            + 'mass="100" pos="0 0 0" material="red"/>'
            + '\n\t\t</body>'
            + f'\n\n\t\t<body name="obstacle_1" pos="{self.obstacle_x_pos} {self.obstacle_1_y_pos} 0.03" gravcomp="1">'
            + f'\n\t\t\t<geom name="obstacle_geom_1" type="box" size="{self.obstacle_size_x} {self.obstacle_size_y} 0.03" '
            + 'mass="100" pos="0 0 0" material="red"/>'
            + '\n\t\t</body>'
        )

        custom_worldbody_xml_str = custom_model_xml_strings.get('custom_worldbody_xml_str', None)
        if custom_worldbody_xml_str is not None:
            custom_worldbody_xml_str += custom_object_xml_str
        else:
            custom_worldbody_xml_str = custom_object_xml_str
        custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate a new model XML string with new start positions for mover and object and a new object goal position and reload the
        model. In this environment, it is necessary to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover,
            defaults to None. If set to None, the movers will be placed in the center of the tiles that are added to the XML string
            first.
        """
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # update cached mujoco data
        self.update_cached_mover_mujoco_data()
        self.update_cached_actuator_mujoco_data()

        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the start position of mover and object and the object goal position and reload the model. It is ensured that the
        new start position of the mover is collision-free, i.e. no wall collision and no collision with the object.
        In addition, the object's start position is chosen such that the mover fits between the wall and the object. This is important
        to ensure that the object can be pushed in all directions.

        It is also possible to explicitly specify start positions for mover and object, as well as the object's goal position by using
        the keys 'mover_xy_start_pos', 'object_xy_start_pos', and 'object_xy_goal_pos' in the ``options``-dict (each numpy arrays
        of shape (2,)). The start position of mover and object are sampled uniformly at random if the corresponding key is not in the
        ``options``-dict.  Note that there is no check whether mover or object collide with an obstacle due to computational
        efficiency. Since min and max x,y start positions of object and mover are chosen such that a collision with the obstacle
        cannot occur if the start positions are randomly sampled. In addition, if the object's start position is specified by the user,
        the distance to the mover is not checked.

        :param options: can be used to specify start positions for mover and object and a goal position for the object
            (keys: 'mover_xy_start_pos', 'object_xy_start_pos', 'object_xy_goal_pos')
        """
        if options is None:
            options = {}
        # sample new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # quaternion (1,0,0,0)

        # choose a new start position for the mover
        start_qpos[:, :2] = options.get(
            'mover_xy_start_pos',
            self.np_random.uniform(low=self.min_mover_start_xy_pos, high=self.max_mover_start_xy_pos, size=(self.num_movers, 2)),
        )

        if 'object_xy_start_pos' in options.keys():
            self.object_xy_start_pos = options['object_xy_start_pos']
        else:
            # sample a new start position for the object and ensure that it does not collide with the mover
            counter = 0
            dist_start_valid = False
            while not dist_start_valid:
                counter += 1
                if counter > 0 and counter % 100 == 0:
                    logger.warn(
                        'Trying to find a start position for the object.'
                        + f'No valid configuration found within {counter} trails. Consider choosing more tiles.'
                    )
                self.object_xy_start_pos = self.np_random.uniform(
                    low=self.min_object_start_xy_pos, high=self.max_object_start_xy_pos, size=(1, 2)
                )
                # check distance between object and mover
                dist_start_valid = (np.linalg.norm(self.object_xy_start_pos - start_qpos[:, :2], ord=2, axis=1) > self.min_mo_dist).all()

            self.object_xy_start_pos = self.object_xy_start_pos.flatten()

        # sample a new goal position for the object or set a pre-defined goal position
        self.object_xy_goal_pos = options.get(
            'object_xy_goal_pos', self.np_random.uniform(low=self.min_goal_xy_pos, high=self.max_goal_xy_pos, size=(2,))
        )

        # curriculum learning
        self.ep_counter += 1
        if self.obstacle_mode == 'curriculum' and self.ep_counter > self.cl_num_ep_before_incr:
            self.ep_counter = 1
            self.tmp_obstacle_mode = 'medium' if self.tmp_obstacle_mode == 'simple' else 'hard'
        # update obstacle size if obstacle mode == "random" or obstacle_mode == "curriculum"
        if self.obstacle_mode == 'random':
            self._update_obstacle_size()
        elif self.obstacle_mode == 'curriculum':
            self._update_obstacle_size(obstacle_mode=self.tmp_obstacle_mode)

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2])

        # reset corrective movement measurement
        self.cm_measurement.reset()

        # reset throughput measurement
        self.num_ms_success_first_time = -1
        self.num_elapsed_cycles = 0

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
            _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=next_acc_tmp)
            if (next_acc_tmp != next_acc).any():
                next_jerk = (next_acc - acc) / self.cycle_time
            ctrl = next_jerk.copy()
        else:
            _, next_acc = self.ensure_max_dyn_val(current_values=vel, max_value=self.v_max, next_derivs=action)
            ctrl = next_acc.copy()

        self.data.ctrl[self.mover_actuator_x_ids] = ctrl[:, 0]
        self.data.ctrl[self.mover_actuator_y_ids] = ctrl[:, 1]

        # update impedance controller
        self.impedance_controller.update(
            model=self.model,
            data=self.data,
            pos_d=np.array([0, 0, self.initial_mover_zpos + self.mover_size[0, 2]]),
            quat_d=np.array([1, 0, 0, 0]),
        )

    def _after_mujoco_step_callback(self):
        """Check whether corrective movements (overshoot or distance corrections) occurred and increase the corresponding counter if
        necessary.
        """
        current_object_pose = self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + self.object_joint_qpos_ndim][:2]
        object_target_pose = self.object_xy_goal_pos.copy()

        # corrective movements
        self.cm_measurement.update_distance_corrections(current_object_pose=current_object_pose, object_target_pose=object_target_pose)
        self.cm_measurement.update_overshoot_corrections(current_object_pose=current_object_pose, object_target_pose=object_target_pose)

        # throughput
        if self.num_ms_success_first_time == -1:
            self.num_elapsed_cycles += 1
            dist = self._calc_eucl_dist_xy(achieved_goal=current_object_pose, desired_goal=object_target_pose)
            assert dist.shape == (1,)
            is_success = bool((dist <= self.threshold_pos)[0])
            if is_success:
                self.num_ms_success_first_time = self.num_elapsed_cycles

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. A state is terminal when the mover or object collides with a wall or the
        obstacle.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,). An entry is True if the state is terminal, False otherwise
            - if batch_size = 1:
                True if the state is terminal, False otherwise
        """
        reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
        terminated = reward == self.reward_collision
        return terminated

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. The truncation condition (a time limit in this environment) is
        automatically checked by the Gymnasium TimeLimit Wrapper, which is why this method always returns False.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return:

            - if batch_size > 1:
                a numpy array of shape (batch_size,) in which all entries are False
            - if batch_size = 1:
                False
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        return np.array([False] * batch_size) if batch_size > 1 else False

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray | float:
        """Compute the immediate reward.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        batch_size, _, wall_collisions, obstacle_object_collisions, obstacle_mover_collisions = self._preprocess_info_dict(info=info)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        mask_no_wall_collision = np.bitwise_not(wall_collisions)
        mask_no_obstacle_collision = np.bitwise_and(np.bitwise_not(obstacle_object_collisions), np.bitwise_not(obstacle_mover_collisions))
        mask_no_collision = np.bitwise_and(mask_no_wall_collision, mask_no_obstacle_collision)
        mask_collision = np.bitwise_not(mask_no_collision)

        dist_goal = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal)
        assert dist_goal.shape == (batch_size,)
        mask_goal_reached = dist_goal <= self.threshold_pos

        reward = self.reward_collision * mask_collision.astype(np.float64)
        reward += -1.0 * mask_no_collision.astype(np.float64)
        reward[np.bitwise_and(mask_goal_reached, mask_no_collision)] = 0

        assert reward.shape == (batch_size,)
        return reward if batch_size > 1 else reward[0]

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a dictionary containing the following keys and values:

            - 'observation':

                - if ``learn_jerk=True``:
                    a numpy array of shape (22,) containing the (x,y)-position, (x,y)-velocities, and (x,y)-accelerations of the mover,
                    as well as the x and y positions of the vertices of both obstacles
                - if ``learn_jerk=False``:
                    a numpy array of shape (20,) containing the (x,y)-position, (x,y)-velocities, and (x,y)-accelerations of the mover,
                    as well as the x and y positions of the vertices of both obstacles
            - 'achieved_goal':
                a numpy array of shape (2,) containing the current (x,y)-position of the object
            - 'desired_goal':
                a numpy array of shape (2,) containing the desired (x,y)-position of the object
        """
        # observation
        mover_xy_pos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=True)[:, :2]
        mover_xy_velos = self.get_mover_qvel(mover_names=self.mover_names, add_noise=True)[:, :2]
        if self.learn_jerk:
            # no noise, because only SetAcc is available in a real system
            mover_xy_accs = self.get_mover_qacc(mover_names=self.mover_names, add_noise=False)[:, :2]
            observation = np.concatenate(
                (mover_xy_pos, mover_xy_velos, mover_xy_accs, self.obstacle_0_vert_xy, self.obstacle_1_vert_xy), axis=0
            )
        else:
            observation = np.concatenate((mover_xy_pos, mover_xy_velos, self.obstacle_0_vert_xy, self.obstacle_1_vert_xy), axis=0)

        # achieved goal
        object_xy_pos = self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + self.object_joint_qpos_ndim][:2]
        achieved_goal = object_xy_pos + self.rng_noise.normal(loc=0.0, scale=self.object_noise_xy_pos, size=2)

        # desired goal
        desired_goal = self.object_xy_goal_pos.copy()

        return OrderedDict(
            [
                ('observation', observation.flatten()),
                ('achieved_goal', achieved_goal),
                ('desired_goal', desired_goal),
            ]
        )

    def _check_for_other_collisions_callback(self):
        """Check whether the object or mover collide with an obstacle.

        :return:
            - whether there is a collision (mover-obstacle or object-obstacle, bool)
            - a dictionary with keys ``obstacle_object_collision`` and ``obstacle_mover_collision`` that contains bool values
              indicating what type of collision occurred (mover-obstacle or object-obstacle)
        """
        mask_contacts_obstacle_0_geom = (self.data.contact.geom1 == self.id_obstacle_0_geom).astype(np.int8) + (
            self.data.contact.geom2 == self.id_obstacle_0_geom
        ).astype(np.int8)
        mask_contacts_obstacle_1_geom = (self.data.contact.geom1 == self.id_obstacle_1_geom).astype(np.int8) + (
            self.data.contact.geom2 == self.id_obstacle_1_geom
        ).astype(np.int8)
        mask_contacts_object_geom = (self.data.contact.geom1 == self.id_object_geom).astype(np.int8) + (
            self.data.contact.geom2 == self.id_object_geom
        ).astype(np.int8)
        mask_contacts_mover_geom = (self.data.contact.geom1 == self.id_mover_geom).astype(np.int8) + (
            self.data.contact.geom2 == self.id_mover_geom
        ).astype(np.int8)

        obstacle_0_object_collision = ((mask_contacts_obstacle_0_geom + mask_contacts_object_geom) == 2).any()
        obstacle_0_mover_collision = ((mask_contacts_obstacle_0_geom + mask_contacts_mover_geom) == 2).any()

        obstacle_1_object_collision = ((mask_contacts_obstacle_1_geom + mask_contacts_object_geom) == 2).any()
        obstacle_1_mover_collision = ((mask_contacts_obstacle_1_geom + mask_contacts_mover_geom) == 2).any()

        obstacle_object_collision = obstacle_0_object_collision or obstacle_1_object_collision
        obstacle_mover_collision = obstacle_0_mover_collision or obstacle_1_mover_collision

        obstacle_collision = obstacle_object_collision or obstacle_mover_collision
        collision_info = {'obstacle_object_collision': obstacle_object_collision, 'obstacle_mover_collision': obstacle_mover_collision}

        return obstacle_collision, collision_info

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
        :param other_collision: whether there are collisions with the obstacle
        :param achieved_goal: a numpy array of shape (length achieved_goal,) containing the already achieved (x,y)-position of the
            object
        :param desired_goal: a numpy array of shape (length achieved_goal,) containing the desired (x,y)-position of the object
        :param collision_info: a dictionary that contains information about whether the mover or the object collided with the obstacle
        :return: the info dictionary with keys 'is_success', 'mover_collision', 'wall_collision', and 'dist' (Euclidean distance to the
            object's goal position)
        """
        assert not mover_collision
        dist = self._calc_eucl_dist_xy(achieved_goal=achieved_goal, desired_goal=desired_goal).flatten()
        assert dist.shape == (1,)
        is_success = bool((dist <= self.threshold_pos)[0]) and not bool(wall_collision) and not bool(other_collision)
        assert not isinstance(is_success, np.ndarray)
        assert not isinstance(mover_collision, np.ndarray)
        assert not isinstance(wall_collision, np.ndarray)
        assert not isinstance(other_collision, np.ndarray)
        info = {
            'is_success': is_success,
            'mover_collision': mover_collision,
            'wall_collision': wall_collision,
            'dist': dist,
            'num_overshoot_corrections': self.get_current_num_overshoot_corrections(),
            'num_distance_corrections': self.get_current_num_distance_corrections(),
            'num_ms_success_first_time': self.num_ms_success_first_time,
        }
        info.update(collision_info)
        return info

    def close(self) -> None:
        """Close the environment."""
        super().close()

    def ensure_max_dyn_val(self, current_values: np.ndarray, max_value: float, next_derivs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensure the minimum and maximum dynamic values.

        :param current_values: the current velocity or acceleration specified as a numpy array of shape (2,) or (num_checks,2)
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
                np.broadcast_to(norm_next_values_tmp[mask_norm, np.newaxis], (np.sum(mask_norm), 2)),
            )
            next_derivs_new[mask_norm] = (next_values[mask_norm] - current_values[mask_norm]) / self.cycle_time

        return next_values, next_derivs_new

    def _calc_eucl_dist_xy(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """Calculate the Euclidean distance.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :return: a numpy array of shape (batch_size,), which contains the distances between the achieved and the desired goals
        """
        batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        return np.linalg.norm(achieved_goal - desired_goal, ord=2, axis=1)

    def _preprocess_info_dict(self, info: np.ndarray | dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract information about mover collisions, wall collisions and the batch size from the info dictionary.

        :param info: the info dictionary or an array of info dictionary to be preprocessed. All dictionaries must contain the keys
            ``mover_collision``, ``wall_collision``, ``obstacle_object_collision``, and ``obstacle_mover_collision``.
        :return:
            - the batch_size (int)
            - a numpy array of shape (batch_size,) containing the mover collision values (bool)
            - a numpy array of shape (batch_size,) containing the wall collision values (bool)
            - a numpy array of shape (batch_size,) containing the obstacle-object collision values (bool)
            - a numpy array of shape (batch_size,) containing the obstacle-mover collision values (bool)
        """
        if isinstance(info, np.ndarray):
            batch_size = info.shape[0]
            mover_collisions = np.zeros(batch_size).astype(bool)
            wall_collisions = np.zeros(batch_size).astype(bool)
            obstacle_object_collisions = np.zeros(batch_size).astype(bool)
            obstacle_mover_collisions = np.zeros(batch_size).astype(bool)

            for i in range(0, batch_size):
                mover_collisions[i] = info[i]['mover_collision']
                wall_collisions[i] = info[i]['wall_collision']
                obstacle_object_collisions[i] = info[i]['obstacle_object_collision']
                obstacle_mover_collisions[i] = info[i]['obstacle_mover_collision']
        else:
            assert isinstance(info, dict)
            batch_size = 1
            mover_collisions = np.array([info['mover_collision']])
            wall_collisions = np.array([info['wall_collision']])
            obstacle_object_collisions = np.array([info['obstacle_object_collision']])
            obstacle_mover_collisions = np.array([info['obstacle_mover_collision']])

        return batch_size, mover_collisions, wall_collisions, obstacle_object_collisions, obstacle_mover_collisions

    def get_current_num_overshoot_corrections(self) -> int:
        """Return the current number of overshoot corrections measured within the current episode.

        :return: the current number of overshoot corrections measured within the current episode
        """
        return self.cm_measurement.get_current_num_overshoot_corrections()

    def get_current_num_distance_corrections(self) -> int:
        """Return the current number of distance corrections measured within the current episode.

        :return: the current number of distance corrections measured within the current episode
        """
        return self.cm_measurement.get_current_num_distance_corrections()

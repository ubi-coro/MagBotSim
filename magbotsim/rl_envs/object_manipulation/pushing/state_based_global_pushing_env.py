############################################################################
# Copyright (c) 2024 Cedric Grothues & Lara Bergmann, Bielefeld University #
############################################################################

from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import shapely
import shapely.geometry as sg
from gymnasium import logger

from magbotsim import BasicMagBotSingleAgentEnv, MoverImpedanceController
from magbotsim.utils import mujoco_utils, benchmark_utils

DEFAULT_OBJECT_TYPES = [
    'square_box',
    'box',
    'cylinder',
    't_shape',
    'l_shape',
    'plus_shape',
]
DEFAULT_OBJECT_RANGES = {
    'width': (0.06, 0.06),  # [m]
    'height': (0.06, 0.06),  # [m]
    'depth': (0.02, 0.02),  # [m]
    'segment_width': (0.02, 0.02),  # [m]
    'radius': (0.05, 0.05),  # [m]
    'mass': (0.3, 0.3),  # [kg]
}


class StateBasedGlobalPushingEnv(BasicMagBotSingleAgentEnv):
    """An object pushing environment.

    :param num_movers: the number of movers in the environment, defaults to 1
    :param mover_params: a dictionary that can be used to specify the mass and size of each mover using the keys 'mass' or 'size',
        defaults to None. To use the same mass and size for each mover, the mass can be specified as a single float value and the
        size as a numpy array of shape (3,). However, the movers can also be of different types, i.e. different masses and sizes.
        In this case, the mass and size should be specified as numpy arrays of shapes (num_movers,) and (num_movers,3),
        respectively. If set to None or only one key is specified, both mass and size or the missing value are set to the following
        default values:

        - mass: 1.24 [kg]
        - size: [0.155/2, 0.155/2, 0.012/2] (x,y,z) [m] (note: half-size)
    :param layout_tiles: a numpy array of shape (height, width) that specifies the layout of the tiles, defaults to None. If None, a
        4x3 grid of tiles is used.
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.003
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle: whether to call 'render' after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if ``render_mode != None``.
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

    :param object_types: a sequence of strings that specifies which object types can be used, defaults to DEFAULT_OBJECT_TYPES.
        Valid object types are: 'square_box', 'box', 'cylinder', 't_shape', 'l_shape', 'plus_shape'
    :param object_ranges: a dictionary that specifies the range of object parameters for random sampling, defaults to
        DEFAULT_OBJECT_RANGES.
        Keys can include 'width', 'height', 'depth', 'segment_width', 'radius', 'mass'. Each value is a tuple (min_value, max_value)
    :param object_sliding_friction: the sliding friction coefficient of the object, defaults to 1.0
    :param object_torsional_friction: the torsional friction coefficient of the object, defaults to 0.005
    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk (only used if 'learn_jerk=True'), defaults to 100.0 [m/s³]
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param learn_pose: whether to learn both position and orientation of the object, defaults to False. If set to False, only
        position is learned.
    :param early_termination_steps: the number of consecutive steps at goal after which the episode terminates early, defaults to None
        (no early termination)
    :param max_position_err: the position threshold used to determine whether the object has reached its goal position, defaults
        to 0.05 [m]
    :param min_coverage: the minimum coverage ratio for goal achievement when learn_pose=True, defaults to 0.9
    :param collision_penalty: the reward penalty applied when a collision occurs, defaults to -10.0
    :param per_step_penalty: the small negative reward applied at each time step to encourage efficiency, defaults to -0.01
    :param object_at_goal_reward: the positive reward given when the object reaches the goal without collisions, defaults to 1.0
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        num_movers: int = 1,
        mover_params: dict[str, Any] | None = None,
        layout_tiles: np.ndarray | None = None,
        initial_mover_zpos: float = 0.003,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        object_type: str = 'square_box',
        object_ranges: MutableMapping[str, tuple[float, float]] = DEFAULT_OBJECT_RANGES,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        object_sliding_friction: float = 1.0,
        object_torsional_friction: float = 0.005,
        learn_jerk: bool = False,
        learn_pose: bool = False,
        early_termination_steps: int | None = None,
        max_position_err: float = 0.05,
        min_coverage: float = 0.9,
        collision_penalty: float = -10.0,
        per_step_penalty: float = -0.01,
        object_at_goal_reward: float = 1.0,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        self.num_movers = num_movers
        self.learn_jerk = learn_jerk
        self.learn_pose = learn_pose
        self.early_termination_steps = early_termination_steps

        # tile configuration
        if layout_tiles is None:
            layout_tiles = np.ones((4, 3))

        # position threshold in m
        self.max_position_err = max_position_err
        self.min_coverage = min_coverage

        # rewards
        self.collision_penalty = collision_penalty
        self.per_step_penalty = per_step_penalty
        self.object_at_goal_reward = object_at_goal_reward
        self.steps_at_goal = 0

        # object parameters
        self.object_xy_start_pos = np.array([0.12, 0.36])
        self.object_xy_goal_pos = np.array([0.36, 0.36])
        self.object_noise_xy_pos = 1e-5
        self.object_noise_yaw = 1e-5
        self.object_start_yaw = 0.0
        self.object_goal_yaw = 0.0

        self.object_sliding_friction = object_sliding_friction
        self.object_torsional_friction = object_torsional_friction

        self.impedance_controllers = None

        assert isinstance(object_type, str) and object_type in DEFAULT_OBJECT_TYPES
        assert isinstance(object_ranges, MutableMapping) and all(len(value) == 2 for value in object_ranges.values())

        self.object_type = object_type
        self.object_ranges = object_ranges

        self.object_width = 0
        self.object_height = 0
        self.object_depth = 0
        self.object_radius = 0
        self.object_segment_width = 0
        self.object_mass = 0

        self._sample_object_params(options={})

        # cam config
        default_cam_config = {
            'distance': 0.8,
            'azimuth': 160.0,
            'elevation': -55.0,
            'lookat': np.array([0.8, 0.2, 0.4]),
        }

        super().__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
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

        # observation space
        max_x = np.max(self.x_pos_tiles) + (self.tile_size[0] / 2)
        max_y = np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)

        if not self.learn_pose:
            low_goals = np.zeros(2)
            high_goals = np.array([max_x, max_y])
        else:
            low_goals = np.array([0, 0, -1, -1])
            high_goals = np.array([max_x, max_y, 1, 1])

        self.observation_space = gym.spaces.Dict(
            {
                'observation': gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_movers * (2 + int(self.learn_jerk)) * 2,),
                    dtype=np.float64,
                ),
                'achieved_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
                'desired_goal': gym.spaces.Box(low=low_goals, high=high_goals, dtype=np.float64),
            }
        )

        action_scale = self.j_max if self.learn_jerk else self.a_max
        self.action_space = gym.spaces.Box(low=-action_scale, high=action_scale, shape=(self.num_movers * 2,), dtype=np.float64)

        # minimum and maximum possible mover (x,y)-positions
        safety_margin = self.c_size + self.c_size_offset_wall + self.c_size_offset + 0.03
        self.min_xy_pos = np.zeros(2) + safety_margin
        self.max_xy_pos = (
            np.array([np.max(self.x_pos_tiles) + (self.tile_size[0] / 2), np.max(self.y_pos_tiles) + (self.tile_size[1] / 2)])
            - safety_margin
        )

        # minimum and maximum possible object (x,y)-positions
        actual_safety_margin = safety_margin + 0.02

        self.object_min_xy_pos = self.min_xy_pos + actual_safety_margin
        self.object_max_xy_pos = self.max_xy_pos - actual_safety_margin

        self.object_goal_min_xy_pos = self.min_xy_pos + actual_safety_margin
        self.object_goal_max_xy_pos = self.max_xy_pos - actual_safety_margin

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

        self.reload_model()

        # minimum distance between object and mover after env reset
        max_side = self.object_radius if self.object_type == 'cylinder' else max(self.object_width, self.object_height)
        max_size = self.mover_size.max(axis=0) if self.mover_size.ndim == 2 else self.mover_size

        side_coll = np.linalg.norm(max_side + max_size[:2], ord=2)

        c_offset = self.c_size + self.c_size_offset if self.c_shape == 'circle' else np.linalg.norm(self.c_size + self.c_size_offset, ord=2)

        self.min_mover_object_dist = np.maximum(side_coll, c_offset)
        self.min_object_goal_dist = 0.15

        # corrective movements
        self.cm_measurement = benchmark_utils.CorrectiveMovementMeasurement(
            distance_func=self._calc_eucl_dist_xy, threshold=self.max_position_err
        )

        # throughput
        self.time_to_success_s = -1
        self.num_elapsed_cycles = 0
        self.success_counter = 0

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

        # object
        object_joint_name = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='object')[0]
        self.object_joint_qpos_adr, _, self.object_joint_qpos_ndim, _ = mujoco_utils.get_joint_addrs_and_ndims(
            self.model, object_joint_name
        )

    def _object_geom_xml(self) -> str:
        """Generate MuJoCo XML geometry definition for the current object.

        :return: a string containing the MuJoCo XML geometry definition
        """

        def box_volume(width: float, height: float, depth: float) -> float:
            """Calculate the volume of a box geometry.

            :param width: the width (x-direction half-size) of the box
            :param height: the height (y-direction half-size) of the box
            :param depth: the depth (z-direction half-size) of the box
            :return: the volume of the box in cubic meters (accounts for half-sizes by factor of 8)
            """
            return 8 * width * height * depth

        def object_masses(shape: str) -> tuple[float, float]:
            """Calculate mass distribution for object components.

            For simple shapes (box, cylinder), returns all mass to the main component.
            For complex shapes, distributes mass proportionally based on volume.

            :param shape: the object shape type
            :return: a tuple of (main_body_mass, segment_mass)
            """
            if shape in {'square_box', 'box', 'cylinder'}:
                return (self.object_mass, 0)

            b_volume = box_volume(self.object_segment_width, self.object_height, self.object_depth)
            s_volume = box_volume(segment_width_half, self.object_segment_width, self.object_depth)
            volume = b_volume + {'t_shape': 2, 'plus_shape': 2, 'l_shape': 1}[shape] * s_volume

            return (
                self.object_mass * (b_volume / volume),
                self.object_mass * (s_volume / volume),
            )

        def box_geom(width: float, height: float, depth: float, mass: float, x: float = 0, y: float = 0, z: float = 0) -> str:
            """Generate XML for a box geometry.

            :param width: the width (half-size) of the box
            :param height: the height (half-size) of the box
            :param depth: the depth (half-size) of the box
            :param mass: the mass of the box
            :param x: the x-position offset from parent body center, defaults to 0
            :param y: the y-position offset from parent body center, defaults to 0
            :param z: the z-position offset from parent body center, defaults to 0
            :return: a string containing the MuJoCo XML for the box geometry
            """
            pos_str = f' pos="{x} {y} {z}"' if any([x, y, z]) else ''
            return f'\t\t\t<geom type="box" size="{width} {height} {depth}" mass="{mass}" {pos_str} />'

        def cylinder_geom(radius: float, depth: float, mass: float) -> str:
            """Generate XML for a cylinder geometry.

            :param radius: the radius of the cylinder
            :param depth: the height (half-size) of the cylinder
            :param mass: the mass of the cylinder
            :return: a string containing the MuJoCo XML for the cylinder geometry
            """
            return f'\t\t\t<geom type="cylinder" size="{radius} {depth}" mass="{mass}" />'

        segment_width_half = (self.object_width - self.object_segment_width) / 2
        segment_pos = segment_width_half + self.object_segment_width
        masses = object_masses(self.object_type)

        match self.object_type:
            case 'square_box':
                return box_geom(self.object_width, self.object_width, self.object_depth, masses[0])
            case 'box':
                return box_geom(self.object_width, self.object_height, self.object_depth, masses[0])
            case 'cylinder':
                return cylinder_geom(self.object_radius, self.object_depth, masses[0])
            case 't_shape':
                return '\n'.join(
                    [
                        box_geom(self.object_segment_width, self.object_height, self.object_depth, masses[0]),
                        box_geom(
                            segment_width_half,
                            self.object_segment_width,
                            self.object_depth,
                            masses[1],
                            -segment_pos,
                            self.object_height - self.object_segment_width,
                            0,
                        ),
                        box_geom(
                            segment_width_half,
                            self.object_segment_width,
                            self.object_depth,
                            masses[1],
                            segment_pos,
                            self.object_height - self.object_segment_width,
                            0,
                        ),
                    ]
                )
            case 'l_shape':
                return '\n'.join(
                    [
                        box_geom(self.object_segment_width, self.object_height, self.object_depth, masses[0]),
                        box_geom(
                            segment_width_half,
                            self.object_segment_width,
                            self.object_depth,
                            masses[1],
                            segment_pos,
                            -1 * (self.object_height - self.object_segment_width),
                            0,
                        ),
                    ]
                )
            case 'plus_shape':
                return '\n'.join(
                    [
                        box_geom(self.object_segment_width, self.object_height, self.object_depth, masses[0]),
                        box_geom(segment_width_half, self.object_segment_width, self.object_depth, masses[1], -segment_pos, 0, 0),
                        box_geom(segment_width_half, self.object_segment_width, self.object_depth, masses[1], segment_pos, 0, 0),
                    ]
                )
            case shape:
                raise ValueError(f'Unsupported object shape {shape}')

    @property
    def _object_xml(self) -> str:
        """Generate the complete MuJoCo XML definition for the object body.

        :return: a string containing the complete MuJoCo XML body definition including joint and geometry
        """
        geom_xml = self._object_geom_xml()
        goal_xml = geom_xml if self.learn_pose else '\n\t\t\t<geom name="object_goal" type="sphere" material="red" size="0.02" />'

        return (
            f'\n\t\t<!-- object -->\n\t\t<body name="object" '
            f'pos="{self.object_xy_start_pos[0]} {self.object_xy_start_pos[1]} {self.object_depth}" '
            f'euler="0 0 {self.object_start_yaw}" childclass="object">'
            f'\n\t\t\t<joint name="object_joint" type="free" damping="0.01" />'
            f'\n{geom_xml}\n\t\t</body>\n\t\t<body name="object_goal" '
            f'pos="{self.object_xy_goal_pos[0]} {self.object_xy_goal_pos[1]} {self.object_depth}" '
            f'euler="0 0 {self.object_goal_yaw}" childclass="object_goal">\n{goal_xml}\n\t\t</body>'
        )

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict | None) -> dict[str, str]:
        """Generate custom XML strings for the environment model.

        This method is called during model initialization to add custom objects and modifications
        to the base MuJoCo model.

        :return: a dict of XML strings to be inserted into the model
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}

        # actuators
        if self.impedance_controllers is not None:
            mover_actuator_list = ['\n\t<actuator>', '\t\t<!-- mover actuators -->']

            for idx_mover in range(self.num_movers):
                joint_name = self.mover_joint_names[idx_mover]
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
                    mover_actuator_list.extend(
                        [
                            f'\t\t<general name="mover_actuator_x_{idx_mover}" joint="{joint_name}" gear="1 0 0 0 0 0" dyntype="none" '
                            f'gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>',
                            f'\t\t<general name="mover_actuator_y_{idx_mover}" joint="{joint_name}" gear="0 1 0 0 0 0" '
                            f'dyntype="none" gaintype="fixed" gainprm="{mover_mass} 0 0" biastype="none"/>',
                        ]
                    )

                impedance_controller = self.impedance_controllers[idx_mover]
                mover_actuator_list.append(impedance_controller.generate_actuator_xml_string(idx_mover=idx_mover))

            mover_actuator_list.append('\t</actuator>')

            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            mover_actuator_xml_str = '\n'.join(mover_actuator_list)
            if custom_outworldbody_xml_str is not None:
                custom_outworldbody_xml_str += mover_actuator_xml_str
            else:
                custom_outworldbody_xml_str = mover_actuator_xml_str
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        custom_model_xml_strings['custom_default_xml_str'] = (
            '\n\t\t<default class="object">\n\t\t\t<geom material="red" group="2" '
            f'friction="{self.object_sliding_friction} {self.object_torsional_friction} 0.0001" priority="1" />\n\t\t</default>'
            '\n\t\t<default class="object_goal">\n\t\t\t<geom material="red_transparent" contype="0" conaffinity="0" group="1" />'
            '\n\t\t</default>'
        )

        # object
        custom_worldbody_xml_str = custom_model_xml_strings.get('custom_worldbody_xml_str', None)
        if custom_worldbody_xml_str is not None:
            custom_worldbody_xml_str += self._object_xml
        else:
            custom_worldbody_xml_str = self._object_xml
        custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate a new model XML string with new start positions for mover and object and a new object goal position and reload the
        model. In this environment, it is necessary to reload the model to ensure that the actuators work as expected.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover,
            defaults to None. If set to None, the movers will be placed in the center of the tiles that are added to the XML string
            first.
        """
        custom_model_xml_strings = self._custom_xml_string_callback(
            custom_model_xml_strings=self.custom_model_xml_strings_before_cb,
        )
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos,
            mover_goal_xy_pos=None,
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
        self.render()

    def _sample_object_params(self, options: dict[str, Any]) -> None:
        """Sample random object parameters from the configured ranges.

        :param options: a dictionary of parameter overrides. Keys can include 'object_type', 'object_start_yaw',
            'object_goal_yaw', and any object property like 'object_width', 'object_height', etc.
        """
        self.object_start_yaw = options.get('object_start_yaw', self.np_random.uniform(low=-np.pi, high=np.pi))
        self.object_goal_yaw = options.get('object_goal_yaw', self.np_random.uniform(low=-np.pi, high=np.pi))

        for prop in ['width', 'height', 'depth', 'radius', 'segment_width', 'mass']:
            setattr(self, f'object_{prop}', options.get(f'object_{prop}', self.np_random.uniform(*self.object_ranges[prop])))

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """Reset the start positions of the movers and object and the object goal position and reload the model. It is ensured that the
        new start positions of the movers are collision-free, i.e. no wall collision, no collision with other movers and no collision
        with the object. In addition, the object's start position is chosen such that the mover fits between the wall and the object.
        This is important to ensure that the object can be pushed in all directions.

        :param options: can be used to override mover and object parameters
        """
        options = options or {}

        # sample new mover start positions
        start_qpos = np.zeros((self.num_movers, 7))
        start_qpos[:, 2] = self.initial_mover_zpos
        start_qpos[:, 3] = 1  # Quaternion (1,0,0,0)

        self.object_xy_goal_pos = options.get(
            'object_xy_goal_pos',
            self.np_random.uniform(low=self.object_goal_min_xy_pos, high=self.object_goal_max_xy_pos, size=(2,)),
        )

        self._sample_object_params(options)

        if 'mover_xy_start_pos' in options.keys():
            start_qpos[:, :2] = options['mover_xy_start_pos']

        if 'object_xy_start_pos' in options.keys():
            self.object_xy_start_pos = options['object_xy_start_pos']
        else:
            # sample a new start position for the object and ensure that it does not collide with the mover
            cnt = 0
            all_ok = False
            while not all_ok:
                cnt += 1
                if cnt > 0 and cnt % 100 == 0:
                    logger.warn(
                        f'Trying to find a start position for the object. No valid configuration found within {cnt} trails. '
                        'Consider choosing more tiles.'
                    )

                # choose a new start position for the mover
                if 'mover_xy_start_pos' in options.keys():
                    start_qpos[:, :2] = options['mover_xy_start_pos']
                else:
                    start_qpos[:, :2] = self.np_random.uniform(low=self.min_xy_pos, high=self.max_xy_pos, size=(self.num_movers, 2))
                pos_ok = self.qpos_is_valid(qpos=start_qpos, c_size=self.c_size, add_safety_offset=True)

                mover_collision = self.check_mover_collision(
                    mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=True, mover_qpos=start_qpos
                )

                self.object_xy_start_pos = self.np_random.uniform(low=self.object_min_xy_pos, high=self.object_max_xy_pos, size=(2,))
                mover_object_dist_ok = (
                    np.linalg.norm(self.object_xy_start_pos - start_qpos[:, :2], ord=2, axis=1) > self.min_mover_object_dist
                )
                object_goal_dist_ok = np.linalg.norm(self.object_xy_start_pos - self.object_xy_goal_pos, ord=2) > self.min_object_goal_dist
                all_ok = not mover_collision and pos_ok.all() and mover_object_dist_ok.all() and object_goal_dist_ok

        self.object_xy_start_pos = self.object_xy_start_pos.flatten()
        self.steps_at_goal = 0

        # reload model with new start pos and goal pos
        self.reload_model(mover_start_xy_pos=start_qpos[:, :2])

        # reset corrective movement measurement
        self.cm_measurement.reset()

        # reset throughput measurement
        self.time_to_success_s = -1
        self.num_elapsed_cycles = 0
        self.success_counter = 0

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

    def _after_mujoco_step_callback(self):
        """Check whether corrective movements (overshoot or distance corrections) occurred and increase the corresponding counter if
        necessary.
        """
        current_object_pose = self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + self.object_joint_qpos_ndim]
        current_object_pose = self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + self.object_joint_qpos_ndim]
        object_achieved_goal = current_object_pose[:2]
        object_desired_goal = self.object_xy_goal_pos.copy()

        # corrective movements
        self.cm_measurement.update_distance_corrections(current_object_pose=object_achieved_goal, object_target_pose=object_desired_goal)
        self.cm_measurement.update_overshoot_corrections(current_object_pose=object_achieved_goal, object_target_pose=object_desired_goal)

        # throughput
        if self.success_counter < 150:
            self.num_elapsed_cycles += 1
            object_geoms = None
            if self.learn_pose:
                object_geoms = self._geoms(name='object')[np.newaxis]
                w, x, y, z = current_object_pose[3:]
                object_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                object_achieved_goal = np.append(object_achieved_goal, [np.sin(object_yaw), np.cos(object_yaw)])
                object_desired_goal = np.append(object_desired_goal, [np.sin(self.object_goal_yaw), np.cos(self.object_goal_yaw)])
            goal_reached, _ = self._is_goal_reached(object_achieved_goal[np.newaxis], object_desired_goal[np.newaxis], object_geoms)
            if isinstance(goal_reached, np.ndarray):
                assert goal_reached.shape == (1,)
                goal_reached = goal_reached[0]
            if goal_reached:
                if self.success_counter == 0:
                    self.time_to_success_s = self.num_elapsed_cycles * (1 / 1000)  # in s
                self.success_counter += 1
            else:
                self.success_counter = 0

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Compute whether the episode should terminate.

        The episode terminates if there is a collision between movers, a collision with a wall, or if early termination conditions are
        met (when configured).

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
        assert info is not None

        early_termination = self.early_termination_steps is not None and info['steps_at_goal'] >= self.early_termination_steps

        return bool(info['mover_collision'] or info['wall_collision'] or early_termination)

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

    def _geom_to_shapely(self, geom: np.ndarray) -> sg.Polygon:
        """Convert MuJoCo geometry data to a Shapely polygon.

        :param geom_data: a numpy array containing [type, size_x, size_y, pos_x, pos_y]
            where type indicates geometry type (6=box, 5=cylinder)
        :return: a Shapely Polygon representing the geometry
        """
        type, sx, sy, px, py = geom

        match np.int8(type):
            case mujoco.mjtGeom.mjGEOM_CYLINDER:  # type: ignore
                shape = sg.Polygon([(sx * np.cos(angle), sx * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 32)])

            case mujoco.mjtGeom.mjGEOM_BOX:  # type: ignore
                shape = sg.box(-sx, -sy, sx, sy)

            case _:
                raise ValueError(f'Unsupported geometry type: {type}')

        return shapely.affinity.translate(shape, px, py)

    def _body_to_shapely(
        self,
        geoms: np.ndarray,
        pose: np.ndarray,
    ) -> sg.MultiPolygon:
        """Convert a MuJoCo body to a Shapely polygon representation.

        :param body_name: the name of the MuJoCo body to convert
        :return: a Shapely Polygon representing the union of all geometries in the body
        """
        x, y, sin_yaw, cos_yaw = pose.squeeze()
        yaw = np.arctan2(sin_yaw, cos_yaw)

        return shapely.affinity.translate(
            shapely.affinity.rotate(
                sg.MultiPolygon([self._geom_to_shapely(geoms[i]) for i in range(geoms.shape[0]) if geoms[i].any()]),
                yaw,
                origin='center',
                use_radians=True,
            ),
            x,
            y,
        )

    def _object_coverage(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        object_geoms: np.ndarray,
    ) -> np.ndarray:
        """Calculate the coverage ratio between achieved and desired object poses.

        Used in pose learning mode to determine how well the object's current pose
        matches the desired pose by computing the intersection area ratio.

        :param achieved_goal: a numpy array containing the current object pose [x, y, yaw]
        :param desired_goal: a numpy array containing the target object pose [x, y, yaw]
        :param object_geoms: a numpy array containing geometry data for the object
        :return: the coverage ratio between 0.0 and 1.0, where 1.0 means perfect overlap
        """
        object_polys = [self._body_to_shapely(geoms, pose) for geoms, pose in zip(object_geoms, achieved_goal, strict=True)]
        goal_polys = [self._body_to_shapely(geoms, pose) for geoms, pose in zip(object_geoms, desired_goal, strict=True)]

        return np.array(
            [
                np.clip(goal_poly.intersection(object_poly).area / goal_poly.area, 0, 1)
                for object_poly, goal_poly in zip(object_polys, goal_polys, strict=True)
            ]
        )

    def _is_goal_reached(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        object_geoms: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check whether the object has reached its goal position or pose.

        The goal achievement condition depends on the learning mode:
        - Position mode (learn_pose=False): Uses Euclidean distance threshold
        - Pose mode (learn_pose=True): Uses coverage ratio threshold

        :param achieved_goal: a numpy array containing the current object state. For position mode,
            shape (2,) with [x, y]. For pose mode, shape (3,) with [x, y, yaw]
        :param desired_goal: a numpy array containing the target object state. Same shape as achieved_goal
        :param object_geoms: a numpy array containing geometry data for the object, defaults to None.
            Required when learn_pose=True, ignored otherwise
        :return: True if the goal is reached according to the current learning mode, False otherwise. Additionally, the coverage
            (if ``learn_pose=True``) or the distance to the goal (if ``learn_pose=False``) is returned.
        """
        if self.learn_pose:
            assert object_geoms is not None

            coverage = self._object_coverage(
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
                object_geoms=object_geoms,
            )
            return coverage >= self.min_coverage, coverage
        else:
            dist_goal = self._calc_eucl_dist_xy(
                achieved_goal=achieved_goal,
                desired_goal=desired_goal,
            )
            return dist_goal <= self.max_position_err, dist_goal

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> np.ndarray | float:
        """Compute the immediate reward.

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            already achieved (x,y)-positions of an object
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            (x,y) goal positions of an object
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        assert achieved_goal is not None and desired_goal is not None and info is not None

        batch_size, mover_collisions, wall_collisions, object_geoms = self._preprocess_info_dict(info=info)
        if batch_size == 1:
            achieved_goal = achieved_goal.reshape(batch_size, -1)
            desired_goal = desired_goal.reshape(batch_size, -1)

        has_collision = mover_collisions | wall_collisions
        goal_reached, dist_or_coverage = self._is_goal_reached(
            achieved_goal,
            desired_goal,
            object_geoms,
        )

        reward = self.per_step_penalty * np.ones(shape=goal_reached.shape)
        reward[has_collision] = self.collision_penalty
        if not self.learn_pose:
            reward[np.logical_and(goal_reached, np.logical_not(has_collision))] = self.object_at_goal_reward
        else:
            coverage_largerzero = np.logical_and(dist_or_coverage > 0, np.logical_not(has_collision))
            reward[coverage_largerzero] = dist_or_coverage[coverage_largerzero]

        return reward

    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a dictionary containing the following keys and values:

            - 'observation':

                - if ``learn_jerk=True``:
                    a numpy array of shape (2*2,) containing the (x,y)-position, (x,y)-velocities and (x,y)-accelerations of the mover
                - if ``learn_jerk=False``:
                    a numpy array of shape (2,) containing the (x,y)-position and (x,y)-velocities and of the mover
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
            observation = np.concatenate((mover_xy_pos, mover_xy_velos, mover_xy_accs), axis=0)
        else:
            observation = np.concatenate((mover_xy_pos, mover_xy_velos), axis=0)

        # achieved goal
        object_qpos = self.data.qpos[self.object_joint_qpos_adr : self.object_joint_qpos_adr + self.object_joint_qpos_ndim]
        achieved_goal = object_qpos[:2] + self.rng_noise.normal(loc=0.0, scale=self.object_noise_xy_pos, size=2)

        # desired goal
        desired_goal = self.object_xy_goal_pos.copy()

        if self.learn_pose:
            w, x, y, z = object_qpos[3:]
            object_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            achieved_goal = np.append(
                achieved_goal,
                [
                    np.sin(object_yaw + self.rng_noise.normal(loc=0.0, scale=self.object_noise_yaw)),
                    np.cos(object_yaw + self.rng_noise.normal(loc=0.0, scale=self.object_noise_yaw)),
                ],
            )

            desired_goal = np.append(desired_goal, [np.sin(self.object_goal_yaw), np.cos(self.object_goal_yaw)])

        return OrderedDict(
            [
                ('observation', observation.flatten()),
                ('achieved_goal', achieved_goal),
                ('desired_goal', desired_goal),
            ]
        )

    def _geoms(self, name: str) -> np.ndarray:
        """Extract geometry information from a MuJoCo body.

        Iterates through all geometries in the model and collects those belonging to the
        specified body. For each geometry, extracts type, size, and position information.

        :param name: the name of the MuJoCo body to extract geometries from
        :return: a numpy array of shape (max_num_geoms, 5) containing geometry data.
            Each row contains [type, size_x, size_y, pos_x, pos_y]. Unused rows are zero-filled.
            Geometry types: 6=box, 5=cylinder
        """
        body = self.model.body(name)

        max_num_geoms = 3
        geoms = np.zeros((max_num_geoms, 5))

        geom_idx = 0
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body.id:
                if geom_idx >= max_num_geoms:
                    logger.warn(
                        f"Body '{name}' has more than {max_num_geoms} geometries. Only the first {max_num_geoms} will be processed."
                    )
                    break

                geom = self.model.geom(geom_id)
                geoms[geom_idx, 0] = geom.type
                geoms[geom_idx, 1:3] = geom.size[:2]
                geoms[geom_idx, 3:5] = geom.pos[:2]
                geom_idx += 1

        return geoms

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
        :param achieved_goal: a numpy array of shape (length achieved_goal,) containing the already achieved (x,y)-position of the
            object
        :param desired_goal: a numpy array of shape (length achieved_goal,) containing the desired (x,y)-position of the object
        :param collision_info: a dictionary that is intended to contain additional information about collisions, e.g.
            collisions with obstacles. Defaults to None (not used in this environment)
        :return: the info dictionary with keys 'is_success', 'mover_collision' and 'wall_collision'
        """
        assert achieved_goal is not None and desired_goal is not None

        info_dict: dict[str, bool | np.ndarray | int] = {
            'mover_collision': mover_collision,
            'wall_collision': wall_collision,
        }

        object_geoms = None
        if self.learn_pose:
            info_dict['object_geoms'] = self._geoms(name='object')
            object_geoms = info_dict['object_geoms'][np.newaxis]

        goal_reached, _ = self._is_goal_reached(achieved_goal[np.newaxis], desired_goal[np.newaxis], object_geoms)
        if isinstance(goal_reached, np.ndarray):
            assert goal_reached.shape == (1,)
            goal_reached = goal_reached[0]
        self.steps_at_goal = self.steps_at_goal + 1 if goal_reached else 0
        is_success = bool(goal_reached) and not bool(mover_collision) and not bool(wall_collision)
        info_dict.update(
            {
                'is_success': is_success,
                'steps_at_goal': self.steps_at_goal,
                'num_overshoot_corrections': self.get_current_num_overshoot_corrections(),
                'num_distance_corrections': self.get_current_num_distance_corrections(),
                'time_to_success_s': self.time_to_success_s,
            }
        )

        return info_dict

    def close(self) -> None:
        """Close the environment."""
        super().close()

    def ensure_max_dyn_val(self, current_values: np.ndarray, max_value: float, next_derivs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def _preprocess_info_dict(
        self,
        info: np.ndarray | dict[str, Any] | None,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None]:
        """Extract information about mover collisions, wall collisions and the batch size from the info dictionary.

        :param info: the info dictionary or an array of info dictionary to be preprocessed. All dictionaries must contain the keys
            'mover_collision' and 'wall_collision'.
        :return: the batch_size (int), a numpy array of shape (batch_size,) containing the mover collision values (bool),
            a numpy array of shape (batch_size,) containing the wall collision values (bool), information about object geoms (if pose
            task, else None)
        """
        if isinstance(info, np.ndarray):
            batch_size = info.shape[0]
            mover_collisions = np.zeros(batch_size).astype(bool)
            wall_collisions = np.zeros(batch_size).astype(bool)
            if self.learn_pose:
                object_geoms = np.zeros((batch_size, 3, 5), np.float32)

            for i in range(0, batch_size):
                mover_collisions[i] = info[i]['mover_collision']
                wall_collisions[i] = info[i]['wall_collision']
                if self.learn_pose:
                    object_geoms[i] = info[i]['object_geoms']
        else:
            assert isinstance(info, dict)
            batch_size = 1
            mover_collisions = np.array([info['mover_collision']])
            wall_collisions = np.array([info['wall_collision']])
            if self.learn_pose:
                object_geoms = np.array([info['object_geoms']])

        return batch_size, mover_collisions, wall_collisions, object_geoms if self.learn_pose else None

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

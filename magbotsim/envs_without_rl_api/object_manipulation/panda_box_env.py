##########################################################
# Copyright (c) 2025 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
import mujoco

from magbotsim import BasicMagBotEnv, MoverImpedanceController
from magbotsim.utils import mujoco_utils


class PandaBoxExampleEnv(BasicMagBotEnv):
    """Simple example environment with 4 movers, a Franka Emika Panda robot, and a box.

    :param initial_panda_qpos_dict: _description_, defaults to None
    """

    def __init__(self, initial_panda_qpos_dict: dict | None = None) -> None:
        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264, 0.639, 0.639, 3.424])
        bumper_mass = np.array([0.05, 0.034, 0.034, 0.01])
        mover_params = {
            'shape': 'mesh',
            'mesh': {
                'mover_stl_path': [
                    'beckhoff_apm4330_mover',
                    'beckhoff_apm4220_mover',
                    'beckhoff_apm4220_mover',
                    'beckhoff_apm4550_mover',
                ],
                'bumper_stl_path': [
                    'beckhoff_apm4330_bumper',
                    'beckhoff_apm4220_bumper',
                    'beckhoff_apm4220_bumper',
                    'beckhoff_apm4550_bumper',
                ],
            },
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        # we model the mover as a simple 2D box to check collisions
        # since we are using Beckhoff XPlanar movers in this example, we
        # use the sizes specified in the technical drawings plus a safety margin
        collision_params = {
            'shape': 'box',
            'size': np.array(
                [
                    [0.155 / 2, 0.155 / 2],
                    [0.113 / 2, 0.113 / 2],
                    [0.113 / 2, 0.113 / 2],
                    [0.235 / 2, 0.235 / 2],
                ]
            )
            + 0.001,
        }
        self.initial_mover_start_xy_pos = np.array(
            [
                [0.55, 0.48],
                [0.124, 0.126],
                [0.8, 0.23],
                [0.3, 0.7],
            ]
        )
        self.initial_mover_z_pos = 0.002
        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((4, 4)),
            num_movers=4,
            mover_params=mover_params,
            initial_mover_zpos=self.initial_mover_z_pos,
            table_height=0.2,
            collision_params=collision_params,
            initial_mover_start_xy_pos=self.initial_mover_start_xy_pos,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=True,
        )
        # remember initial mover positions and orientations
        self.initial_mover_xyz_pos = np.zeros((self.num_movers, 3))
        self.initial_mover_xyz_pos[:, :2] = self.initial_mover_start_xy_pos
        self.initial_mover_xyz_pos[:, -1] = self.initial_mover_z_pos
        # impedance contoller
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=np.array([0, 1, 1, 1, 1, 1]),
                translational_stiffness=np.array([1.0, 1.0, 1.0]),
                rotational_stiffness=np.array([0.1, 0.1, 1]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # remember Panda joint names
        self.panda_joint_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='joint', name_pattern='panda')
        # reload model to add actuators
        self.reload_model(mover_start_xy_pos=self.initial_mover_start_xy_pos)

        # remember mover actuator names
        self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(self.model, obj_type='actuator', name_pattern='mover_actuator_x')

        # set initial EE pos (Panda robot)
        if initial_panda_qpos_dict is None:
            self.initial_panda_qpos_dict = {
                'panda_joint1': 0.0,
                'panda_joint2': 0.238,
                'panda_joint3': 0.0,
                'panda_joint4': -1.66,
                'panda_joint5': 0.0,
                'panda_joint6': 2.18,
                'panda_joint7': 0.0,
                'panda_finger_joint1': 0.0204,
                'panda_finger_joint2': 0.0152,
            }
        else:
            self.initial_panda_qpos_dict = initial_panda_qpos_dict
        self.reset_panda_qpos(panda_qpos_dict=self.initial_panda_qpos_dict)

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the Franka Emika Panda robot and a box to the MuJoCo model, as well as actuators for the movers and
        the Panda robot by modifying the ``custom_model_xml_strings``-dict.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}

        # add panda robot
        mujoco_xml_path = self.assetdir / 'mujoco_xmls'
        panda_xml_path = mujoco_xml_path / 'panda.xml'
        mp_xml_str = f'\n\n\t<include file="{panda_xml_path}"/>'
        custom_outworldbody_xml_str = mp_xml_str

        # add box
        custom_mover_body_xml_strs = [None] * self.num_movers
        mb_xml_str_list = [
            '\n\n\t\t\t<body name="box" pos="0 0 0.02" euler="0 0 2.75" gravcomp="1">',
            '\n\t\t\t\t<geom name="box_geom" type="box" size="0.02 0.02 0.02" mass="0.01" pos="0 0 0" material="blue"/>',
            '\n\t\t\t</body>',
        ]
        custom_mover_body_xml_strs[0] = ''.join(mb_xml_str_list)
        custom_model_xml_strings['custom_mover_body_xml_str_list'] = custom_mover_body_xml_strs

        # add more light
        light_xml_str_list = [
            '\n\t\t<light pos=".0 .0 .3" dir="-2 2 -1.5" diffuse=".6 .6 .6"/>',
            '\n\t\t<light pos="0.5 -2.5 .7" dir="-1 5 -1.5" diffuse=".6 .6 .6" castshadow="true"/>',
        ]
        custom_model_xml_strings['custom_worldbody_xml_str'] = ''.join(light_xml_str_list)

        # add actuators
        # we don't know the mover joint names in advance, but after the fist compilation of the MuJoCo model
        if hasattr(self, 'mover_joint_names'):
            # add mover actuators
            actuator_lines = ['\n\n\t<actuator>']

            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(f'\n\t\t<!-- actuators mover {idx_mover} -->')
                actuator_lines.append(
                    f'\n\t\t<general name="mover_actuator_x_{idx_mover}" '
                    f'joint="{self.mover_joint_names[idx_mover]}" '
                    f'gear="1 0 0 0 0 0" dyntype="none" gaintype="fixed" '
                    f'gainprm="{self.mover_mass[idx_mover]} 0 0" biastype="none"/>'
                )
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
                actuator_lines.append('\n')

            # add actuators for the Panda robot
            actuator_lines.append('\n\n\t\t<!-- actuators panda robot -->')
            self.panda_actuator_names = []
            for joint_name in self.panda_joint_names:
                actuator_name = f'motor_{joint_name}'
                self.panda_actuator_names.append(actuator_name)
                actuator_lines.append(f'\n\t\t<motor name="{actuator_name}" joint="{joint_name}" gear="1"/>')

            actuator_lines.append('\n\t</actuator>')
            custom_outworldbody_xml_str += ''.join(actuator_lines)

        custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray | None = None) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover,
            defaults to None. If set to None, the movers will be placed in the center of the tiles that are added to the XML string
            first.
        """
        # generate a new model XML string
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        # compile the MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # update cached mujoco data
        self.update_cached_mover_mujoco_data()
        for idx_mover in range(0, self.num_movers):
            self.impedance_controllers[idx_mover].update_cached_mujoco_data(self.model)

        # render the environment after reloading
        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def reset_panda_qpos(self, panda_qpos_dict: dict[str, float]) -> None:
        """Set specific joint qpos for the joints of the Panda robot.

        :param panda_qpos_dict: a dictionary containing the (joint_name, new_qpos_value)-pairs. If only a subset of joints is
            included, the qpos-values for the remaining joints are not changed.
        """
        for name, value in panda_qpos_dict.items():
            mujoco_utils.set_joint_qpos(self.model, self.data, name, value)
        mujoco.mj_forward(self.model, self.data)

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
                    model=self.model, data=self.data, pos_d=self.initial_mover_xyz_pos[idx_mover, :], quat_d=np.array([1, 0, 0, 0])
                )
            # set controls Panda
            for idx_pa, actuator_name in enumerate(self.panda_actuator_names):
                mujoco_utils.set_actuator_ctrl(model=self.model, data=self.data, actuator_name=actuator_name, value=panda_ctrls[idx_pa])
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

##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

from typing import Any

import numpy as np
from pettingzoo import ParallelEnv

from magbotsim import BasicMagBotEnv


class BasicMagBotMultiAgentEnv(BasicMagBotEnv, ParallelEnv):
    """A base class for multi-agent reinforcement learning environments in the field of magnetic robotics that follow the PettingZoo
    API. A more detailed explanation of all parameters can be found in the documentation of the ``BasicMagneticRoboticsEnv``.

    :param layout_tiles: the tile layout
    :param num_movers: the number of movers
    :param tile_params: tile parameters such as the size and mass, defaults to None
    :param mover_params: mover parameters such as the size and mass, defaults to None
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.005 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'
    :param default_cam_config: dictionary with attribute values of the viewer's default camera,
        https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global, defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param collision_params: a dictionary that can be used to specify collision parameters, defaults to None
    :param initial_mover_start_xy_pos: the initial (x,y) starting positions of the movers, defaults to None
    :param initial_mover_goal_xy_pos: the initial (x,y) goal positions of the movers, defaults to None
    :param custom_xml_strings: a dictionary containing additional XML strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model. The keys determine where to add a string in the XML structure and the values contain
        the XML string to add. The following keys are accepted:

        - ``custom_compiler_xml_str``:
            A custom 'compiler' XML section. Note that the entire default 'compiler' section is replaced.
        - ``custom_visual_xml_str``:
            A custom 'visual' XML section. Note that the entire default 'visual' section is replaced.
        - ``custom_option_xml_str``:
            A custom 'option' XML section. Note that the entire default 'option' section is replaced.
        - ``custom_assets_xml_str``:
            This XML string adds elements to the 'asset' section.
        - ``custom_default_xml_str``:
            This XML string adds elements to the 'default' section.
        - ``custom_worldbody_xml_str``:
            This XML string adds elements to the 'worldbody' section.
        - ``custom_contact_xml_str``:
            This XML string adds elements to the 'contact' section.
        - ``custom_outworldbody_xml_str``:
            This XML string should be used to include files or add sections.
        - ``custom_mover_body_xml_str_list``:
            This list of XML strings should be used to attach objects to a mover. Note that this a list with length num_movers.
            If nothing is attached to a mover, add None at the corresponding mover index.

        If set to None, only the basic XML string is generated, containing tiles, movers (excluding actuators),
        and possibly goals; defaults to None. This dictionary can be further modified using the ``_custom_xml_string_callback()``.
    :param use_mj_passive_viewer: whether the MuJoCo passive_viewer should be used, defaults to False. If set to False, the Gymnasium
        MuJoCo WindowViewer with custom overlays is used.
    """

    def __init__(
        self,
        layout_tiles: np.ndarray,
        num_movers: int,
        tile_params: dict[str, Any] | None = None,
        mover_params: dict[str, Any] | None = None,
        initial_mover_zpos: float = 0.005,
        table_height: float = 0.4,
        std_noise: np.ndarray | float = 1e-5,
        render_mode: str | None = 'human',
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        super(BasicMagBotEnv, self).__init__(
            layout_tiles=layout_tiles,
            num_movers=num_movers,
            tile_params=tile_params,
            mover_params=mover_params,
            initial_mover_zpos=initial_mover_zpos,
            table_height=table_height,
            std_noise=std_noise,
            render_mode=render_mode,
            default_cam_config=default_cam_config,
            width_no_camera_specified=width_no_camera_specified,
            height_no_camera_specified=height_no_camera_specified,
            collision_params=collision_params,
            initial_mover_start_xy_pos=initial_mover_start_xy_pos,
            initial_mover_goal_xy_pos=initial_mover_goal_xy_pos,
            custom_model_xml_strings=custom_model_xml_strings,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

        self.agents = self.mover_names
        self.possible_agents = self.mover_names

##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import logger

from magbotsim import BasicMagBotEnv


class BasicMagBotSingleAgentEnv(BasicMagBotEnv, gym.Env, ABC):
    """A base class for single-agent reinforcement learning environments in the field of Magnetic Robotics that follow the Gymnasium
    API. A more detailed explanation of all parameters can be found in the documentation of the ``BasicMagBotEnv``.

    :param layout_tiles: the tile layout
    :param num_movers: the number of movers
    :param tile_params: tile parameters such as the size and mass, defaults to None
    :param mover_params: mover parameters such as the size and mass, defaults to None
    :param initial_mover_zpos: the initial distance between the bottom of the mover and the top of a tile, defaults to 0.005 [m]
    :param table_height: the height of a table on which the tiles are placed, defaults to 0.4 [m]
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 1e-5
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'
    :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to False.
        Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive. Thus, this
        parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene is always
        rendered after 'num_cycles' have been executed if 'render_mode != None'.
    :param default_cam_config: dictionary with attribute values of the viewer's default camera (see
        `MuJoCo docs <https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=camera#visual-global>`_), defaults to None
    :param width_no_camera_specified: if render_mode != 'human' and no width is specified, this value is used, defaults to 1240
    :param height_no_camera_specified: if render_mode != 'human' and no height is specified, this value is used, defaults to 1080
    :param num_cycles: the number of control cycles for which to apply the same action, defaults to 40
    :param collision_params: a dictionary that can be used to specify collision parameters, defaults to None
    :param initial_mover_start_xy_pos: the initial (x,y) starting positions of the movers, defaults to None
    :param initial_mover_goal_xy_pos: the initial (x,y) goal positions of the movers, defaults to None
    :param custom_xml_strings: a dictionary containing additional XML  strings to provide the ability to add actuators, sensors,
        objects, robots, etc. to the model. The keys determine where to add a string in the XML  structure and the values contain
        the XML  string to add. The following keys are accepted:

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
        render_every_cycle: bool = False,
        default_cam_config: dict[str, Any] | None = None,
        width_no_camera_specified: int = 1240,
        height_no_camera_specified: int = 1080,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        initial_mover_start_xy_pos: np.ndarray | None = None,
        initial_mover_goal_xy_pos: np.ndarray | None = None,
        custom_model_xml_strings: dict[str, str] | None = None,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        super().__init__(
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

        # rendering
        self.render_every_cycle = render_every_cycle
        # number of control cycles for which to apply the same action
        self.num_cycles = num_cycles

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment returning an initial observation and auxiliary information. More detailed information about the
        parameters and return values can be found in the Gymnasium documentation:
        https://gymnasium.farama.org/api/env/#gymnasium.Env.reset.

        This method performs the following steps:

        - reset RNG, if desired
        - call ``_reset_callback(option)`` to give the user the opportunity to add more functionality
        - call ``mj_forward()``
        - check whether there are mover, wall, or other collisions, e.g. collisions with an obstacle
        - call ``render()``
        - get initial observation and info dictionary

        :param seed: if set to None, the RNG is not reset; if int, sets the desired seed; defaults to None
        :param options: a dictionary that can be used to specify additional reset options, e.g. object parameters; defaults to None
        :return: initial observation and auxiliary information contained in the 'info' dictionary
        """
        # reset RNG of the environment if seed is not None
        super().reset(seed=seed)
        if seed is not None:
            self.rng_noise = np.random.default_rng(seed=seed)

        # custom callback to add more functionality
        self._reset_callback(options)

        # update sim
        mujoco.mj_forward(self.model, self.data)
        # check mover and wall collision
        wall_collision = self.check_wall_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=True, mover_qpos=None, add_qpos_noise=True
        ).any()
        # check mover collision
        mover_collision = self.check_mover_collision(
            mover_names=self.mover_names, c_size=self.c_size, add_safety_offset=False, mover_qpos=None, add_qpos_noise=True
        ).any()
        # check for other collisions, e.g. collisions with an obstacle
        other_collision, collision_info = self._check_for_other_collisions_callback()

        # rendering
        self.render()

        # get new observation and info
        observation = self._get_obs()
        if isinstance(observation, dict) and 'achieved_goal' in observation.keys() and 'desired_goal' in observation.keys():
            info = self._get_info(
                mover_collision=mover_collision,
                wall_collision=wall_collision,
                other_collision=other_collision,
                achieved_goal=observation['achieved_goal'],
                desired_goal=observation['desired_goal'],
                collision_info=collision_info,
            )
        else:
            info = self._get_info(
                mover_collision=mover_collision,
                wall_collision=wall_collision,
                other_collision=other_collision,
                achieved_goal=None,
                desired_goal=None,
                collision_info=collision_info,
            )

        return observation, info

    def step(self, action: int | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one step of the environment's dynamics applying the given action.
        Note that the environment executes as many MuJoCo simulation steps as the number of cycles specified for this environment
        (``num_cycles``). The duration of one cycle is determined by the cycle time, which must be specified in the MuJoCo XML
        string using the ``option/timestep`` parameter. The same action is applied for all cycles.

        This method performs the following steps:

        - check whether the dimension of the action matches the dimension of the action space
        - if the action space does not contain the specified action, the action is clipped to the interval edges of
          the action space
        - call ``_step_callback(action)`` to give the user the opportunity to add more functionality
        - execute MuJoCo simulation steps (``mj_step()``). After each simulation step, it is checked whether there are mover, wall, or
          other collisions, e.g. collisions with an obstacle. To check for other collisions besides mover and wall collisions the
          ``_check_for_other_collisions_callback()`` is called. In case of a collision, no further simulation steps are performed, as
          a real system would typically stop as well due to position lag errors. In addition, ``render()`` can be called after each
          simulation step to provide a smooth visualization of the movement (set ``render_every_cycle=True``).
          The callbacks ``_before_mujoco_step_callback(action)`` and ``_after_mujoco_step_callback()`` are executed before and
          after ``mujoco.mj_step(self.model, self.data, nstep=1)`` is called and can be used to add functionality.
          This can be useful, for example, to ensure velocity or acceleration limits within each cycle.
        - call ``render()``
        - get return values
        - call ``_on_step_end_callback(observation)`` to give the user the opportunity to add more functionality

        More detailed information about the parameters and return values can be found in the Gymnasium documentation:
        https://gymnasium.farama.org/api/env/#gymnasium.Env.step.

        :param action: the action to apply
        :return:
                - the next observation
                - the immediate reward for taking the action
                - whether a terminal state is reached
                - whether the truncation condition is satisfied
                - auxiliary information contained in the 'info' dictionary
        """
        # make sure that shape is correct and action is within action space
        if not isinstance(action, int):
            assert action.shape == self.action_space.shape, 'action dim != action_space dim'
            if not self.action_space.contains(action):
                logger.warn(f'Action {action} not in action space. Will clip invalid values to interval edges.')
                action = np.clip(action, self.action_space.low, self.action_space.high)

        # custom callback to add more functionality
        self._step_callback(action)

        # integration and collision check
        for _ in range(0, self.num_cycles):
            self._before_mujoco_step_callback(action)
            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            self._after_mujoco_step_callback()
            # render every cycle for a smooth visualization of the movement
            if self.render_every_cycle:
                self.render()
            # check wall and mover collision every cycle to ensure that the collisions are detected and all intermediate
            # mover positions are valid and without collisions
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
            # check for further collisions, e.g. with obstacles
            other_collision, collision_info = self._check_for_other_collisions_callback()
            if mover_collision or wall_collision or other_collision:
                break

        self.render()

        # get next observation
        observation = self._get_obs()
        if isinstance(observation, dict) and 'achieved_goal' in observation.keys() and 'desired_goal' in observation.keys():
            # goal-conditioned RL
            info = self._get_info(
                mover_collision=mover_collision,
                wall_collision=wall_collision,
                other_collision=other_collision,
                achieved_goal=observation['achieved_goal'],
                desired_goal=observation['desired_goal'],
                collision_info=collision_info,
            )
            reward = self.compute_reward(achieved_goal=observation['achieved_goal'], desired_goal=observation['desired_goal'], info=info)
            terminated = self.compute_terminated(
                achieved_goal=observation['achieved_goal'], desired_goal=observation['desired_goal'], info=info
            )
            truncated = self.compute_truncated(
                achieved_goal=observation['achieved_goal'], desired_goal=observation['desired_goal'], info=info
            )
        else:
            info = self._get_info(
                mover_collision=mover_collision,
                wall_collision=wall_collision,
                other_collision=other_collision,
                achieved_goal=None,
                desired_goal=None,
                collision_info=collision_info,
            )
            reward = self.compute_reward(achieved_goal=None, desired_goal=None, info=info)
            terminated = self.compute_terminated(achieved_goal=None, desired_goal=None, info=info)
            truncated = self.compute_truncated(achieved_goal=None, desired_goal=None, info=info)
        # check reward shape
        if isinstance(reward, np.ndarray) and reward.shape[0] > 1:
            logger.warn(
                f"Unexpected shape of reward returned by 'env.compute_reward()'. Current shape: {reward.shape}, expected shape: (1,)"
            )
        elif isinstance(reward, np.ndarray) and reward.shape[0] == 1:
            reward = reward[0]

        self._on_step_end_callback(observation)

        return observation, reward, terminated, truncated, info

    def _reset_callback(self, options: dict[str, Any] | None = None) -> None:
        """A callback that should be used to add further functionality to the ``reset()`` method (see documentation of the ``reset()``
        method for more information about when the callback is called).

        :param options: a dictionary that can be used to specify additional reset options, e.g. object parameters; defaults to None
        """
        pass

    def _step_callback(self, action: int | np.ndarray) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).

        :param action: the action to apply
        """
        pass

    def _on_step_end_callback(self, observation: dict[str, np.ndarray] | np.ndarray) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).

        :param observation: the next observation after the action was applied
        """
        pass

    def _before_mujoco_step_callback(self, action: int | np.ndarray) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).

        :param action: the action to apply
        """
        pass

    def _after_mujoco_step_callback(self) -> None:
        """A callback that should be used to add further functionality to the ``step()`` method (see documentation of the ``step()``
        method for more information about when the callback is called).
        """
        pass

    def _check_for_other_collisions_callback(self) -> tuple[bool, dict[str, Any] | None]:
        """A callback that is intended to use to check for other collisions besides mover or wall collisions, e.g. collisions with
        obstacles.

        :return:
            - whether there is a collision (bool)
            - a dictionary that is intended to contain additional information about the collision (can be None)
        """
        other_collision = False
        collision_info = None
        return other_collision, collision_info

    @abstractmethod
    def compute_terminated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether a terminal state is reached. This method can be used for both goal-conditioned RL and standard RL.
        Since Hindsight Experience Replay (HER) is commonly used in goal-conditioned RL, this method receives
        the 'achieved_goal' and 'desired_goal' corresponding to the requirements of the HER implementation of stable-baselines3
        (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single bool value or a numpy array of shape (batch_size,) containing Boolean values, where True indicates that
            a terminal state has been reached
        """
        pass

    @abstractmethod
    def compute_truncated(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | bool:
        """Check whether the truncation condition is satisfied. This method can be used for both goal-conditioned RL and standard RL.
        Since Hindsight Experience Replay (HER) is commonly used in goal-conditioned RL, this method receives
        the 'achieved_goal' and 'desired_goal' corresponding to the requirements of the HER implementation of stable-baselines3
        (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single bool value or a numpy array of shape (batch_size,) containing Boolean values, where True indicates that
            a the truncation condition is satisfied
        """
        pass

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray | None = None, desired_goal: np.ndarray | None = None, info: dict[str, Any] | None = None
    ) -> np.ndarray | float:
        """Compute the immediate reward. This method is required by the stable-baselines3 implementation of Hindsight Experience
        Replay (HER) (for more information, see https://stable-baselines3.readthedocs.io/en/master/modules/her.html).

        :param achieved_goal: a numpy array of shape (batch_size, length achieved_goal) or (length achieved_goal,) containing the
            goals already achieved (goal-conditioned RL); defaults to None (standard RL)
        :param desired_goal: a numpy array of shape (batch_size, length desired_goal) or (length desired_goal,) containing the
            desired goals (goal-conditioned RL); defaults to None (standard RL)
        :param info: a dictionary containing auxiliary information, defaults to None
        :return: a single float value or a numpy array of shape (batch_size,) containing the immediate rewards
        """
        pass

    @abstractmethod
    def _get_obs(self) -> dict[str, np.ndarray] | np.ndarray:
        """Return an observation based on the current state of the environment.

        :return: a numpy array or a dictionary (dictionary observation space - required by HER implementation of stable-baselines3)
        """
        pass

    @abstractmethod
    def _get_info(
        self,
        mover_collision: bool,
        wall_collision: bool,
        other_collision: bool,
        achieved_goal: np.ndarray | None = None,
        desired_goal: np.ndarray | None = None,
        collision_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a dictionary that contains auxiliary information that may depend on the 'achieved_goal' and 'desired_goal' in
        goal-conditioned RL.

        :param mover_collision: whether there is a collision between two movers
        :param wall_collision: whether there is a collision between a mover and a wall
        :param other_collision: whether there are other collisions besides wall or mover collisions, e.g. collisions with an obstacle
        :param achieved_goal: a numpy array containing the goal which already achieved (goal-conditioned RL) - the shape
            depends on the shape of the observation space; defaults to None
        :param desired_goal: a numpy array containing the desired goal (goal-conditioned RL) - the shape
            depends on the shape of the observation space; defaults to None
        :param collision_info: a dictionary that is intended to contain additional information about collisions, e.g.
            collisions with obstacles. Defaults to None
        :return: a dictionary with auxiliary information
        """
        pass

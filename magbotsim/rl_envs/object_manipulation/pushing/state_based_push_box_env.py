from typing import Any

import numpy as np

from magbotsim.rl_envs.object_manipulation.pushing.state_based_global_pushing_env import (
    DEFAULT_OBJECT_RANGES,
    StateBasedGlobalPushingEnv,
)

"""A simplified object pushing environment for basic manipulation tasks.

    This environment is a preconfigured version of ``StateBasedObjectPushingEnv`` designed for
    introductory object manipulation tasks and algorithm development. It simplifies the
    problem by using only square box objects and position-only learning, making it ideal
    for initial experimentation and baseline comparisons.
    """


class StateBasedPushBoxEnv(StateBasedGlobalPushingEnv):
    """A simplified object pushing environment with a box to be pushed.

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
    :param std_noise: the standard deviation of a Gaussian with zero mean used to add noise, defaults to 0.00001. The standard
        deviation can be used to add noise to the mover's position, velocity and acceleration. If you want to use different
        standard deviations for position, velocity and acceleration use a numpy array of shape (3,); otherwise use a single float
        value, meaning the same standard deviation is used for all three values.
    :param render_mode: the mode that is used to render the frames ('human', 'rgb_array' or None), defaults to 'human'. If set to
        None, no viewer is initialized and used, i.e. no rendering. This can be useful to speed up training.
    :param render_every_cycle:  whether to call 'render' after each integrator step in the ``step()`` method, defaults to False.
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
    :param v_max: the maximum velocity, defaults to 2.0 [m/s]
    :param a_max: the maximum acceleration, defaults to 10.0 [m/s²]
    :param j_max: the maximum jerk (only used if 'learn_jerk=True'), defaults to 100.0 [m/s³]
    :param object_sliding_friction: the sliding friction coefficient of the object, defaults to 0.6
    :param object_torsional_friction: the torsional friction coefficient of the object, defaults to 0.0001
    :param learn_jerk: whether to learn the jerk, defaults to False. If set to False, the acceleration is learned, i.e. the policy
        output.
    :param early_termination_steps: the number of consecutive steps at goal after which the episode terminates early, defaults to None
        (no early termination)
    :param max_position_err: the position threshold used to determine whether the object has reached its goal position, defaults
        to 0.05 [m]
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
        std_noise: np.ndarray | float = 0.00001,
        render_mode: str | None = 'human',
        render_every_cycle: bool = False,
        num_cycles: int = 40,
        collision_params: dict[str, Any] | None = None,
        v_max: float = 2.0,
        a_max: float = 10.0,
        j_max: float = 100.0,
        object_sliding_friction: float = 0.6,
        object_torsional_friction: float = 0.0001,
        learn_jerk: bool = False,
        early_termination_steps: int | None = None,
        max_position_err: float = 0.05,
        collision_penalty: float = -10,
        per_step_penalty: float = -0.01,
        object_at_goal_reward: float = 1.0,
        use_mj_passive_viewer: bool = False,
    ) -> None:
        super().__init__(
            num_movers=num_movers,
            mover_params=mover_params,
            layout_tiles=layout_tiles,
            initial_mover_zpos=initial_mover_zpos,
            std_noise=std_noise,
            render_mode=render_mode,
            render_every_cycle=render_every_cycle,
            num_cycles=num_cycles,
            collision_params=collision_params,
            object_type='square_box',
            object_ranges=DEFAULT_OBJECT_RANGES,
            v_max=v_max,
            a_max=a_max,
            j_max=j_max,
            object_sliding_friction=object_sliding_friction,
            object_torsional_friction=object_torsional_friction,
            learn_jerk=learn_jerk,
            learn_pose=False,
            early_termination_steps=early_termination_steps,
            max_position_err=max_position_err,
            min_coverage=0.0,
            collision_penalty=collision_penalty,
            per_step_penalty=per_step_penalty,
            object_at_goal_reward=object_at_goal_reward,
            use_mj_passive_viewer=use_mj_passive_viewer,
        )

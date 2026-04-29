##########################################################
# Copyright (c) 2025 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
import pytest

from magbotsim.rl_envs.object_manipulation.pushing.state_based_global_pushing_env import StateBasedGlobalPushingEnv


@pytest.mark.parametrize(
    'num_movers, mover_mass, jerk, num_cycles, test_x, test_y',
    [
        (1, 0.628, 100, 1, True, True),
        (1, 0.628, 100, 1, True, False),
        (1, 0.628, 100, 1, False, True),
        (1, 1.237, 100, 1, True, True),
        (1, 0.628, -100, 1, True, True),
        (1, 0.628, -100, 1, True, False),
        (1, 0.628, -100, 1, False, True),
        (1, 1.237, -100, 1, True, True),
        (1, 0.628, 100, 42, True, True),
        (1, 1.237, 100, 42, True, True),
        (1, 0.628, -100, 42, True, True),
        (1, 1.237, -100, 42, True, True),
        (1, 0.628, 150, 1, True, True),
        (1, 0.628, 150, 1, True, False),
        (1, 0.628, 150, 1, False, True),
        (1, 1.237, 150, 1, True, True),
        (1, 0.628, -150, 1, True, True),
        (1, 0.628, -150, 1, True, False),
        (1, 0.628, -150, 1, False, True),
        (1, 1.237, -150, 1, True, True),
        (1, 0.628, 150, 42, True, True),
        (1, 1.237, 150, 42, True, True),
        (1, 0.628, -150, 42, True, True),
        (1, 1.237, -150, 42, True, True),
        (2, 0.628, 100, 1, True, True),
        (2, 0.628, 100, 1, True, False),
        (2, 0.628, 100, 1, False, True),
        (2, 1.237, 100, 1, True, True),
        (2, 0.628, -100, 1, True, True),
        (2, 0.628, -100, 1, True, False),
        (2, 0.628, -100, 1, False, True),
        (2, 1.237, -100, 1, True, True),
        (2, 0.628, 100, 42, True, True),
        (2, 1.237, 100, 42, True, True),
        (2, 0.628, -100, 42, True, True),
        (2, 1.237, -100, 42, True, True),
        (2, 0.628, 150, 1, True, True),
        (2, 0.628, 150, 1, True, False),
        (2, 0.628, 150, 1, False, True),
        (2, 1.237, 150, 1, True, True),
        (2, 0.628, -150, 1, True, True),
        (2, 0.628, -150, 1, True, False),
        (2, 0.628, -150, 1, False, True),
        (2, 1.237, -150, 1, True, True),
        (2, 0.628, 150, 42, True, True),
        (2, 1.237, 150, 42, True, True),
        (2, 0.628, -150, 42, True, True),
        (2, 1.237, -150, 42, True, True),
    ],
)
def test_jerk_actuator(num_movers, mover_mass, jerk, num_cycles, test_x, test_y):
    # environment
    v_max = 0.01  # [m/s]
    a_max = 0.2  # [m/s²]
    j_max = 150  # [m/s³]
    learn_jerk = True
    collision_shape = 'box'
    collision_size = np.array([0.113 / 2 + 1e-6, 0.113 / 2 + 1e-6])
    collision_offset = 0.0
    collision_offset_wall = 0.0

    mover_params = {'size': np.array([0.113 / 2, 0.113 / 2, 0.012 / 2]), 'mass': mover_mass}

    collision_params = {
        'shape': collision_shape,
        'size': collision_size,
        'offset': collision_offset,
        'offset_wall': collision_offset_wall,
    }

    env = StateBasedGlobalPushingEnv(
        num_movers=num_movers,
        layout_tiles=np.ones((9, 9)),
        mover_params=mover_params,
        std_noise=0.0,
        collision_params=collision_params,
        render_mode=None,
        render_every_cycle=False,
        num_cycles=num_cycles,
        v_max=v_max,
        a_max=a_max,
        j_max=j_max,
        learn_jerk=learn_jerk,
        use_mj_passive_viewer=False,
    )
    if num_movers == 1:
        mover_start_xy_pos = np.array([[0.72, 0.36]])
    elif num_movers == 2:
        mover_start_xy_pos = np.array([[0.96, 0.96], [1.2, 1.2]])
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    velo_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    acc_mj_actuator = np.zeros((num_steps, 2 * num_movers))

    pos_mj_manual = np.zeros((num_steps, 2 * num_movers))
    velo_mj_manual = np.zeros((num_steps, 2 * num_movers))
    acc_mj_manual = np.zeros((num_steps, 2 * num_movers))

    for step in range(0, num_steps):
        if test_x and test_y:
            jerk_arr = np.array([jerk, jerk] * num_movers)
        elif test_x and not test_y:
            jerk_arr = np.array([jerk, 0] * num_movers)
        elif not test_x and test_y:
            jerk_arr = np.array([0, jerk] * num_movers)
        else:
            jerk_arr = np.array([0, 0] * num_movers)
        jerk_arr_step = jerk_arr.copy()

        next_j = jerk_arr.reshape((env.num_movers, 2))
        # ensure maximum jerk
        action_norm_tmp = np.linalg.norm(next_j, ord=2, axis=1)
        action_norm = np.where(action_norm_tmp <= env.j_max, 1.0, action_norm_tmp)[:, None]
        action_max_vals = np.where(action_norm == 1.0, 1.0, env.j_max)
        next_j = np.divide(next_j, action_norm) * action_max_vals

        for idx_mover in range(0, num_movers):
            if step > 0:
                v = velo_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
                p = pos_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
                a = acc_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
            else:
                v = np.zeros(2)
                p = mover_start_xy_pos[idx_mover, :].copy()
                a = np.zeros(2)

            for _ in range(0, dt):
                next_a, _ = env.ensure_max_dyn_val(a, env.a_max, next_j[idx_mover, :])
                v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a)

                a = a_tmp.copy()
                p = timestep * v + p
            pos_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = p.flatten().copy()
            velo_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = v.flatten().copy()
            acc_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = a.flatten().copy()

        # set jerk in env
        env.step(action=jerk_arr_step)

        # measure position, velocity and acceleration
        for idx_mover in range(0, num_movers):
            pos_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qpos(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]
            velo_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qvel(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]
            acc_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qacc(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]

            norm_velo = np.linalg.norm(velo_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2], ord=2)
            norm_acc = np.linalg.norm(acc_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2], ord=2)
            assert np.allclose(norm_velo, env.v_max) or float(norm_velo) < env.v_max
            assert np.allclose(norm_acc, env.a_max) or float(norm_acc) < env.a_max

    env.close()

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)


@pytest.mark.parametrize(
    'num_movers, mover_mass, acc, num_cycles, test_x, test_y',
    [
        (1, 0.628, 0.15, 1, True, True),
        (1, 0.628, 0.15, 1, True, False),
        (1, 0.628, 0.15, 1, False, True),
        (1, 1.237, 0.15, 1, True, True),
        (1, 0.628, -0.15, 1, True, True),
        (1, 0.628, -0.15, 1, True, False),
        (1, 0.628, -0.15, 1, False, True),
        (1, 1.237, -0.15, 1, True, True),
        (1, 0.628, 0.15, 42, True, True),
        (1, 1.237, 0.15, 42, True, True),
        (1, 0.628, -0.15, 42, True, True),
        (1, 1.237, -0.15, 42, True, True),
        (1, 0.628, 0.2, 1, True, True),
        (1, 0.628, 0.2, 1, True, False),
        (1, 0.628, 0.2, 1, False, True),
        (1, 1.237, 0.2, 1, True, True),
        (1, 0.628, -0.2, 1, True, True),
        (1, 0.628, -0.2, 1, True, False),
        (1, 0.628, -0.2, 1, False, True),
        (1, 1.237, -0.2, 1, True, True),
        (1, 0.628, 0.2, 42, True, True),
        (1, 1.237, 0.2, 42, True, True),
        (1, 0.628, -0.2, 42, True, True),
        (1, 1.237, -0.2, 42, True, True),
        (2, 0.628, 0.15, 1, True, True),
        (2, 0.628, 0.15, 1, True, False),
        (2, 0.628, 0.15, 1, False, True),
        (2, 1.237, 0.15, 1, True, True),
        (2, 0.628, -0.15, 1, True, True),
        (2, 0.628, -0.15, 1, True, False),
        (2, 0.628, -0.15, 1, False, True),
        (2, 1.237, -0.15, 1, True, True),
        (2, 0.628, 0.15, 42, True, True),
        (2, 1.237, 0.15, 42, True, True),
        (2, 0.628, -0.15, 42, True, True),
        (2, 1.237, -0.15, 42, True, True),
        (2, 0.628, 0.2, 1, True, True),
        (2, 0.628, 0.2, 1, True, False),
        (2, 0.628, 0.2, 1, False, True),
        (2, 1.237, 0.2, 1, True, True),
        (2, 0.628, -0.2, 1, True, True),
        (2, 0.628, -0.2, 1, True, False),
        (2, 0.628, -0.2, 1, False, True),
        (2, 1.237, -0.2, 1, True, True),
        (2, 0.628, 0.2, 42, True, True),
        (2, 1.237, 0.2, 42, True, True),
        (2, 0.628, -0.2, 42, True, True),
        (2, 1.237, -0.2, 42, True, True),
    ],
)
def test_acceleration_actuator(num_movers, mover_mass, acc, num_cycles, test_x, test_y):
    # environment
    v_max = 0.01  # [m/s]
    a_max = 0.2  # [m/s²]
    j_max = 150  # [m/s³]
    learn_jerk = False
    collision_shape = 'box'
    collision_size = np.array([0.113 / 2 + 1e-6, 0.113 / 2 + 1e-6])
    collision_offset = 0.0
    collision_offset_wall = 0.0

    mover_params = {'size': np.array([0.113 / 2, 0.113 / 2, 0.012 / 2]), 'mass': mover_mass}

    collision_params = {
        'shape': collision_shape,
        'size': collision_size,
        'offset': collision_offset,
        'offset_wall': collision_offset_wall,
    }

    env = StateBasedGlobalPushingEnv(
        num_movers=num_movers,
        layout_tiles=np.ones((9, 9)),
        mover_params=mover_params,
        std_noise=0.0,
        collision_params=collision_params,
        render_mode=None,
        render_every_cycle=False,
        num_cycles=num_cycles,
        v_max=v_max,
        a_max=a_max,
        j_max=j_max,
        learn_jerk=learn_jerk,
        use_mj_passive_viewer=False,
    )

    if num_movers == 1:
        mover_start_xy_pos = np.array([[0.72, 0.36]])
    elif num_movers == 2:
        mover_start_xy_pos = np.array([[0.96, 0.96], [1.2, 1.2]])
    env.reload_model(mover_start_xy_pos=mover_start_xy_pos)

    num_steps = 100
    dt = num_cycles
    timestep = env.model.opt.timestep

    pos_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    velo_mj_actuator = np.zeros((num_steps, 2 * num_movers))
    acc_mj_actuator = np.zeros((num_steps, 2 * num_movers))

    pos_mj_manual = np.zeros((num_steps, 2 * num_movers))
    velo_mj_manual = np.zeros((num_steps, 2 * num_movers))
    acc_mj_manual = np.zeros((num_steps, 2 * num_movers))

    for step in range(0, num_steps):
        if test_x and test_y:
            acc_arr = np.array([acc, acc] * num_movers)
        elif test_x and not test_y:
            acc_arr = np.array([acc, 0] * num_movers)
        elif not test_x and test_y:
            acc_arr = np.array([0, acc] * num_movers)
        else:
            acc_arr = np.array([0, 0] * num_movers)
        acc_arr_step = acc_arr.copy()

        next_a = acc_arr.reshape((env.num_movers, 2))
        # ensure maximum acceleration
        action_norm_tmp = np.linalg.norm(next_a, ord=2, axis=1)
        action_norm = np.where(action_norm_tmp <= env.a_max, 1.0, action_norm_tmp)[:, None]
        action_max_vals = np.where(action_norm == 1.0, 1.0, env.a_max)
        next_a = np.divide(next_a, action_norm) * action_max_vals

        for idx_mover in range(0, num_movers):
            if step > 0:
                v = velo_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
                p = pos_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
                a = acc_mj_manual[step - 1, 2 * idx_mover : 2 * idx_mover + 2].copy()
            else:
                v = np.zeros(2)
                p = mover_start_xy_pos[idx_mover, :].copy()
                a = np.zeros(2)

            for _ in range(0, dt):
                v, a_tmp = env.ensure_max_dyn_val(v, env.v_max, next_a[idx_mover, :])

                a = a_tmp.copy()
                p = timestep * v + p
            pos_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = p.flatten().copy()
            velo_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = v.flatten().copy()
            acc_mj_manual[step, 2 * idx_mover : 2 * idx_mover + 2] = a.flatten().copy()

        # set acc in env
        env.step(action=acc_arr_step)

        # measure position, velocity and acceleration
        for idx_mover in range(0, num_movers):
            pos_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qpos(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]
            velo_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qvel(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]
            acc_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2] = env.get_mover_qacc(
                mover_names=env.mover_names[idx_mover], add_noise=False
            )[0, :2]

            norm_velo = np.linalg.norm(velo_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2], ord=2)
            norm_acc = np.linalg.norm(acc_mj_actuator[step, 2 * idx_mover : 2 * idx_mover + 2], ord=2)
            assert np.allclose(norm_velo, env.v_max) or float(norm_velo) < env.v_max
            assert np.allclose(norm_acc, env.a_max) or float(norm_acc) < env.a_max

    env.close()

    assert np.allclose(pos_mj_manual, pos_mj_actuator)
    assert np.allclose(velo_mj_manual, velo_mj_actuator)
    assert np.allclose(acc_mj_manual, acc_mj_actuator)

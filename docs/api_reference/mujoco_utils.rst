.. _mujoco_utils:

MuJoCo Utils
============

MagBotSim offers various MuJoCo utility functions that can help to develop your custom 
MuJoCo environments.

.. autofunction:: magbotsim.utils.mujoco_utils.set_actuator_ctrl

.. autofunction:: magbotsim.utils.mujoco_utils.get_joint_qacc

.. autofunction:: magbotsim.utils.mujoco_utils.get_mujoco_type_names

.. autofunction:: magbotsim.utils.mujoco_utils.get_joint_addrs_and_ndims

.. note::
    The following functions and classes are completely or at least partially adopted from gymnasium-robotics:
    https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/utils/mujoco_utils.py

.. autofunction:: magbotsim.utils.mujoco_utils.set_joint_qpos

.. autofunction:: magbotsim.utils.mujoco_utils.get_joint_qpos

.. autofunction:: magbotsim.utils.mujoco_utils.set_joint_qvel

.. autofunction:: magbotsim.utils.mujoco_utils.get_joint_qvel

.. autofunction:: magbotsim.utils.mujoco_utils.extract_mj_names

.. autoclass:: magbotsim.utils.mujoco_utils.MujocoModelNames
  :members:
  :inherited-members:
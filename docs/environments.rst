.. _environments:

Environments
============

MagBotSim contains environments for object manipulation and trajectory planning with a MagLev system. In this context, the term ''environment'' does not only refer to RL environments,
but to (industrial) environments in general that include a MagLev system, since MagBotSim can be used without the RL API.

.. note::
   We will add new environments in the future based on our research.

A detailed documentation of all environments can be found in the following subsections:

.. toctree::
   :maxdepth: 1
   :caption: Object Manipulation Environments

   environments/state_based_global_pushing_env
   environments/state_based_push_t_env
   environments/state_based_push_x_env
   environments/state_based_push_l_env
   environments/state_based_push_box_env
   environments/state_based_static_obstacle_pushing_env

.. toctree::
    :maxdepth: 1
    :caption: Trajectory Planning Environments

    environments/long_horizon_global_trajectory_planning_env

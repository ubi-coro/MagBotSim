.. _choosing_parent_class:

Choosing a Parent Class for Your Custom Environment
===================================================

MagBotSim contains three basic environments that should be used as parent classes for custom environments: 

- :ref:`basic_magbot_env` (``BasicMagBotEnv``) 
- :ref:`basic_magbot_single_agent_env` (``BasicMagBotSingleAgentEnv``)
- :ref:`basic_magbot_multi_agent_env` (``BasicMagBotMultiAgentEnv``)

The ``BasicMagBotEnv`` provides functionality that is required in every RL environment with a MagLev system, such as collision checking 
and rendering functionality, as well as the possibility to generate a MuJoCo XML string from the current mover-tile-configuration. Note that the MuJoCo 
model can be customized by the user by defining additional XML strings that are automatically added by the ``BasicMagBotEnv``. For example, this 
allows adding custom actuators for the movers, sensors, or robots to the MuJoCo model, so that the user can build a simulation of a specific (industrial) application. 
The ``BasicMagBotEnv`` is designed independently from any RL API, since the main focus is on providing functionality that is related to the physics engine and the 
MagLev system.

To provide the user with a basic structure for single-agent and multi-agent RL, MagBotSim contains the ``BasicMagBotSingleAgentEnv`` and 
``BasicMagBotMultiAgentEnv`` that include the functionality of the ``BasicMagBotEnv``. A custom environment can be inherited from either 
the single-agent or the multi-agent environment. The multi-agent environment follows the PettingZoo parallel API, since all movers, i.e. agents, 
must be updated simultaneously in every control cycle. The single-agent environment follows the Gymnasium API and provides basic functionality for 
standard RL and goal-conditioned RL, such as a ``compute_reward`` method that takes the arguments ``achieved_goal`` and ``desired_goal``. This is a typical 
requirement of Hindsight Experience Replay (HER) implementations of common RL libraries, such as `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_  
or `Tianshou <https://tianshou.org/en/stable/>`_.

If you work without RL, choose the ``BasicMagBotEnv`` as the parent class for your environment. Otherwise, if you want to use RL, first decide whether you 
formulate your RL problem as a single or multi-agent problem. Note that many problems, even for environments containing multiple movers, can be 
formulated in both ways. For a single-agent formulation, use the ``BasicMagBotSingleAgentEnv``, and for a multi-agent formulation, use the ``BasicMagBotMultiAgentEnv`` 
as the parent class.
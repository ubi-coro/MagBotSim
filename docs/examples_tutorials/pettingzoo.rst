PettingZoo API for Multi-Agent Reinforcement Learning
=====================================================

MagBotSim supports both single-agent and multi-agent reinforcement learning
scenarios. While the library includes pre-built examples for single-agent
reinforcement learning using the Gymnasium API, it also provides seamless
integration with PettingZoo's Parallel API for multi-agent reinforcement
learning (MARL).

Creating a Multi-Agent Environment
----------------------------------

To create a custom multi-agent environment, inherit from :ref:`basic_magbot_multi_agent_env` (``BasicMagBotMultiAgentEnv``)
and implement the required PettingZoo methods. The ``BasicMagBotMultiAgentEnv`` class
already handles the basic setup by inheriting from both :ref:`basic_magbot_env` (``BasicMagBotEnv``) and PettingZoo's
`ParallelEnv <https://pettingzoo.farama.org/api/parallel/#parallelenv>`_, and sets up the ``agents`` and ``possible_agents`` properties using the mover names.

Define a minimal multi-agent environment by subclassing
:class:`BasicMagBotMultiAgentEnv`:

.. code-block:: python

    import mujoco
    import numpy as np
    from gymnasium import spaces

    from magbotsim import BasicMagBotMultiAgentEnv
    from magbotsim.utils import mujoco_utils


    class SimpleMultiAgentEnv(BasicMagBotMultiAgentEnv):
        """A minimal multi-agent environment where agents move around freely."""

        def __init__(self, num_movers=3, max_steps=200, **kwargs):
            # Create a simple 3x3 tile layout
            layout_tiles = np.ones((3, 3))

            super().__init__(layout_tiles=layout_tiles, num_movers=num_movers, render_mode=kwargs.get('render_mode', 'human'), **kwargs)

            self.max_steps = max_steps
            self.current_step = 0
            self.max_acceleration = 5.0

            self._setup_actuators()

Here we define a callback to customize the MuJoCo XML model. This allows us to
programmatically add actuators for each agent, specifying which joints they
control and how they respond to actions:

.. code-block:: python

    def _custom_xml_string_callback(self, custom_model_xml_strings=None):
        """Add actuators for the movers."""
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}

        custom_outworldbody_xml_str = ''

        if hasattr(self, 'mover_joint_names'):
            actuator_lines = ['\n\n\t<actuator>']
            dof_names = ['x', 'y']  # Only x, y for 2D movement

            for idx_mover in range(self.num_movers):
                mover_lines = [f'\n\t\t<!-- actuators mover {idx_mover} -->']

                for idx_dof, dof_name in enumerate(dof_names):
                    str_gear = '0 0 0 0 0 0'
                    str_gear = str_gear[: 2 * idx_dof] + '1' + str_gear[2 * idx_dof + 1 :]
                    actuator_name = f'mover_actuator_{dof_name}_{idx_mover}'
                    mover_lines.append(
                        f'\n\t\t<general name="{actuator_name}" joint="{self.mover_joint_names[idx_mover]}" '
                        f'gear="{str_gear}" dyntype="none" gaintype="fixed" gainprm="1 0 0" biastype="none"/>'
                    )

                actuator_lines.append(''.join(mover_lines))

            actuator_lines.append('\n\t</actuator>')
            custom_outworldbody_xml_str += ''.join(actuator_lines)

        custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str
        return custom_model_xml_strings

Next, we build the MuJoCo model and store the actuator names for each agent.
This ensures we can reference and control the actuators efficiently during
simulation steps:

.. code-block:: python

        def _setup_actuators(self):
            """Set up actuators by reloading the model."""
            self.reload_model()
            self.reload_model()

            self.mover_actuator_x_names = mujoco_utils.get_mujoco_type_names(
                self.model, obj_type='actuator', name_pattern='mover_actuator_x'
            )
            self.mover_actuator_y_names = mujoco_utils.get_mujoco_type_names(
                self.model, obj_type='actuator', name_pattern='mover_actuator_y'
            )

Define the action and observation spaces for each agent. Here, each agent can
control acceleration in 2D and observes its position and velocity:

.. code-block:: python

        @property
        def action_spaces(self):
            """Each agent controls 2D acceleration."""
            action_space = spaces.Box(
                low=-self.max_acceleration,
                high=self.max_acceleration,
                shape=(2,),  # x, y acceleration
                dtype=np.float32,
            )
            return {agent: action_space for agent in self.agents}

        @property
        def observation_spaces(self):
            """Each agent observes its position and velocity."""
            # [pos_x, pos_y, vel_x, vel_y]
            obs_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32,
            )
            return {agent: obs_space for agent in self.agents}

Implement the ``reset`` method to initialize the environment state. This places
all agents in their starting positions and velocities and prepares the
observation dictionary:

.. code-block:: python

        def reset(self, seed=None, options=None):
            """Reset the environment."""
            super().reset()
            self.current_step = 0

            observations = {}
            for agent in self.agents:
                pos = self.get_mover_qpos(agent)[0,:2]
                vel = self.get_mover_qvel(agent)[0,:2]
                observations[agent] = np.concatenate([pos, vel]).astype(np.float32)

            infos = {agent: {} for agent in self.agents}
            return observations, infos

Implement the ``step`` method to advance the environment. Actions are applied to
the actuators, the simulation runs for multiple cycles, and new observations and
rewards are collected:

.. code-block:: python

        def step(self, actions):
            """Execute one step."""
            self.current_step += 1

            # Run the simulation for multiple cycles (like in single-agent environments)
            for _ in range(40):  # num_cycles = 40 by default
                self._apply_actions(actions)
                mujoco.mj_step(self.model, self.data, nstep=1)

            # Get observations
            observations = {}
            rewards = {}
            for agent in self.agents:
                pos = self.get_mover_qpos(agent)[0,:2]
                vel = self.get_mover_qvel(agent)[0,:2]
                observations[agent] = np.concatenate([pos, vel]).astype(np.float32)
                rewards[agent] = 0.0

            terminations = {agent: False for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {} for agent in self.agents}

            return observations, rewards, terminations, truncations, infos

Finally, define a helper to apply agent actions to actuators.

.. code-block:: python

        def _apply_actions(self, actions):
            """Apply acceleration actions to each mover."""
            for i, agent in enumerate(self.agents):
                if agent in actions:
                    # Clip actions to valid range
                    action = np.clip(actions[agent], -self.max_acceleration, self.max_acceleration)

                    # Apply acceleration to x and y actuators
                    mujoco_utils.set_actuator_ctrl(
                        model=self.model, data=self.data, actuator_name=self.mover_actuator_x_names[i], value=action[0]
                    )
                    mujoco_utils.set_actuator_ctrl(
                        model=self.model, data=self.data, actuator_name=self.mover_actuator_y_names[i], value=action[1]
                    )

Usage Examples
--------------

**Single-Agent (Gymnasium API):**

.. code-block:: python

    import gymnasium as gym
    import magbotsim

    env = gym.make("StateBasedPushBoxEnv-v0")
    observation, info = env.reset()

    for step in range(100):
        action = env.action_space.sample()  # Single action
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

**Multi-Agent (PettingZoo API):**

.. code-block:: python

    env = SimpleMultiAgentEnv(num_movers=3)
    observations, infos = env.reset()

    for step in range(100):
        actions = {agent: env.action_spaces[agent].sample()
                    for agent in env.agents}  # Dict of actions
        observations, rewards, terminated, truncated, infos = env.step(actions)
        if any(terminated.values()) or any(truncated.values()):
            break

The key difference is that PettingZoo uses dictionaries for all agent-related data,
while Gymnasium uses single values for the single agent.

For detailed implementation guidance, see the `PettingZoo documentation <https://pettingzoo.farama.org/api/parallel/>`_.

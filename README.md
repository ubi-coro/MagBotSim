# MagBotSim (Magnetic Robotics Simulation)

<img src="https://github.com/ubi-coro/MagBotSim/raw/main/docs/images/visual_abstract.png" />

MagBotSim is a library for physics-based simulation environments for motion planning 
and object manipulation in the field of Magnetic Robotics. The main component of every environment is a 
*magnetic levitation (MagLev)* system, which consists of two basic components, as shown in the Figure above. Firstly, 
dynamically actuated shuttles as passive motor modules, so-called *movers*, consist of a housing and a complex permanent 
magnet structure based on Halbach arrays on the lower side of the mover. Secondly, static motor modules, so-called *tiles*, 
are the active component of the drive system. As shown in the Figure above, the tiles enable the coil-induced emission 
of electromagnetic fields (yellow) that interact with the mover's field (blue). During operation, the movers hover above the 
tiles and can be controlled in six dimensions by adjusting the currents in the coils contained in the tiles. 

MagBotSim is designed to match real-world applications, so that control policies can be transferred to real MagLev systems without 
further training or calibration. Since the library is based on the [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) 
physics engine, MagBotSim enables users to perform object manipulation tasks with MagLev systems. In addition, as reinforcement 
learning (RL) is frequently used in multi-agent path finding and object manipulation, MagBotSim includes basic environments with 
single-agent ([Gymnasium](https://gymnasium.farama.org/)) and multi-agent ([PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)) RL 
APIs that can serve as starting points for developing new research-specific environments. However, MagBotSim can also be 
used without RL and provides several utilities, such as impedance control for the movers.

## Installation
The MagBotSim package can be installed via PIP:
```
pip install magbotsim
```
To install optional dependencies, to build the documentation, or to run the tests, use:
```
pip install magbotsim[docs, tests]
```
**Note:** Depending on your shell (e.g. when using Zsh), you may need to use additional quotation marks: 
```
pip install "magbotsim[docs, tests]"
```

## Documentation
The documentation is available at: https://ubi-coro.github.io/MagBotSim/

## License
MagBotSim is published under the GNU General Public License v3.0.

## Reinforcement Learning Example
The following example shows how to use a trained policy with an environment that follows the Gymnasium API:

```python
import gymnasium as gym
import magbotsim

gym.register_envs(magbotsim)

mover_params = {
    'shape': 'mesh',
    'mesh': {'mover_stl_path': 'beckhoff_apm4220_mover', 'bumper_stl_path': 'beckhoff_apm4220_bumper'},
    'mass': 0.639 - 0.034,
    'bumper_mass': 0.034,
}
env_kwargs = {
    'mover_params': mover_params,
    'initial_mover_zpos': 0.002,
    'render_mode': 'human',
    'render_every_cycle': True,
}

env = gym.make("StateBasedPushTEnv-v0", **env_kwargs)
observation, info = env.reset(seed=42)

for _ in range(0,100):
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()  # use custom policy instead
        observation, reward, terminated, truncated, info = env.step(action)

    observation, info = env.reset()
env.close()
```

## Maintainer
MagBotSim is currently maintained by Lara Bergmann (@lbergmann1).

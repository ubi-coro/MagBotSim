.. _magbots:

MagBots
=======

.. raw:: html

   <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">

      <iframe width="640" height="360"
            src="https://www.youtube.com/embed/gAb7l0zvhG0?autoplay=1&mute=1&loop=1&playlist=gAb7l0zvhG0"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

      <iframe width="640" height="360"
            src="https://www.youtube.com/embed/ipE8ZpbkTMI?autoplay=1&mute=1&loop=1&playlist=ipE8ZpbkTMI"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

   </div>
   <br>

The movers of a MagLev system can be coupled into a composite robotic platform, i.e. MagBots, thereby 
expanding the workspace, payload, and functional dexterity of the MagLev system. Additional feeding
technology is commonly used in a machine to compensate for the limited workspace of MagLev systems. However,
feeding technologies, such as robots, are expensive and slow compared to a MagLev mover. In contrast, MagBots are 
low-cost, can utilize the full dynamics of the MagLev movers, and expand the task-specific capabilities of
machines without requiring additional feeding technology. Moreover, MagBots can reduce product processing times, 
as product transport and processing steps can be carried out simultaneously, e.g.
mixing liquids during transport. Therefore, the application of reconfigurable MagBots significantly reduces the costs and
footprint of a machine, offering a cost-effective alternative to traditional machines that rely on feeding mechanisms.

The MuJoCo models and the inverse kinematics controllers of the following MagBots are currently integrated into MagBotSim:

.. toctree::
   :maxdepth: 1

   magbots/sixD_platform

Examples and Tutorials
----------------------
A tutorial on how to integrate MagBots to a custom environment can be found here: :ref:`magbot_tutorial`.
Example environments with MagBots can be found `here <https://github.com/ubi-coro/MagBotSim/blob/main/magbotsim/envs_without_rl_api/magbots>`_.

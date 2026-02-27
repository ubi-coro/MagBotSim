.. _sixD_platform_magbot:

6D-Platform MagBot
==================

.. raw:: html

   <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">

      <iframe width="888" height="500"
            src="https://www.youtube.com/embed/2AJgtwbnyGg?autoplay=1&mute=1&loop=1&playlist=2AJgtwbnyGg"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

   </div>
   <br>

To expand the workspace of a MagLev system in all dimensions, payload and functional dexterity compared to
a single mover, the 6D-Platform MagBot has been developed - a low-cost parallel kinematic with six DoF that 
couples two movers into a composite robotic platform. The 6D-Platform MagBot is designed to be
compatible with all industrial MagLev systems. Additionally, the MagBot can be autonomously dropped off or 
picked up by the mover using a docking station. The platform of the MagBot is controlled by the x-, y-, and :math:`\gamma`-axes of the movers.
To enable the development of motion planning approaches for Magnetic Robotics, a MuJoCo model of the 6D-Platform MagBot 
is integrated in the MagBotSim together with the inverse kinematics controller.

API References
--------------
The API references can be found here: :ref:`sixD_platform_magbot_api_reference`.

.. note::
    This MagBots class assumes that the MagBot is used with a Beckhoff XPlanar system and two APM4330-0000-0000 movers, because 
    the holes for the screws of mesh files "mover_to_mount_a.STL" and "mover_to_mount_a.STL" are designed for these types of movers.
    However, the 6D-Platform MagBot can also be used with other MagLev systems provided the movers have a similar or larger payload 
    capacity. Only the holes for the screws of the aforementioned mesh files may need to be adjusted.


Component List, CAD Files, and Assembly Instructions
----------------------------------------------------
A complete list of all components, CAD files, and assembly instructions can be found on the `website <https://sites.google.com/view/6d-platform-magbot?usp=sharing>`_.

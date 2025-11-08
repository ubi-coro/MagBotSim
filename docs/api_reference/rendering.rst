Rendering
=========

MagBotSim offers various viewers for 2D and 3D rendering as well as off-screen rendering.

3D Viewer and Off-Screen Rendering
----------------------------------
.. autoclass:: magbotsim.utils.rendering.MujocoViewerCollection
    :members:
    :inherited-members:

.. autoclass:: magbotsim.utils.rendering.MujocoOffScreenViewer
    :members:
    :inherited-members:

2D Matplotlib Viewer
--------------------
.. note::
    The 2D viewer is intended for debugging and analyzing certain situations during motion planning. 
    Therefore, it is not a 2D equivalent to the 3D viewer, but rather a simplified version of it that only 
    displays the tile layout, the movers, and their collision shapes/offsets. Other objects that may be 
    part of an environment are not displayed.


.. autoclass:: magbotsim.utils.rendering.Matplotlib2DViewer
  :members:
  :inherited-members:
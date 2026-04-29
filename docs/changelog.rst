Changelog
=========

Pre-Release v1.0.4a0 (2026-04-29)
---------------------------------

General
^^^^^^^
1. The ``_step_callback`` of the ``BasicMagBotSingleAgentEnv`` now returns the possibly modified action instead of None.

Bug Fixes
^^^^^^^^^
1. Fix wrong mover bumper mass and corresponding examples in the documentation
2. Fix wrong gainprm in all RL environments
3. Ensure correct maximum dynamics of the actions (RL environments). This is fixed by using the ``_step_callback``.

Release v1.0.3 (2026-03-25)
---------------------------

General
^^^^^^^
1. Fix bug in ``StateBasedGlobalPushingEnv-v0`` 
2. Update RL examples

Release v1.0.2 (2026-02-27)
---------------------------

General
^^^^^^^
1. Add the 6D-Platform MagBot (``SixDPlatformMagBotsAPM4330``)
2. Add tutorial on how to add MagBots to custom environments
3. Add example environments with MagBots (``SixDPlatformMagBotApplicationExampleEnv`` and ``SixDPlatformMagBotExampleEnv``)

Release v1.0.1 (2025-11-24)
---------------------------

General
^^^^^^^
1. Use MuJoCo functions in ``MoverImpedanceController`` instead of ``scipy.spatial.transform.Rotation`` (better performance, but ``scipy.spatial.transform.Rotation`` is not completely replaced)
2. Add benchmark environments
3. Add ``StateBasedPushXEnv-v0`` and ``StateBasedPushLEnv-v0``
4. Update throughput calculation in pushing environments and update benchmark results accordingly

Release v1.0.0 (2025-11-08)
---------------------------

General
^^^^^^^
1.  Added the basic environments: ``BasicMagBotEnv``, ``BasicMagBotSingleAgentEnv``, ``BasicMagBotMultiAgentEnv``
2.  Added the following example environments for trajectory planning and object manipulation:
    ``LongHorizonGlobalTrajectoryPlanningEnv-v0``, ``StateBasedStaticObstaclePushingEnv-v0``, ``StateBasedGlobalPushingEnv-v0``,
    ``StateBasedPushBoxEnv-v0``, ``StateBasedPushTEnv-v0``
3.  Added a ``MoverImpedanceController`` that solves a position and orientation task
4.  Added the ``MujocoOffScreenViewer``: an extension of the Gymnasium OffScreenViewer that allows to also specify the groups
    of geoms to be rendered by the off-screen renderer
5.  Added the ``MujocoViewerCollection``: a manager for all renderers in a MuJoCo environment
6.  Added the ``Matplotlib2DViewer``: a simple viewer that displays the tile and mover configuration together with the mover
    collision offsets for debugging and analyzing planning tasks
7.  Added auxiliary functions for MuJoCo, collision checking and rotations
8.  Added benchmarks and latest results for trajectory planning and object pushing
9.  Added tutorials and examples on how to use MagBotSim

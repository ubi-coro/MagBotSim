Changelog
=========

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

Benchmarks
==========

In order to make future research results comparable, we propose several task-specific benchmarks that 
allow users to measure the performance of suggested motion planning algorithms for object manipulation 
and trajectory planning in the context of magnetic levitation systems. To simplify measurements for users, 
MagBotSim includes auxiliary functions (see :ref:`benchmark_utils`) to measure specific properties, e.g. 
corrective movements. In the following, :math:`n_1, n_2, n_3, n_4\in\mathbb{N}` denote the total number 
of goals or the number of successfully reached goals used in the measurements.

.. note::
    Since MagBotSim currently contains examples for object pushing and trajectory planning, the following
    benchmarks focus on these two tasks. However, we will add new environments to MagBotSim in the future
    and update the benchmarks accordingly.

Benchmarks Object Pushing
-------------------------
We propose the following benchmarks for object pushing tasks in which the object has to be pushed to a
specific goal. The goal can either be a position or a pose (position and orientation). We distinguish
two types of corrective movements, as suggested in our previous work
`Precision-Focused Reinforcement Learning Model for Robotic Object Pushing <https://doi.org/10.48550/arXiv.2411.08622>`_:
overshoot and distance corrections. If the distance between the object and target is smaller than the position
threshold in time step :math:`t` and larger than the position threshold in the next time step :math:`t+1`, a corrective
movement is necessary to push the object back to the target area. Therefore, these corrections are referred
to as *overshoot corrections*. If the distance between the object and goal increases within one time
step and decreases later within the episode, this correction is considered as the second type of corrective
movement, i.e. a *distance correction*. Our latest results can be found here: :ref:`results_object_pushing`.

===================== ================================================================================================================
Metric                Measurement                                                                                
===================== ================================================================================================================
Success Rate          Number of successfully reached goals divided by the total number of goals :math:`n_1`.                 
Throughput            Number of successfully reached goals :math:`n_2` divided by the time required to reach these goals.
Overshoot Corrections Mean number of overshoot corrections for successfully reached goals :math:`n_2`.
Distance Corrections  Mean number of distance corrections for successfully reached goals :math:`n_2`.
                      Can be measured using :ref:`benchmark_utils_corrective_movements`.
Collisions            Mean number of collisions for the total number of goals :math:`n_1` (including any obstacle - object collisions). 
                      Of these:
                      
                      - **Mover-Mover Collisions**: Mean number of mover-mover collisions if more than one mover is used.
                      - **Mover-Obstacle Collisions**: Mean number of collisions between a mover and any static obstacle.        

We have created 10 benchmark environments (B0-B9) for pushing tasks, each with a defined
number of movers and tile configurations.

.. list-table::
   :header-rows: 1

   * - Benchmark Environment
     - Environment Image
     - Environment Shape (in Tiles)
     - Number of Movers
   * - B0
     - .. image:: _static/benchmark_envs/b0_pushing.png
     - (3, 3)
     - 1
   * - B1
     - .. image:: _static/benchmark_envs/b1_pushing.png
     - (4, 3)
     - 1
   * - B2
     - .. image:: _static/benchmark_envs/b2_pushing.png
     - (8, 5)
     - 1
   * - B3
     - .. image:: _static/benchmark_envs/b3_pushing.png
     - (7, 7)
     - 2
   * - B4
     - .. image:: _static/benchmark_envs/b4_pushing.png
     - (8, 6)
     - 2
   * - B5
     - .. image:: _static/benchmark_envs/b5_pushing.png
     - (11, 6)
     - 2
   * - B6
     - .. image:: _static/benchmark_envs/b6_pushing.png
     - (8, 6)
     - 2
   * - B7
     - .. image:: _static/benchmark_envs/b7_pushing.png
     - (7, 7)
     - 2
   * - B8
     - .. image:: _static/benchmark_envs/b8_pushing.png
     - (8, 6)
     - 2
   * - B9
     - .. image:: _static/benchmark_envs/b9_pushing.png
     - (8, 8)
     - 2

Benchmarks Trajectory Planning
------------------------------
We propose the following benchmarks for trajectory planning tasks in which one or typically many movers have
to reach varying goals without leaving tiles or colliding with static or dynamic obstacles. Similar to the
object pushing task, a mover-specific goal can either be a position or a pose (position and orientation).
:math:`n\in\mathbb{N}` denotes the number of movers. Our latest results can be found here: :ref:`results_trajectory_planning`.

===================== ==============================================================================================================================================
Metric                Measurement                                                                              
===================== ==============================================================================================================================================
Success Rate          Number of successfully reached goals divided by the total number of goals :math:`n_1`.                 
Makespan              Number of milliseconds required by the slowest mover to reach the total number of goals :math:`n_2`.
Throughput            Number of milliseconds required by all movers to reach the total number of goals :math:`n_3`.
Collisions            Number of collisions for the total number of goals :math:`n_1`. Of these:

                      - **Mover-Mover Collisions**: Number of mover-mover collisions if more than one mover is used.
                      - **Mover-Obstacle Collisions**: Number of collisions between a mover and any static obstacle.
Smoothness            Minimum, maximum, and mean weighted sum of jerk, acceleration, and velocity of all movers within the total time period :math:`t\in\mathbb{N}`.
                      Can be measured :ref:`benchmark_utils_energy_efficiency_measurement`.
Process Time          Process time required by all movers to successfully reach the total number of goals :math:`n_4`.
===================== ==============================================================================================================================================

We have created 10 benchmark environments (B0-B9) for trajectory planning tasks, each with a defined
number of movers and tile configurations.

.. list-table::
   :header-rows: 1

   * - Benchmark Environment
     - Environment Image
     - Environment Shape (in Tiles)
     - Number of Movers
   * - B0
     - .. image:: _static/benchmark_envs/b0_planning.png
     - (4, 3)
     - 3
   * - B1
     - .. image:: _static/benchmark_envs/b1_planning.png
     - (4, 3)
     - 3
   * - B2
     - .. image:: _static/benchmark_envs/b2_planning.png
     - (8, 5)
     - 4
   * - B3
     - .. image:: _static/benchmark_envs/b3_planning.png
     - (7, 7)
     - 4
   * - B4
     - .. image:: _static/benchmark_envs/b4_planning.png
     - (8, 6)
     - 4
   * - B5
     - .. image:: _static/benchmark_envs/b5_planning.png
     - (11, 6)
     - 3
   * - B6
     - .. image:: _static/benchmark_envs/b6_planning.png
     - (8, 6)
     - 4
   * - B7
     - .. image:: _static/benchmark_envs/b7_planning.png
     - (7, 7)
     - 4
   * - B8
     - .. image:: _static/benchmark_envs/b8_planning.png
     - (8, 6)
     - 5
   * - B9
     - .. image:: _static/benchmark_envs/b9_planning.png
     - (8, 8)
     - 5

Latest Results
==============
.. toctree::
    :maxdepth: 1

    benchmarks/results_object_pushing
    benchmarks/results_trajectory_planning

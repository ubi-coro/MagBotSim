.. _results_trajectory_planning:

Latest Trajectory Planning Results
==================================

Simulation
----------
============================= ========================= ========================================================================================================================
Metric, Algorithm, Parameters :math:`n_1,n_2,n_3,n_4,t` Long Horizon Trajectory Planning                         
============================= ========================= ========================================================================================================================
Number of Movers              ---                       3
Number of Measurements        ---                       100
Success Rate (%)              :math:`15000`             :math:`99.3\pm0.001`                   
Makespan (ms)                 :math:`1000`              :math:`511463.58\pm11108.03`         
Throughput (ms)               :math:`3000`              :math:`495261.89\pm7345.36`           
Collisions                    :math:`15000`             :math:`32.09\pm8.49`             
Mover-Mover Collisions        :math:`15000`             :math:`26.45\pm7.66`                            
Mover-Obstacle Collisions     :math:`15000`             :math:`5.64\pm2.01`
Smoothness                    :math:`1000` ms           :math:`4610.97\pm490.87`
Process Time                  :math:`300`               :math:`0.1178\pm0.0035`
Link to MagBotSim Environment ---                       :ref:`lh_global_trajectory_planning_env`
Benchmark Configuration       ---                       B0
Algorithm                     ---                       Soft Actor-Critic + Hindsight Experience Replay        
Library                       ---                       TorchRL        
Link to Parameters            ---                       `Planning Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/trajectory_planning_sac_her_B0.yaml>`_
============================= ========================= ========================================================================================================================

Real System (Sim2Real)
----------------------
============================= ========================= ========================================================================================================================
Metric, Algorithm, Parameters :math:`n_1,n_2,n_3,n_4,t` Long Horizon Trajectory Planning                         
============================= ========================= ========================================================================================================================
Number of Movers              ---                       3
Number of Measurements        ---                       3
Success Rate (%)              :math:`15000`             :math:`99.97\pm0.02`                   
Makespan (ms)                 :math:`1000`              :math:`530970.33\pm14461.78`         
Throughput (ms)               :math:`3000`              :math:`502553.00\pm6412.13`           
Collisions                    :math:`15000`             :math:`0.0\pm0.0`             
Mover-Mover Collisions        :math:`15000`             :math:`0.0\pm0.0`                            
Mover-Obstacle Collisions     :math:`15000`             :math:`0.0\pm0.0`
Link to MagBotSim Environment ---                       :ref:`lh_global_trajectory_planning_env`
Benchmark Configuration       ---                       B0
Algorithm                     ---                       Soft Actor-Critic + Hindsight Experience Replay        
Library                       ---                       TorchRL        
Link to Training Parameters   ---                       `Planning Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/trajectory_planning_sac_her_B0.yaml>`_
MagLev System                 ---                       Beckhoff XPlanar
============================= ========================= ========================================================================================================================
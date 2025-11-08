.. _results_trajectory_planning:

Latest Trajectory Planning Results
==================================

============================= ===========================================================================================================================
Metric, Algorithm, Parameters Long Horizon Trajectory Planning                         
============================= ===========================================================================================================================
Number of movers              3
Success Rate (%)              :math:`99.3\pm0.001`                   
Throughput (ms)               :math:`511463.58\pm11108.03`         
Makespan (ms)                 :math:`495261.89\pm7345.36`           
Collisions                    :math:`32.09\pm8.49`             
Mover-Mover Collisions        :math:`26.45\pm7.66`                            
Mover-Obstacle Collisions     :math:`5.64\pm2.01`
Smoothness                    :math:`4610.97\pm490.87`
Process Time                  :math:`0.1178\pm0.0035`
Link to MagBotSim Environment :ref:`lh_global_trajectory_planning_env`
Algorithm                     Soft Actor-Critic + Hindsight Experience Replay        
Library                       TorchRL        
Link to Parameters            `Planning Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/trajectory_planning_sac_nMovers_3.yaml>`_
============================= ===========================================================================================================================
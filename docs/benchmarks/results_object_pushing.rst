.. _results_object_pushing:

Latest Object Pushing Results
=============================

Push-Box
--------
=============================== =============== ============================================================================================================= 
Metric, Algorithm, Parameters   :math:`n_1,n_2` Push-Box                                                                                                      
=============================== =============== ============================================================================================================= 
Number of Movers                ---             1 
Number of Measurements          ---             100                                                                                                           
Success Rate (%)                :math:`200`     :math:`99.56\pm0.44`                                                                                                 
Throughput (s)                  :math:`100`     :math:`1.1829\pm0.0331`                                                                                       
Overshoot Corrections           :math:`100`     :math:`0.0061\pm0.0081`                                                                                        
Distance Corrections            :math:`100`     :math:`0.2822\pm0.0611`                                                                                         
Collisions                      :math:`200`     :math:`0.002\pm0.0033`                                                                                        
Mover-Mover Collisions          :math:`200`     ---                                                                                                           
Mover-Obstacle Collisions       :math:`200`     :math:`0.002\pm0.0033`                                                                                   
Link to MagBotSim Environment   ---             :ref:`state_based_push_box_env`                                                                               
Benchmark Configuration         ---             B0                                                                                                            
Algorithm                       ---             Soft Actor-Critic + Hindsight Experience Replay                                                               
Library                         ---             Stable-Baselines3                                                                                             
Link to Parameters and Versions ---             `Push-Box Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/push_box_sac_her_B0.yaml>`_  
=============================== =============== ============================================================================================================= 

Push-T
------
=============================== =============== =========================================================================================================
Metric, Algorithm, Parameters   :math:`n_1,n_2` Push-T                                                                                                   
=============================== =============== =========================================================================================================
Number of Movers                ---             1
Number of Measurements          ---             100                                                                                                
Success Rate (%)                :math:`200`     :math:`73.13\pm2.96`                                                                                            
Throughput (s)                  :math:`100`     :math:`0.5099\pm0.0172`                                                                                 
Overshoot Corrections           :math:`100`     :math:`0.3241\pm0.0546`                                                                                    
Distance Corrections            :math:`100`     :math:`2.0763\pm0.119`                                                                                   
Collisions                      :math:`200`     :math:`0.0046\pm0.0041`                                                                                   
Mover-Mover Collisions          :math:`200`     ---                                                                                                      
Mover-Obstacle Collisions       :math:`200`     :math:`0.0046\pm0.0041`                                                                                  
Link to MagBotSim Environment   ---             :ref:`state_based_push_t_env`                                                                            
Benchmark Configuration         ---             B0                                                                                                       
Algorithm                       ---             Soft Actor-Critic + Hindsight Experience Replay                                                          
Library                         ---             Stable-Baselines3                                                                                        
Link to Parameters and Versions ---             `Push-T Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/push_t_sac_her_B0.yaml>`_
=============================== =============== =========================================================================================================

Push-Box with Static Obstacles
------------------------------
=============================== =============== =======================================================================================================================================
Metric, Algorithm, Parameters   :math:`n_1,n_2` Push with Obstacles
=============================== =============== =======================================================================================================================================
Number of Movers                ---             1
Number of Measurements          ---             100
Success Rate (%)                :math:`200`     :math:`91.26\pm1.92`
Throughput (s)                  :math:`100`     :math:`0.9665\pm0.0322`
Overshoot Corrections           :math:`100`     :math:`0.0036\pm0.0061`
Distance Corrections            :math:`100`     :math:`0.1346\pm0.0394`
Collisions                      :math:`200`     :math:`0.0451\pm0.0129`
Mover-Mover Collisions          :math:`200`     ---
Mover-Obstacle Collisions       :math:`200`     :math:`0.007\pm0.006`
Link to MagBotSim Environment   ---             :ref:`state_based_pushing_with_static_obstacle_env`
Benchmark Configuration         ---             B0                                      
Algorithm                       ---             Soft Actor-Critic + Hindsight Experience Replay
Library                         ---             Stable-Baselines3
Link to Parameters and Versions ---             `Push with Obstacles Params <https://github.com/ubi-coro/MagBotSim/blob/main/docs/parameters/pushing_static_obstacle_sac_her_B0.yaml>`_
=============================== =============== =======================================================================================================================================
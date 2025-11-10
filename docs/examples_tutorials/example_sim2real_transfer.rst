Sim2Real Transfer
=================

.. raw:: html

    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">

       <iframe width="320" height="180"
            src="https://www.youtube.com/embed/o_LCuxNir2w?autoplay=1&mute=1&loop=1&playlist=o_LCuxNir2w"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>

    </div>
    <br>

Control policies trained with MagBotSim can be transferred to a real MagLev system.
We provide an example for using a control policy trained with MagBotSim in real-time 
to control the movers of a Beckhoff XPlanar system and TwinCAT3 in 
`this repository <https://github.com/ubi-coro/magbotsim-sim2real-example>`_.
Since the policy needs to be converted to ONNX, we also provide the following example scripts:

- Stable-Baselines3 SAC to ONNX
- TorchRL SAC to ONNX

SONIC-A3 Documentation
======================

.. image:: https://img.shields.io/badge/License-Apache%202.0%20%7C%20NVIDIA%20Open%20Model-blue.svg
   :target: resources/license.html
   :alt: License

.. image:: https://img.shields.io/badge/IsaacLab-2.3.0-blue.svg
   :target: https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0
   :alt: IsaacLab

Welcome to the documentation for **SONIC-A3**, an unofficial personal research
derivative of NVIDIA's GR00T Whole-Body Control / SONIC stack. This project is
not affiliated with or endorsed by NVIDIA or Zhiyuan Robotics.

Many pages below are retained upstream references for G1, not validated A3
capabilities or proof that removed deployment packages are bundled here.

A3 project entry points
----------------------

* `Current scope and status <https://github.com/ytzdolp/SONIC-A3-public/blob/main/README.md>`_
* `A3 installation <https://github.com/ytzdolp/SONIC-A3-public/blob/main/docs/a3/INSTALLATION.md>`_
* `Results and evidence limits <https://github.com/ytzdolp/SONIC-A3-public/blob/main/docs/a3/RESULTS.md>`_
* `Assets and licenses <https://github.com/ytzdolp/SONIC-A3-public/blob/main/docs/a3/ASSETS_AND_LICENSES.md>`_
* `Outstanding publication items <https://github.com/ytzdolp/SONIC-A3-public/blob/main/docs/a3/PUBLICATION_FOLLOWUP.md>`_

What is SONIC-A3?
-----------------

This codebase adapts selected upstream motion representation, training, and
control components to the Zhiyuan A3 29-DoF contract. The current A3 result is
a simulation-only episodic locomotion research baseline; it is not a validated
real-robot release.

The upstream codebase provides:

- **Decoupled WBC** models used in NVIDIA Isaac-Gr00t, Gr00t N1.5 and N1.6 (see :doc:`detailed reference <references/decoupled_wbc>`)
- **GEAR-SONIC Series**: State-of-the-art controllers from the GEAR team

Upstream GEAR-SONIC news (retained for reference)
-------------------------------------------------

- **[06/16]** **Isaac Teleop Setup (CloudXR / DeviceIO, in-process)** — added bring-up docs for the in-process CloudXR path via ``isaacteleop[cloudxr]``, with no separate publisher container. See `Isaac Teleop Setup <tutorials/isaac_teleop_publisher_setup.html>`_.
- **[2026-06-16]** **Low-latency teleoperation checkpoint** — released a SONIC checkpoint with 4-frame SMPL reference lookahead for more responsive whole-body teleoperation. See the `Model Card <model_card.html>`_, `Download Models <getting_started/download_models.html#low-latency-teleoperation-checkpoint>`_, and `VLA Inference <tutorials/vla_inference.html#low-latency-teleoperation-checkpoint>`_ for usage.
- **[2026-05-07]** **End-to-end VLA workflow on G1** — collect teleop data, fine-tune Isaac-GR00T N1.7, and deploy with SONIC whole-body control. See `Data Collection <tutorials/data_collection.html>`_, `VLA Workflow <tutorials/vla_workflow.html>`_, and `VLA Inference <tutorials/vla_inference.html>`_.
- **[2026-04-14]** `Live web demo <https://nvlabs.github.io/GEAR-SONIC/demo.html>`_ — try SONIC interactively in your browser. Features `Kimodo <https://github.com/nv-tlabs/kimodo>`_ text-to-motion generation.
- **[2026-04-10]** Released **SONIC training code and checkpoint** on `HuggingFace <https://huggingface.co/nvidia/GEAR-SONIC>`_. Train from scratch or finetune. **Additional embodiment support** and **VLA data collection pipeline**. See `Training Guide <user_guide/training.html>`_.
- **[2026-03-24]** C++ inference stack update: motor error monitoring, TTS alerts, ZMQ protocol v4, idle-mode readaptation. **ZMQ header size changed to 1280 bytes.**
- **[2026-03-16]** `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_ open-sourced — 142K+ human motions (~288 hours) with G1 MuJoCo trajectories.
- **[2026-02-19]** Released GEAR-SONIC: pretrained checkpoints, C++ inference, VR teleoperation, and documentation.
- **[2025-11-12]** Initial release with Decoupled WBC for GR00T N1.5 and N1.6.

Upstream demonstrations
-----------------------

The original G1 demonstrations belong to the upstream project and are not
evidence of A3 performance. Videos and GIFs are deliberately not bundled.

See the `upstream SONIC project and demonstrations <https://nvlabs.github.io/GEAR-SONIC/>`_
and `upstream source <https://github.com/NVlabs/GR00T-WholeBodyControl>`_.
For this derivative, use the A3 entry points above.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   model_card
   getting_started/installation_deploy
   getting_started/download_models
   getting_started/quickstart
   getting_started/vr_teleop_setup

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/keyboard
   tutorials/gamepad
   tutorials/zmq
   tutorials/manager
   tutorials/isaac_teleop_publisher_setup
   tutorials/vr_wholebody_teleop
   tutorials/data_collection
   tutorials/vla_workflow
   tutorials/vla_inference

.. toctree::
   :maxdepth: 2
   :caption: Training

   getting_started/installation_training
   user_guide/training
   user_guide/training_data
   user_guide/new_embodiments

.. toctree::
   :maxdepth: 2
   :caption: Best Practices

   user_guide/teleoperation
   user_guide/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

..    api/index
..    api/teleop

.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation

   references/index
   user_guide/configuration
   references/conventions
   references/training_code
   references/deployment_code
   references/observation_config
   references/motion_reference
   references/planner_onnx
   references/jetpack6
   references/decoupled_wbc
   

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   resources/citations
   resources/license
   resources/support
..    resources/contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

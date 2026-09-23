# Reasoning Hydra for HFLEX-EQA

[![License](https://img.shields.io/badge/License-BSD-blue.svg)](LICENSE)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-22314E.svg)](https://docs.ros.org/en/jazzy/)

This repository contains the scene graph and mapping library used by [HFLEX-EQA](https://arxiv.org/abs/2609.26360), an embodied question answering system for previously unseen indoor environments. It extends [Hydra](https://github.com/MIT-SPARK/Hydra) and the [relationship-aware hierarchical scene graph](https://github.com/ntnu-arl/reasoning_hydra) with the map, navigation, frontier, and visual-memory layers needed for exploration. The high-level EQA planner and ROS launch composition live in [hvlm_planner](https://github.com/ntnu-arl/hvlm_planner/tree/hflex_eqa) and [hvlm_planner_ros](https://github.com/ntnu-arl/hvlm_planner_ros/tree/hflex_eqa).

## Table of Contents

- [EQA Scene Graph](#eqa-scene-graph)
- [Setup](#setup)
- [Usage](#usage)
  - [Habitat Simulation](#habitat-simulation)
  - [Robot Deployment](#robot-deployment)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## EQA Scene Graph

From posed RGB-D observations and semantic features, this fork incrementally builds a graph of labeled rooms, objects, navigation nodes, semantic frontiers, agent views, and a metric-semantic mesh. The EQA-specific parts are:

- **Open-vocabulary occupancy map:** depth and pixel-level language features are fused into a TSDF, then projected into a 2D occupancy grid. Cells retain free/occupied/unknown state and averaged semantic features. See [`OccupancyIntegrator`](src/reconstruction/occupancy_integrator.cpp) and the [`occupancy` settings](config/datasets/habitat.yaml).
- **Traversability graph:** a 2D generalized Voronoi diagram (GVD) extracted from the occupancy map produces the `TRAVERSABILITY` navigation layer used to connect rooms and exploration goals. See [`GvdPlace2DExtractor`](src/frontend/gvd_place_2d_extractor.cpp).
- **Semantic frontiers:** wavefront detection finds boundaries between observed free space and unexplored space. Frontier nodes carry position, direction, size, and nearby occupancy-map features, which the planner uses to score where to explore next. See [`OccupancyFrontierExtractor`](src/frontend/occupancy_frontier_extractor.cpp).
- **Key views:** selected agent nodes retain an RGB image and image embedding when pose and visual novelty thresholds are met. These views become room-associated visual memory for EQA. See [`UpdateAgentsFunctor`](src/backend/update_agents_functor.cpp) and its `enable_agent_keyframes` settings.

The [Habitat config](config/datasets/habitat.yaml) and [ANYmal config](config/datasets/anymal.yaml) enable these layers with different camera-height, occupancy, and GVD parameters. The optional topological floorplan is handled by the EQA planner; it is separate from the online scene graph.

## Setup

Use Ubuntu 24.04, ROS 2 Jazzy, an NVIDIA GPU, and the Docker workflow in the [HFLEX-EQA installation guide](https://github.com/ntnu-arl/hvlm_planner/tree/hflex_eqa#installation-with-docker). Import its [desktop](https://github.com/ntnu-arl/hvlm_planner/blob/hflex_eqa/install/default.repos) or [Jetson Thor](https://github.com/ntnu-arl/hvlm_planner/blob/hflex_eqa/install/thor.repos) dependency manifest; both select this repository's `hflex_eqa` branch. Build the workspace inside the container:

```bash
cd /developer/hflex_eqa_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --continue-on-error
source install/setup.bash
```

This is a C++ library; launch the EQA system through its ROS packages. The associated ROS interface is in [`hydra_ros`](https://github.com/ntnu-arl/hydra_ros_private/tree/hflex_eqa).

## Usage

### Habitat Simulation

Follow the [dataset and floorplan setup](https://github.com/ntnu-arl/hvlm_planner/tree/hflex_eqa#download-hm3d-and-eqa-benchmarks), then run one scene with [`habitat_eqa.launch.yaml`](https://github.com/ntnu-arl/hvlm_planner_ros/blob/hflex_eqa/hvlm_planner_ros/launch/habitat/habitat_eqa.launch.yaml):

```bash
ros2 launch hvlm_planner_ros habitat_eqa.launch.yaml \
  scene_file:=/developer/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb \
  question:="What color is the microwave"
```

The ROS launch selects [`config/datasets/habitat.yaml`](config/datasets/habitat.yaml). For OpenEQA or ExploreEQA dataset loops, use `ros2 launch simulation_manager_ros simulate.launch.yaml` as described in the [main guide](https://github.com/ntnu-arl/hvlm_planner/tree/hflex_eqa#run-openeqa-or-exploreeqa).

### Robot Deployment

On a Jetson Thor mounted on ANYmal, use the [Thor Docker instructions](https://github.com/ntnu-arl/hvlm_planner/tree/hflex_eqa#deploy-on-jetson-thor-and-anymal). The robot must supply the camera topics and transforms configured by [`scene_graph.launch.yaml`](https://github.com/ntnu-arl/hvlm_planner_ros/blob/hflex_eqa/hvlm_planner_ros/launch/scene_graph.launch.yaml). After building and sourcing the workspace, launch the mapping and EQA stacks:

```bash
ros2 launch hvlm_planner_ros scene_graph.launch.yaml
ros2 launch hvlm_planner_ros eqa.launch.yaml \
  floorplan_json_path:=/path/to/building_floorplan.json
```

The robot scene graph launch selects [`config/datasets/anymal.yaml`](config/datasets/anymal.yaml). Adjust its occupancy-height and GVD parameters for a different sensor mount.

## Citation

Please cite [HFLEX-EQA](https://arxiv.org/abs/2609.26360) for this integration and the [relationship-aware scene graph work](https://github.com/ntnu-arl/reasoning_hydra) on which our scene graph is based. Hydra's [original paper](http://www.roboticsproceedings.org/rss18/p050.pdf) should also be cited when using its core mapping system.

```bibtex
@article{puigjaner2026hflex-eqa,
  title={Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering},
  author={Gassol Puigjaner, Albert and Alexis, Kostas},
  journal={arXiv},
  year={2026}
}

@inproceedings{puigjaner2026reasoninggraph,
  title={Relationship-Aware Hierarchical 3D Scene Graph},
  author={Gassol Puigjaner, Albert and Zacharia, Angelos and Alexis, Kostas},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}

@inproceedings{hughes2022hydra,
  title={Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization},
  author={Hughes, Nathan and Chang, Yun and Carlone, Luca},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2022}
}
```

## License

See the repository [LICENSE](LICENSE) and preserve the license notices in individual source files.

## Acknowledgements

This work was supported by the European Commission through Project SYNERGISE (Horizon Europe Grant Agreement No. 101121321) and by the Research Council of Norway through Grant NCEI (No. 357451).

## Contact

For questions about the EQA integration, use [GitHub Issues](https://github.com/ntnu-arl/hvlm_planner/issues) or contact [Albert Gassol Puigjaner](mailto:albert.g.puigjaner@ntnu.no) and [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no).

# Checkpoint Usage

Current deployment checkpoint is stored outside this ROS2 repository:

/home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt

All ROS2 simulation and deployment commands should use this ROS2 parameter:

-p checkpoint_path:=/home/dk/drl_repos/DRL-path-finding/deploy_checkpoints/A_full_method_last.pt

Do not commit .pt, .pth, or .onnx files into this ROS2 repository.

The DRL training repository is managed separately at:

/home/dk/drl_repos/DRL-path-finding

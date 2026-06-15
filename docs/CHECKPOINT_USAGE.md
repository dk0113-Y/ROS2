# Checkpoint Usage

Current deployment checkpoint is stored outside this ROS2 repository. Use an
environment variable or ROS2 parameter to point to the file on your machine:

```bash
export DRL_REPO="$HOME/drl_repos/DRL-path-finding"
export CHECKPOINT="$DRL_REPO/deploy_checkpoints/A_full_method_last.pt"
```

All ROS2 simulation and deployment commands should use this ROS2 parameter:

```bash
-p checkpoint_path:="$CHECKPOINT"
```

Do not commit .pt, .pth, or .onnx files into this ROS2 repository.

The DRL training repository is managed separately at:

```bash
$DRL_REPO
```

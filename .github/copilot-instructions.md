# GeoRT Copilot Instructions

## Project Overview
GeoRT (Geometric Retargeting) is a neural hand retargeting system that maps human hand motion capture data to robot hand joint positions. The project uses PyTorch for neural network training and SAPIEN for physics simulation.

## Architecture
```
geort/
├── model.py          # FKModel (forward kinematics), IKModel (inverse kinematics)
├── trainer.py        # GeoRTTrainer - main training loop with loss functions
├── export.py         # GeoRTRetargetingModel - deployment wrapper, load_model() API
├── formatter.py      # HandFormatter - joint position normalization [-1, 1]
├── dataset.py        # RobotKinematicsDataset, MultiPointDataset
├── loss.py           # chamfer_distance for point cloud matching
├── config/           # Hand configuration JSON files (allegro_right.json, template.py)
├── env/hand.py       # HandKinematicModel - SAPIEN-based kinematic simulation
├── mocap/            # Motion capture integrations (MediaPipe, Manus gloves)
└── utils/            # Path helpers, config loading utilities
```

## Key Workflows

### Training a Model (~1-2 min)
```bash
python ./geort/trainer.py -hand allegro_right -human_data human_alex -ckpt_tag geort_1
```
First-time training for a new hand generates an FK model checkpoint (~5 min extra).

### Visualizing a Robot Hand
```bash
python geort/env/hand.py --hand allegro_right
```

### Loading a Trained Model (Deployment)
```python
import geort
model = geort.load_model('geort_1', epoch=50)  # epoch=-1 for last checkpoint
qpos = model.forward(mocap_keypoints)  # [N, 3] -> joint angles
```

## Configuration Pattern
Robot hands require a JSON config in `geort/config/` with these fields:
- `name`: Unique identifier
- `urdf_path`: Path to URDF in `assets/`
- `base_link`: Base link name from URDF
- `joint_order`: List defining output joint order (matches robot API format)
- `fingertip_link`: Array mapping robot fingertips to human hand keypoints via `human_hand_id`

See [geort/config/template.py](geort/config/template.py) for full documentation.

## Coordinate Frame Convention
- **+Y**: Palm center → thumb
- **+Z**: Palm center → middle fingertip  
- **+X**: Palm normal (pointing out)

Human mocap and robot URDF must share this orientation. Origin alignment is not required.

## Data Flow
1. **Human Data**: `[N, 21, 3]` NumPy arrays (MediaPipe keypoint format) saved via `geort.save_human_data()`
2. **Training**: Human keypoints → IKModel → normalized joints → FKModel → predicted fingertips → loss
3. **Inference**: `model.forward(keypoints)` extracts relevant `human_hand_id` points → returns raw joint angles

## Loss Functions in Training
- **Chamfer loss** (`w_chamfer=80.0`): Match fingertip point clouds
- **Pinch loss** (`w_pinch=1.0`): Preserve fingertip proximity during pinches
- **Curvature loss** (`w_curvature=0.1`): Ensure smooth FK/IK mapping
- **Direction loss**: Maintain motion direction consistency

## Common Issues & Solutions
- **Segmentation fault on new hands**: Simplify collision meshes in URDF or remove `<collision>` elements
- **MediaPipe unreliable**: Use glove-based mocap (Manus) for deployment; MediaPipe is demo-only
- **Out-of-distribution inputs**: Keep robot joint ranges realistic to human hand motion

## File Naming Conventions
- Checkpoints: `checkpoint/{hand_name}_{timestamp}_{tag}/epoch_{N}.pth`
- Robot kinematics data: `data/{hand_name}.npz`
- Human data: `data/{name}.npy`

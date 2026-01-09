import os
from pathlib import Path

from offline_trainer.registry import DATASET_BUILDER_REGISTRY
from typing import Any



@DATASET_BUILDER_REGISTRY.register('lerobot_dataset_factory')
class LerobotDatasetFactory:
    def build(self, opt_params: dict[str, Any] | None, params) -> dict[str, Any]:
        root = Path(params['root']).expanduser().resolve()
        repo_id = params['repo_id']
        local_files_only = bool(params.get("local_files_only", False))

        # --- 1. Extract Horizon/History Parameters ---
        # Default to 30 FPS if not provided, though ideally this matches the dataset
        HZ = float(params.get('HZ', 20))

        # Action Chunk (e.g., 50 means return current action + 49 future actions)
        action_horizon = int(params.get('action_horizon', 1))
        # Observation History (e.g., 2 means return current frame + 1 past frame)
        obs_proprio_history = int(params.get('obs_proprio_history', 1))
        obs_images_history = int(params.get('obs_images_history', 1))


        # --- 2. Calculate delta_timestamps ---
        # Calculate the time step in seconds
        dt = 1.0 / HZ

        # Actions: from 0 (current) up to (horizon-1) steps in the future
        action_timestamps = [i * dt for i in range(action_horizon)]
        
        # Observations: from -(history-1) steps in the past up to 0 (current)
        obs_proprio_timestamps = [i * dt for i in range(1 - obs_proprio_history, 1)]
        obs_images_timestamps = [i * dt for i in range(1 - obs_images_history, 1)]

        # Construct the config dictionary
        # You may need to adjust keys like 'observation.state' depending on your specific dataset columns
        delta_timestamps = {
            "action": action_timestamps,
            "observation.current": obs_proprio_timestamps,
            "observation.state": obs_proprio_timestamps,
            "observation.images.cam_right": obs_images_timestamps,
            "observation.images.cam_head": obs_images_timestamps,
            "observation.images.cam_left": obs_images_timestamps,
        }

        dataset = None
        stats = None

        os.environ["HF_HUB_OFFLINE"] = "1" if local_files_only else "0"
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset # newer version
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset # older version
        if local_files_only:
            if not root.exists():
                raise FileNotFoundError(f"LeRobot root directory does not exist: {root}")
            try:
                os.environ['HF_LEROBOT_HOME'] = str(root)  # must be set before dataset creation
            except:
                raise Exception("root directory for Lerobot dataset does NOT exist")
            dataset = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps) 
        else:
            dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)

        stats = dataset.meta.stats
        
        return {
            'dataset': dataset,
            'norm_stats': stats
        }
        
        
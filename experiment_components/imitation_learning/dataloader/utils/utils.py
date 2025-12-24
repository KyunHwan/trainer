import torch
import numpy as np
import os
import h5py
import fnmatch
from torch.utils.data import DataLoader

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def validate_hdf5_file(file_path, config_loader=None):
    """Validate that an HDF5 file has the required structure and data."""
    try:
        with h5py.File(file_path, "r") as root:
            if config_loader is not None:
                # Use config-based validation
                required_groups = config_loader.get_required_groups()
            else:
                # Fallback to hardcoded validation
                required_groups = [
                    "/action/joint_pos/left",
                    "/action/joint_pos/right", 
                    "/action/finger_pos/left",
                    "/action/finger_pos/right",
                    "/observation/xpos/left",
                    "/observation/xpos/right",
                    "/observation/quaternion/left",
                    "/observation/quaternion/right",
                    "/observation/joint_pos/left",
                    "/observation/joint_pos/right",
                    "/observation/joint_cur/left",
                    "/observation/joint_cur/right",
                    "/observation/hand_joint_pos/left",
                    "/observation/hand_joint_pos/right",
                    "/observation/hand_joint_cur/left",
                    "/observation/hand_joint_cur/right"
                ]
            
            for group_path in required_groups:
                if group_path not in root:
                    return False, f"Missing required group: {group_path}"
                
                if root[group_path].shape[0] == 0:
                    return False, f"Empty data in group: {group_path}"
            
            return True, "File is valid"
            
    except Exception as e:
        return False, f"Error opening file: {str(e)}"



def get_episode_len(dataset_path_list):
    all_episode_len = []
    valid_dataset_paths = []
    
    print("Validating HDF5 files...")
    for i, dataset_path in enumerate(dataset_path_list):
        try:
            with h5py.File(dataset_path, "r") as root:
                episode_len = root["/observation/joint_pos/left"].shape[0]
                all_episode_len.append(episode_len)
                valid_dataset_paths.append(dataset_path)
                if (i + 1) % 10 == 0:
                    print(f"Validated {i + 1}/{len(dataset_path_list)} files...")
        except Exception as e:
            print(f"Error loading {dataset_path} in get_episode_len: {e}")
            print(f"Skipping corrupted file: {dataset_path}")
            continue
    
    if len(valid_dataset_paths) == 0:
        raise RuntimeError("No valid HDF5 files found in dataset!")
    
    print(f"Found {len(valid_dataset_paths)} valid files out of {len(dataset_path_list)} total files")
    
    dataset_path_list.clear()
    dataset_path_list.extend(valid_dataset_paths)
    
    return all_episode_len
    


def compute_norm_stats(dataset, batch_size=128, max_samples = 100000):
    """
    Computes normalization statistics for robot state and action tensors from the dataset.
    
    Args:
        dataset: a dataset instance with __getitem__ implemented
        batch_size: batch size for efficient loading

    Returns:
        norm_stats: dictionary containing mean, std, min, max for 'action' and 'state'
    """

    # Reduce num_workers to avoid "too many open files" error with large datasets
    num_workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_actions = []
    all_states = []
    total_seen = 0 

    for i, batch in enumerate(loader):
        try:
            robot_state_data, action_data, _ = batch
            # Flatten temporal dimension if needed
            # print(f'action_data.shape: {action_data.shape}')
            B, T, D = action_data.shape
            all_actions.append(action_data)
            # print(f'robot_state_data.shape: {robot_state_data.shape}')
            B, T, D = robot_state_data.shape
            all_states.append(robot_state_data)

            total_seen += B
            if total_seen >= max_samples:
                break
        except Exception as e:
            print(f"[ERROR IN BATCH {i}]: {e}")
            break

    all_actions = torch.cat(all_actions, dim=0)
    all_states = torch.cat(all_states, dim=0)

    all_episode_len = len(all_actions)
    norm_stats = {
        "action_mean": all_actions.mean(dim=0),
        "action_std": all_actions.std(dim=0) + 1e-2,  # avoid divide by zero

        "state_mean": all_states.mean(dim=0),
        "state_std": all_states.std(dim=0) + 1e-2,
    }

    return norm_stats, all_episode_len



def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result



def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

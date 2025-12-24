#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import h5py
import cv2
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from .utils.utils import *

# Set multiprocessing sharing strategy to avoid file descriptor issues
# torch.multiprocessing.set_sharing_strategy('file_system')

import IPython

e = IPython.embed

class EpisodicDataset(Dataset):
    def __init__(
        self,
        dataset_path_list,
        camera_names,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        no_image_mode=False,
        config_loader=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.robot_obs_size = robot_obs_size
        self.img_obs_size = img_obs_size
        self.config_loader = config_loader
        self.cumulative_len = np.cumsum(self.episode_len)
        if len(self.cumulative_len) == 0:
            raise ValueError("Dataset is empty. Please check your data directory.")
        print("dataset size: ", self.cumulative_len[-1])
        self.max_episode_len = max(episode_len)

        self.augment_images = True
        self.separate_left_right = False
        self.img_downsample = True
        self.img_downsample_size = (640, 240)
        
        self.img_debug = False

        self.no_image_mode = no_image_mode

        self.relative_action_mode = True
        self.relative_obs_mode = True
        self.relative_inter_gripper_proprio = False

        self.compressed = True
        self.obs_tracker_delay = 0 # tick
        self.obs_gripper_delay = 0 # tick
        self.action_pose_delay = 0 #tick
        self.action_gripper_delay = 0 #tick
        
    def __len__(self):
        return self.cumulative_len[-1]

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        
        # Try to load the requested episode, with fallback to other episodes if corrupted
        max_retries = 3
        for retry in range(max_retries):
            try:
                with h5py.File(dataset_path, "r") as root:
                    # Load joint and finger positions
                    action_joint_left = root["/action/joint_pos/left"][()]
                    action_joint_right = root["/action/joint_pos/right"][()]
                    action_finger_left = root["/action/finger_pos/left"][()]
                    action_finger_right = root["/action/finger_pos/right"][()]
                    
                    data_len = action_joint_left.shape[0]
                    
                    # Load observations using config if available, otherwise use hardcoded method
                    if self.config_loader is not None:
                        # Use config-based loading
                        obs_sampling = np.clip(
                            range(
                                start_ts - 1 - self.obs_tracker_delay,
                                start_ts - 1 - self.obs_tracker_delay - self.robot_obs_size,
                                -1,
                            ),
                            0,
                            self.max_episode_len,
                        )
                        observation_data = self.config_loader.get_observation_data_from_hdf5(root, obs_sampling)
                        all_observations = self.config_loader.combine_observation_data(observation_data)
                    else:
                        # Fallback to hardcoded method
                        observation_xpos_dict = dict()
                        observation_quat_dict = dict()
                        observation_rotation_dict = dict()
                        observation_rotm_dict = dict()
                        
                        # Load end-effector positions and orientations
                        observation_xpos_dict['left'] = np.array(root["/observation/xpos/left"][()])
                        observation_xpos_dict['right'] = np.array(root["/observation/xpos/right"][()])
                        observation_quat_dict['left'] = np.array(root["/observation/quaternion/left"][()])
                        observation_quat_dict['right'] = np.array(root["/observation/quaternion/right"][()])
                        
                        # Convert quaternions to rotation matrices
                        observation_rotation_dict['left'] = Rotation.from_quat(observation_quat_dict['left'])
                        observation_rotation_dict['right'] = Rotation.from_quat(observation_quat_dict['right'])
                        observation_rotm_dict['left'] = observation_rotation_dict['left'].as_matrix()
                        observation_rotm_dict['right'] = observation_rotation_dict['right'].as_matrix()
                        
                        # Load joint positions
                        observation_joint_pos_left = root["/observation/joint_pos/left"][()]
                        observation_joint_pos_right = root["/observation/joint_pos/right"][()]
                        observation_joint_cur_left = root["/observation/joint_cur/left"][()]
                        observation_joint_cur_right = root["/observation/joint_cur/right"][()]
                        
                        # Load hand joint positions
                        observation_hand_joint_pos_left = root["/observation/hand_joint_pos/left"][()]
                        observation_hand_joint_pos_right = root["/observation/hand_joint_pos/right"][()]
                        observation_hand_joint_cur_left = root["/observation/hand_joint_cur/left"][()]
                        observation_hand_joint_cur_right = root["/observation/hand_joint_cur/right"][()]
                        
                        # Combine all observations
                        all_observations = np.concatenate([
                            observation_xpos_dict['left'],
                            observation_xpos_dict['right'],
                            observation_quat_dict['left'],
                            observation_quat_dict['right'],
                            observation_joint_pos_left,
                            observation_joint_pos_right,
                            observation_joint_cur_left,
                            observation_joint_cur_right,
                            observation_hand_joint_pos_left,    
                            observation_hand_joint_pos_right,
                            observation_hand_joint_cur_left,
                            observation_hand_joint_cur_right,
                        ], axis=-1)
                    
                    # Set the robot_state_sampling array
                    obs_sampling = np.clip(
                        range(
                            start_ts - 1 - self.obs_tracker_delay,
                            start_ts - 1 - self.obs_tracker_delay - self.robot_obs_size,
                            -1,
                        ),
                        0,
                        self.max_episode_len,
                    )
                    
                    if self.config_loader is not None:
                        # Use config-based observation data (already loaded above)
                        robot_state_np = all_observations
                    else:
                        # Use hardcoded method
                        robot_state_np = all_observations[obs_sampling]
                    
                    # Set the action sampling array
                    action_sampling = np.clip(
                        range(
                            start_ts + 1 + self.action_pose_delay,
                            start_ts + 1 + self.action_pose_delay + self.chunk_size,
                            1,
                        ),
                        0,
                        data_len - 1,
                    )
                    
                    # Load actions using config if available, otherwise use hardcoded method
                    if self.config_loader is not None:
                        # Use config-based loading
                        action_data = self.config_loader.get_action_data_from_hdf5(root, action_sampling)
                        action_np = self.config_loader.combine_action_data(action_data)
                    else:
                        # Fallback to hardcoded method
                        action_np = np.concatenate([
                            action_joint_left[action_sampling],
                            action_joint_right[action_sampling],
                            action_finger_left[action_sampling],
                            action_finger_right[action_sampling],
                        ], axis=-1)
                    
                    # Check if we need to pad the action data
                    if len(action_sampling) < self.chunk_size:
                        pad_len = self.chunk_size - len(action_sampling)
                        pad_action = np.zeros((pad_len, action_np.shape[1]))
                        action_np = np.concatenate([action_np, pad_action], axis=0)
                        is_pad = np.concatenate([
                            np.zeros(len(action_sampling)),
                            np.ones(pad_len)
                        ])
                    else:
                        is_pad = np.zeros(len(action_sampling))
                    
                    # Load image data if not in no_image_mode
                    if not self.no_image_mode:
                        image_dict = {}
                        img_sampling = np.clip(
                            range(
                                start_ts - 1 - self.obs_tracker_delay,
                                start_ts - 1 - self.obs_tracker_delay - self.img_obs_size,
                                -1,
                            ),
                            0,
                            self.max_episode_len,
                        )
                        
                        for cam_name in self.camera_names:
                            image_dict[cam_name] = np.array(
                                root[f"/observation/images/{cam_name}"]
                            )[img_sampling]
                        
                        if self.compressed:
                            for cam_name in image_dict.keys():
                                decompressed_image_array = []
                                for t in range(len(image_dict[cam_name])):
                                    decompressed_image = cv2.imdecode(
                                        image_dict[cam_name][t], 1
                                    )

                                    if decompressed_image.shape[0:2] != self.img_downsample_size:
                                        decompressed_image = cv2.resize(decompressed_image, self.img_downsample_size, interpolation=cv2.INTER_AREA)
                                    decompressed_image_array.append(decompressed_image)
                                image_dict[cam_name] = np.array(decompressed_image_array)
                        
                        all_cam_images = []
                        for cam_name in self.camera_names:
                            all_cam_images.append(image_dict[cam_name])
                        all_cam_images = np.stack(all_cam_images, axis=0)

                    
                    # Convert to torch tensors
                    action_data = torch.from_numpy(np.array(action_np)).float()
                    robot_state_data = torch.from_numpy(np.array(robot_state_np)).float()
                    is_pad = torch.from_numpy(is_pad).bool()
                    
                    if not self.no_image_mode:
                        image_data = torch.from_numpy(np.array(all_cam_images))
                        image_data = torch.einsum("k t h w c -> k t c h w", image_data)
                    
                    # Normalize robot state and action data
                    if self.norm_stats["action_mean"] is not None and self.norm_stats["action_std"] is not None:
                        action_data = (
                            action_data - self.norm_stats["action_mean"]
                        ) / self.norm_stats["action_std"]
                    
                    if self.norm_stats["state_mean"] is not None and self.norm_stats["state_std"] is not None:
                        robot_state_data = (
                            robot_state_data - self.norm_stats["state_mean"]
                        ) / self.norm_stats["state_std"]
                    
                    # If we get here, the file loaded successfully
                    break
                    
            except Exception as e:
                print(f"Error loading {dataset_path} in __getitem__ (attempt {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    # Try a different episode as fallback
                    fallback_index = (index + retry + 1) % len(self)
                    episode_id, start_ts = self._locate_transition(fallback_index)
                    dataset_path = self.dataset_path_list[episode_id]
                    print(f"Trying fallback episode: {dataset_path}")
                else:
                    # All retries failed, raise the error
                    raise RuntimeError(f"Failed to load any episode after {max_retries} attempts")
            
        if not self.no_image_mode:
            return image_data, robot_state_data, action_data, is_pad
        else:
            return robot_state_data, action_data, is_pad












def load_data(
    dataset_dir_l,
    camera_names,
    chunk_size=40,
    robot_obs_size=40,
    img_obs_size=1,
    skip_mirrored_data=False,
    config_loader=None,
):
    print(f'Finding all hdf5 files in {dataset_dir_l}')
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [
        find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l
    ]

    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)

    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]

    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # if episode num is -1 use every episode. if not, use sepcified number of episodes
    num_episodes_l = [len(lst) for lst in dataset_path_list_list]

    num_episodes_cumsum = np.cumsum(num_episodes_l)
    num_episodes_0 = num_episodes_l[0]     

    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[: int(num_episodes_0)]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num_episodes) + num_episodes_cumsum[idx]
        for idx, num_episodes in enumerate(num_episodes_l[1:])
    ]
    train_episode_ids = np.concatenate(train_episode_ids_l)

    all_episode_len = get_episode_len(dataset_path_list)
    
    train_episode_len_l = [
        [all_episode_len[i] for i in train_episode_ids]
        for train_episode_ids in train_episode_ids_l
    ]
    
    train_episode_len = flatten_list(train_episode_len_l)

    norm_stats = {
        "action_mean": None,
        "action_std": None,  # avoid divide by zero
        "state_mean": None,
        "state_std": None,
    }
    
    dataset_wo_norm_stats = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        no_image_mode= True,
        config_loader=config_loader
    )

    norm_stats, _ = compute_norm_stats(dataset_wo_norm_stats)

    dataset = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        config_loader=config_loader
    )

    return dataset, norm_stats

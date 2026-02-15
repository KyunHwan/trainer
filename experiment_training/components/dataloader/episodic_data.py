#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import torch
import einops

import os

# This is essential to allow multiple workers to access hdf5 files
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py

import random
from torchvision import transforms
from torch.utils.data import Dataset
import torch.distributed as dist
from scipy.spatial.transform import Rotation
from .utils.utils import *
from typing import Any
from .utils.config_loader import ConfigLoader

from pathlib import Path
import pickle
from trainer.trainer.registry import DATASET_BUILDER_REGISTRY

from typing import Any

import IPython

e = IPython.embed

@DATASET_BUILDER_REGISTRY.register('episodic_dataset_factory')
class EpisodicDatasetFactory:
    def build(self, opt_params: dict[str, Any] | None, params) -> dict[str, Any]:
        local_rank = opt_params['local_rank']
        dist_enabled = opt_params['dist_enabled']
        save_dir = opt_params['save_dir']
        raw_path = os.path.join(params['task_config_path'], f"{params['task_name']}.json")
        file_path = Path(raw_path).expanduser()
        config_loader = ConfigLoader(file_path)
        camera_names = config_loader.get_camera_names()
        
        dataset, norm_stats = load_data(
            local_rank = local_rank,
            dist_enabled = dist_enabled,
            save_dir=save_dir,
            dataset_dir_l=Path(params['dataset_dir_l']).expanduser(),
            camera_names=camera_names,
            chunk_size=params['chunk_size'],
            robot_obs_size=params['robot_obs_size'],
            img_obs_size=params['img_obs_size'],
            skip_mirrored_data=params['skip_mirrored_data'],
            config_loader=config_loader)
        if norm_stats is None:
            return dataset
        else:
            return {"dataset": dataset,
                    "norm_stats": norm_stats}

def load_dummy_data(
    local_rank,
    dist_enabled,
    save_dir,
    dataset_dir_l,
    camera_names,
    chunk_size=40,
    robot_obs_size=40,
    img_obs_size=1,
    skip_mirrored_data=False,
    config_loader=None,
):
    dataset = DummyEpisodicDataset()
    return dataset, None


def load_data(
    local_rank,
    dist_enabled,
    save_dir,
    dataset_dir_l,
    camera_names,
    chunk_size=40,
    robot_obs_size=40,
    img_obs_size=1,
    skip_mirrored_data=False,
    config_loader=None,
):
    if local_rank == 0: print(f'Finding all hdf5 files in {dataset_dir_l}')
    if isinstance(dataset_dir_l, (str, Path)):
        dataset_dir_l = [dataset_dir_l]
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

    all_episode_len = get_episode_len(local_rank, dataset_path_list)
    
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
    # This takes bulk of the dataset loading time...
    norm_stats = None

    try:
        stats_path = Path(os.path.join(save_dir, f"dataset_stats.pkl")).expanduser()
        if os.path.isfile(stats_path):
            with open(stats_path, 'rb') as file:
                norm_stats = pickle.load(file)
    except:
        pass

    if norm_stats is None:
        if local_rank == 0:
            print("calculating norm stats...")
            norm_stats, _ = compute_norm_stats(dataset_wo_norm_stats)
            try:
                stats_path = os.path.join(save_dir, f"dataset_stats.pkl")
                with open(stats_path, "wb") as f:
                    pickle.dump(norm_stats, f)
            except:
                stats_path = Path(os.path.join(save_dir, f"dataset_stats.pkl")).expanduser()
                with open(stats_path, "wb") as f:
                    pickle.dump(norm_stats, f)
        if dist_enabled:
            dist.barrier()

        if local_rank != 0:
            try:
                stats_path = os.path.join(save_dir, f"dataset_stats.pkl")
                with open(stats_path, 'rb') as file:
                    norm_stats = pickle.load(file)
            except:
                stats_path = Path(os.path.join(save_dir, f"dataset_stats.pkl")).expanduser()
                with open(stats_path, 'rb') as file:
                    norm_stats = pickle.load(file)
        
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

class DummyEpisodicDataset(Dataset):
    def __init__(self,):
        self.len = 10000

    def __len__(self,):
        return 10000
    
    def __getitem__(self, index):
        images = {
                'head': torch.rand((3, 321, 432)),
                'left': torch.rand((3, 321, 432)),
                'right': torch.rand((3, 321, 432)),
            }
        data_dict = {
            'images': images,
            'proprio': torch.rand((40, 62)),
            'action': torch.rand((40, 24)),
            'is_pad': torch.bernoulli(torch.full((40,), 0.5)).bool()
        }
        return data_dict

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
        
        self.max_episode_len = max(episode_len)

        self.augment_images = True
        self.separate_left_right = False
        self.img_downsample = False
        self.img_downsample_size = (640, 240) # (w x h)
        
        self.img_debug = False

        self.no_image_mode = no_image_mode

        self.relative_action_mode = True
        self.relative_obs_mode = True
        self.relative_inter_gripper_proprio = False

        self.compressed = True
        self.obs_tracker_delay = 0 # tick
        self.obs_gripper_delay = 0 # tick
        self.action_pose_delay = 0 # tick
        self.action_gripper_delay = 0 #tick

        # Define this in your __init__ method
        self.augmentations = transforms.Compose([
            # 1. Color Jitter: Randomly change brightness, contrast, saturation, and hue
            # Note: 'hue' only works for RGB images. If using grayscale, remove 'hue'.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

            # 2. Gaussian Blur: Blurs image to mimic focus issues or motion
            # kernel_size must be an odd, positive integer (e.g., 3, 5, 7)
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),

            # 3. Random Rotation: Rotates the image by a random angle within the range
            # degrees=10 means range is (-10, +10) degrees
            transforms.RandomRotation(degrees=5) 
        ])
        
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
                    action_finger_left = root["/action/hand_joint_pos/left"][()]
                    action_finger_right = root["/action/hand_joint_pos/right"][()]
                    
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

                                    if self.img_downsample and decompressed_image.shape[0:2] != self.img_downsample_size:
                                        decompressed_image = cv2.resize(decompressed_image, self.img_downsample_size, interpolation=cv2.INTER_AREA)
                                    decompressed_image_array.append(decompressed_image)
                                image_dict[cam_name] = np.array(decompressed_image_array)
                        
                        image_data_dict = {}
                        #all_cam_images = []
                        for cam_name in self.camera_names:
                            #all_cam_images.append(image_dict[cam_name])
                            img = torch.from_numpy(image_dict[cam_name])
                            single_bit = random.randint(0, 1)
                            mono_width = img.shape[2] // 2
                            img = img[:, :, single_bit * mono_width: (single_bit + 1) * mono_width, :]
                            
                            if img.shape[0] != 1 or img.shape[1] == 1:
                                raise ValueError("Img shape in EpisodicDataset is not correct!")
                            # 1. Rearrange to (C, H, W)
                            tensor_img = einops.rearrange(img.squeeze(), 'h w c -> c h w')

                            # 2. Convert to Float and Scale to [0, 1]
                            tensor_img = tensor_img.float() / 255.0

                            # 3. Apply Augmentations (Only during training!)
                            # Assuming you have a flag self.is_train or similar
                            if self.augmentations is not None:
                                tensor_img = self.augmentations(tensor_img)

                            # 4. Store
                            image_data_dict[cam_name] = tensor_img

                        #all_cam_images = np.stack(all_cam_images, axis=0)

                    
                    # Convert to torch tensors
                    action_data = torch.from_numpy(np.array(action_np)).float()
                    robot_state_data = torch.from_numpy(np.array(robot_state_np)).float()
                    is_pad = torch.from_numpy(is_pad).bool()
                    
                    # if not self.no_image_mode:
                    #     image_data = torch.from_numpy(np.array(all_cam_images))
                    #     image_data = torch.einsum("k t h w c -> k t c h w", image_data)
                        
                    
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
            data_dict = {
                'images': image_data_dict,
                'proprio': robot_state_data,
                'action': action_data,
                'is_pad': is_pad
            }
            return data_dict
        else:
            data_dict = {
                'proprio': robot_state_data,
                'action': action_data,
                'is_pad': is_pad
            }
            return data_dict
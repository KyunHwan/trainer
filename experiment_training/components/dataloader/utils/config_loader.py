import os
import sys
import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        
        # task configuration
        self.robot_id = self.config["robot_id"]
        self.task_name = self.config["task_name"]
        self.data_collection_method = self.config["data_collection_method"]
        self.data_collector = self.config["data_collector"]
        self.HZ = self.config["HZ"]
        self.compression = self.config["compression"]
        self.episode_len = self.config["episode_len"]
        self.image_resize = self.config["image_resize"]
        self.data_dict = self.config["data_dict"]
        self.topics = self.config.get("topics", {})
        self.data_dir = config_path

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)
        
    def get_config(self):
        return self.config
    
    def get_config_path(self):
        return self.config_path
    
    def get_state_dim(self):
        dim = 0
        
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/observation/"):
                        if isinstance(field_config, dict) and "slice" in field_config:
                            slice_range = field_config["slice"]
                            if len(slice_range) == 2:
                                dim += slice_range[1] - slice_range[0]
                        elif field_config == "pose.position":
                            dim += 3
                        elif field_config == "pose.orientation":
                            dim += 4
                        elif field_config == "data":
                            # Check if it's image data or simple data
                            if data_key.startswith("/observation/images/"):
                                # Image data - not included in state dim (handled separately)
                                continue
                            else:
                                # Simple data fields like barcode (1D)
                                dim += 1
        return dim
    
    def get_action_dim(self):
        dim = 0
        
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/action/"):
                        if isinstance(field_config, dict) and "slice" in field_config:
                            slice_range = field_config["slice"]
                            if len(slice_range) == 2:
                                dim += slice_range[1] - slice_range[0]
                        elif field_config == "pose.position":
                            dim += 3
                        elif field_config == "pose.orientation":
                            dim += 4
        return dim
    
    def get_detailed_dimensions(self):
        details = {"observations": {}, "actions": {}}
        
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/observation/"):
                        if isinstance(field_config, dict) and "slice" in field_config:
                            slice_range = field_config["slice"]
                            if len(slice_range) == 2:
                                dim = slice_range[1] - slice_range[0]
                                details["observations"][data_key] = dim
                        elif field_config == "pose.position":
                            details["observations"][data_key] = 3
                        elif field_config == "pose.orientation":
                            details["observations"][data_key] = 4
                        elif field_config == "data":
                            # Check if it's image data or simple data
                            if data_key.startswith("/observation/images/"):
                                # Image data - not included in state dim (handled separately)
                                continue
                            else:
                                # Simple data fields like barcode (1D)
                                details["observations"][data_key] = 1
                    
                    elif data_key.startswith("/action/"):
                        if isinstance(field_config, dict) and "slice" in field_config:
                            slice_range = field_config["slice"]
                            if len(slice_range) == 2:
                                dim = slice_range[1] - slice_range[0]
                                details["actions"][data_key] = dim
                        elif field_config == "pose.position":
                            details["actions"][data_key] = 3
                        elif field_config == "pose.orientation":
                            details["actions"][data_key] = 4
        return details
    
    def get_camera_names(self):
        camera_names = []
        for key in self.data_dict.keys():
            if key.startswith("/observation/images/"):
                camera_names.append(key.split("/")[-1])
        return camera_names
    
    def get_observation_keys(self):
        """Get all observation data keys from config (excluding images)"""
        obs_keys = []
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/observation/") and not data_key.startswith("/observation/images/"):
                        obs_keys.append(data_key)
        return obs_keys
    
    def get_observation_field_config(self, data_key):
        """Get field configuration for a specific observation data key"""
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config and data_key in topic_config["fields"]:
                return topic_config["fields"][data_key]
        return None
    
    def get_action_keys(self):
        """Get all action data keys from config"""
        action_keys = []
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/action/"):
                        action_keys.append(data_key)
        return action_keys
    
    def get_observation_data_from_hdf5(self, hdf5_file, sampling_indices):
        """Load observation data from HDF5 file based on config"""
        import h5py
        import numpy as np
        from scipy.spatial.transform import Rotation
        
        observation_data = {}
        
        for topic_name, topic_config in self.topics.items():
            if "fields" not in topic_config:
                continue
                
            for data_key, field_config in topic_config["fields"].items():
                if not data_key.startswith("/observation/"):
                    continue
                    
                try:
                    if isinstance(field_config, dict) and "slice" in field_config:
                        # Handle sliced data (joints, fingers, etc.)
                        slice_range = field_config["slice"]
                        attr = field_config.get("attr", None)
                        
                        # Convert config path to actual HDF5 path
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        
                        if attr:
                            # For JointState messages - but HDF5 doesn't have compound types
                            # Just use the direct path
                            raw_data = hdf5_file[hdf5_path][()]
                        else:
                            # For Float32MultiArray messages
                            raw_data = hdf5_file[hdf5_path][()]
                        
                        if len(slice_range) == 2:
                            # Check if slice range is valid for the data
                            if slice_range[1] <= raw_data.shape[1]:
                                sliced_data = raw_data[:, slice_range[0]:slice_range[1]]
                            else:
                                # If slice range exceeds data size, use all data
                                sliced_data = raw_data
                        else:
                            sliced_data = raw_data
                            
                        observation_data[data_key] = sliced_data[sampling_indices]
                        
                    elif field_config == "pose.position":
                        # Handle position data
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        raw_data = hdf5_file[hdf5_path][()]
                        observation_data[data_key] = raw_data[sampling_indices]
                        
                    elif field_config == "pose.orientation":
                        # Handle quaternion data
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        raw_data = hdf5_file[hdf5_path][()]
                        observation_data[data_key] = raw_data[sampling_indices]
                        
                    elif field_config == "data":
                        # Check if it's image data or simple data
                        if data_key.startswith("/observation/images/"):
                            # Image data - handled separately in EpisodicDataset
                            continue
                        else:
                            # Handle simple data fields (barcode, etc.)
                            hdf5_path = data_key[1:]  # Remove leading '/'
                            raw_data = hdf5_file[hdf5_path][()]
                            sampled_data = raw_data[sampling_indices]
                            # Ensure 2D array for concatenation
                            if sampled_data.ndim == 1:
                                sampled_data = sampled_data.reshape(-1, 1)
                            observation_data[data_key] = sampled_data
                        
                except KeyError:
                    print(f"Warning: {data_key} not found in HDF5 file")
                    continue
                    
        return observation_data
    
    def get_action_data_from_hdf5(self, hdf5_file, sampling_indices):
        """Load action data from HDF5 file based on config"""
        import h5py
        import numpy as np
        
        action_data = {}
        
        for topic_name, topic_config in self.topics.items():
            if "fields" not in topic_config:
                continue
                
            for data_key, field_config in topic_config["fields"].items():
                if not data_key.startswith("/action/"):
                    continue
                
                try:
                    if isinstance(field_config, dict) and "slice" in field_config:
                        # Handle sliced data (joints, fingers, etc.)
                        slice_range = field_config["slice"]
                        attr = field_config.get("attr", None)
                        
                        # Convert config path to actual HDF5 path
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        
                        if attr:
                            # For JointState messages - but HDF5 doesn't have compound types
                            # Just use the direct path
                            raw_data = hdf5_file[hdf5_path][()]
                        else:
                            # For Float32MultiArray messages
                            raw_data = hdf5_file[hdf5_path][()]
                        
                        if len(slice_range) == 2:
                            # Check if slice range is valid for the data
                            if slice_range[1] <= raw_data.shape[1]:
                                sliced_data = raw_data[:, slice_range[0]:slice_range[1]]
                            else:
                                # If slice range exceeds data size, use all data
                                sliced_data = raw_data
                        else:
                            sliced_data = raw_data
                            
                        action_data[data_key] = sliced_data[sampling_indices]
                        
                    elif field_config == "pose.position":
                        # Handle position data
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        raw_data = hdf5_file[hdf5_path][()]
                        action_data[data_key] = raw_data[sampling_indices]
                        
                    elif field_config == "pose.orientation":
                        # Handle quaternion data
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        raw_data = hdf5_file[hdf5_path][()]
                        action_data[data_key] = raw_data[sampling_indices]
                        
                except KeyError:
                    print(f"Warning: {data_key} not found in HDF5 file")
                    continue
                
        return action_data
    
    def combine_observation_data(self, observation_data):
        """Combine observation data into single array based on config order"""
        import numpy as np
        
        combined_data = []
        
        # Get observation keys in config order
        obs_keys = self.get_observation_keys()
        
        for data_key in obs_keys:
            if data_key in observation_data:
                combined_data.append(observation_data[data_key])
        
        if combined_data:
            return np.concatenate(combined_data, axis=-1)
        else:
            return np.array([])
    
    def combine_action_data(self, action_data):
        """Combine action data into single array based on config order"""
        import numpy as np
        
        combined_data = []
        
        # Get action keys in config order
        action_keys = self.get_action_keys()
        
        for data_key in action_keys:
            if data_key in action_data:
                combined_data.append(action_data[data_key])
        
        if combined_data:
            return np.concatenate(combined_data, axis=-1)
        else:
            return np.array([])
    
    def get_required_groups(self):
        """Get all required HDF5 groups for validation"""
        required_groups = []
        
        for topic_name, topic_config in self.topics.items():
            if "fields" in topic_config:
                for data_key, field_config in topic_config["fields"].items():
                    if data_key.startswith("/observation/") or data_key.startswith("/action/"):
                        # Convert config path to actual HDF5 path
                        hdf5_path = data_key[1:]  # Remove leading '/'
                        required_groups.append(hdf5_path)
        
        return required_groups
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "test.json")
    config_loader = ConfigLoader(config_path)
    print(f"Configuration loaded from {config_loader.get_config_path()}")    
    print(f"Task name: {config_loader.task_name}")
    print(f"Robot ID: {config_loader.robot_id}")
    print(f"Data collection method: {config_loader.data_collection_method}")
    print(f"Data collector: {config_loader.data_collector}")
    print(f"HZ: {config_loader.HZ}")
    print(f"Compression: {config_loader.compression}")
    print(f"Episode length: {config_loader.episode_len}")
    print(f"Image resize: {config_loader.image_resize}")
    print(f"Data dict: {config_loader.data_dict}")
    print(f"State dim: {config_loader.get_state_dim()}")
    print(f"Action dim: {config_loader.get_action_dim()}")
    print(f"Camera names: {config_loader.get_camera_names()}")
    details = config_loader.get_detailed_dimensions()
    print("Observations:")
    for key, dim in details["observations"].items():
        print(f"  {key}: {dim}")
        
    expected_keys = config_loader.get_observation_keys()
    print(f"Expected keys: {expected_keys}")
    
    print("Actions:")
    for key, dim in details["actions"].items():
        print(f"  {key}: {dim}")
        
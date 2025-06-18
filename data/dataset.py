import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_image
from torchvision import transforms as T
import yaml


class CombinedDataset(Dataset):
    def __init__(self, mode, transforms=None, config_path=None):
        """
        Args:
            mode: 'train', 'test', or 'val'
            transforms: Optional transforms to apply
            config_path: Path to config file. If None, loads default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Get data directory from config
        data_dir = self.config.get("data", {}).get("data_dir", "/path/to/default/data")

        self.mode = mode.lower()
        self.gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", self.mode)
        self.gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", self.mode)
        self.classes = ["normal", "paroxysmal"]
        self.transforms = transforms
        self.class_to_idx = {"label0": 0, "label1": 1}
        cnt = [0, 0]
        self.data = []

        # Use config for input shape if specified
        self.input_shape = self.config.get("data", {}).get("input_shape", [3, 128, 128])

        # Rest of your implementation remains the same
        subfolders = [
            f
            for f in os.listdir(self.gadf_dir)
            if os.path.isdir(os.path.join(self.gadf_dir, f))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(self.gadf_dir, folder)
            gasf_path = os.path.join(self.gasf_dir, folder)

            image_files = [f for f in os.listdir(gadf_path) if f.endswith(".jpeg")]

            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                if not os.path.exists(gadf_image_path):
                    raise FileNotFoundError(f"Missing GADF image: {gadf_image_path}")
                if not os.path.exists(gasf_image_path):
                    raise FileNotFoundError(f"Missing GASF image: {gasf_image_path}")
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in self.class_to_idx:
                    raise ValueError(f"Unknown label: {label_str} in file {image_file}")
                label = self.class_to_idx[label_str]
                if label == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
                self.data.append((gadf_image_path, gasf_image_path, label))

        # Compute stats safely
        self.gasf_mean, self.gasf_std = self._compute_stats(
            [s[1] for s in self.data], transforms
        )
        self.gadf_mean, self.gadf_std = self._compute_stats(
            [s[0] for s in self.data], transforms
        )

        self.pos_weight = cnt[0] / cnt[1]

    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get the path to the default config
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
            config_path = os.path.join(config_dir, "default.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return config["default_config"]

    # The rest of your methods remain the same
    def _compute_stats(self, image_paths, transform):
        means, stds = [], []
        for img_path in image_paths:
            try:
                # Replace Image.open with read_image
                img = read_image(img_path)

                # Convert to float (0-1 range)
                img = img.float() / 255.0

                if transform:
                    img_tensor = transform(img)
                else:
                    img_tensor = img  # Already a tensor, no need for ToTensor

                means.append(img_tensor.mean(dim=(1, 2)).cpu().numpy())
                stds.append(img_tensor.std(dim=(1, 2)).cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to compute stats for {img_path}: {str(e)}")
                continue

        if not means:
            print(
                "Warning: No valid images for stats computation, using default mean/std"
            )
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        return np.mean(means, axis=0), np.mean(stds, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gadf_path, gasf_path, label = self.data[idx]

        # Replace Image.open with read_image
        gadf_image = read_image(gadf_path)
        gasf_image = read_image(gasf_path)

        # Convert to float first if needed
        gadf_image = gadf_image.float() / 255.0
        gasf_image = gasf_image.float() / 255.0

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gadf_image = T.functional.normalize(
            gadf_image, mean=self.gadf_mean, std=self.gadf_std
        )
        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        modified_image = torch.cat((gadf_image, gasf_image), dim=0)
        return modified_image, label


class GASFDataset(Dataset):
    def __init__(self, mode, transforms=None, config_path=None):
        """
        Args:
            mode: 'train', 'test', or 'val'
            transforms: Optional transforms to apply
            config_path: Path to config file. If None, loads default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Get data directory from config
        data_dir = self.config.get("data", {}).get("data_dir", "/path/to/default/data")

        self.mode = mode.lower()
        self.gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", self.mode)
        self.gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", self.mode)
        self.classes = ["normal", "paroxysmal"]
        self.transforms = transforms
        self.class_to_idx = {"label0": 0, "label1": 1}
        cnt = [0, 0]
        self.data = []

        # Use config for input shape if specified
        self.input_shape = self.config.get("data", {}).get("input_shape", [3, 128, 128])

        # Rest of your implementation remains the same
        subfolders = [
            f
            for f in os.listdir(self.gadf_dir)
            if os.path.isdir(os.path.join(self.gadf_dir, f))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(self.gadf_dir, folder)
            gasf_path = os.path.join(self.gasf_dir, folder)

            image_files = [f for f in os.listdir(gadf_path) if f.endswith(".jpeg")]

            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                if not os.path.exists(gadf_image_path):
                    raise FileNotFoundError(f"Missing GADF image: {gadf_image_path}")
                if not os.path.exists(gasf_image_path):
                    raise FileNotFoundError(f"Missing GASF image: {gasf_image_path}")
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in self.class_to_idx:
                    raise ValueError(f"Unknown label: {label_str} in file {image_file}")
                label = self.class_to_idx[label_str]
                if label == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
                self.data.append((gadf_image_path, gasf_image_path, label))

        # Compute stats safely
        self.gasf_mean, self.gasf_std = self._compute_stats(
            [s[1] for s in self.data], transforms
        )
        self.gadf_mean, self.gadf_std = self._compute_stats(
            [s[0] for s in self.data], transforms
        )

        self.pos_weight = cnt[0] / cnt[1]

    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get the path to the default config
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
            config_path = os.path.join(config_dir, "default.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return config["default_config"]

    # The rest of your methods remain the same
    def _compute_stats(self, image_paths, transform):
        means, stds = [], []
        for img_path in image_paths:
            try:
                # Replace Image.open with read_image
                img = read_image(img_path)

                # Convert to float (0-1 range)
                img = img.float() / 255.0

                if transform:
                    img_tensor = transform(img)
                else:
                    img_tensor = img  # Already a tensor, no need for ToTensor

                means.append(img_tensor.mean(dim=(1, 2)).cpu().numpy())
                stds.append(img_tensor.std(dim=(1, 2)).cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to compute stats for {img_path}: {str(e)}")
                continue

        if not means:
            print(
                "Warning: No valid images for stats computation, using default mean/std"
            )
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        return np.mean(means, axis=0), np.mean(stds, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gadf_path, gasf_path, label = self.data[idx]

        # Replace Image.open with read_image
        gadf_image = read_image(gadf_path)
        gasf_image = read_image(gasf_path)

        # Convert to float first if needed
        gadf_image = gadf_image.float() / 255.0
        gasf_image = gasf_image.float() / 255.0

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        return gasf_image, label


class GADFDataset(Dataset):
    def __init__(self, mode, transforms=None, config_path=None):
        """
        Args:
            mode: 'train', 'test', or 'val'
            transforms: Optional transforms to apply
            config_path: Path to config file. If None, loads default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Get data directory from config
        data_dir = self.config.get("data", {}).get("data_dir", "/path/to/default/data")

        self.mode = mode.lower()
        self.gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", self.mode)
        self.gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", self.mode)
        self.classes = ["normal", "paroxysmal"]
        self.transforms = transforms
        self.class_to_idx = {"label0": 0, "label1": 1}
        cnt = [0, 0]
        self.data = []

        # Use config for input shape if specified
        self.input_shape = self.config.get("data", {}).get("input_shape", [3, 128, 128])

        # Rest of your implementation remains the same
        subfolders = [
            f
            for f in os.listdir(self.gadf_dir)
            if os.path.isdir(os.path.join(self.gadf_dir, f))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(self.gadf_dir, folder)
            gasf_path = os.path.join(self.gasf_dir, folder)

            image_files = [f for f in os.listdir(gadf_path) if f.endswith(".jpeg")]

            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                if not os.path.exists(gadf_image_path):
                    raise FileNotFoundError(f"Missing GADF image: {gadf_image_path}")
                if not os.path.exists(gasf_image_path):
                    raise FileNotFoundError(f"Missing GASF image: {gasf_image_path}")
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in self.class_to_idx:
                    raise ValueError(f"Unknown label: {label_str} in file {image_file}")
                label = self.class_to_idx[label_str]
                if label == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
                self.data.append((gadf_image_path, gasf_image_path, label))

        # Compute stats safely
        self.gasf_mean, self.gasf_std = self._compute_stats(
            [s[1] for s in self.data], transforms
        )
        self.gadf_mean, self.gadf_std = self._compute_stats(
            [s[0] for s in self.data], transforms
        )

        self.pos_weight = cnt[0] / cnt[1]

    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get the path to the default config
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
            config_path = os.path.join(config_dir, "default.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return config["default_config"]

    # The rest of your methods remain the same
    def _compute_stats(self, image_paths, transform):
        means, stds = [], []
        for img_path in image_paths:
            try:
                # Replace Image.open with read_image
                img = read_image(img_path)

                # Convert to float (0-1 range)
                img = img.float() / 255.0

                if transform:
                    img_tensor = transform(img)
                else:
                    img_tensor = img  # Already a tensor, no need for ToTensor

                means.append(img_tensor.mean(dim=(1, 2)).cpu().numpy())
                stds.append(img_tensor.std(dim=(1, 2)).cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to compute stats for {img_path}: {str(e)}")
                continue

        if not means:
            print(
                "Warning: No valid images for stats computation, using default mean/std"
            )
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        return np.mean(means, axis=0), np.mean(stds, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gadf_path, gasf_path, label = self.data[idx]

        # Replace Image.open with read_image
        gadf_image = read_image(gadf_path)
        gasf_image = read_image(gasf_path)

        # Convert to float first if needed
        gadf_image = gadf_image.float() / 255.0
        gasf_image = gasf_image.float() / 255.0

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gadf_image = T.functional.normalize(
            gadf_image, mean=self.gadf_mean, std=self.gadf_std
        )
        return gadf_image, label


class ComplexDataset(Dataset):
    def __init__(self, mode, transforms=None, config_path=None):
        """
        Args:
            mode: 'train', 'test', or 'val'
            transforms: Optional transforms to apply
            config_path: Path to config file. If None, loads default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Get data directory from config
        data_dir = self.config.get("data", {}).get("data_dir", "/path/to/default/data")

        self.mode = mode.lower()
        self.gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", self.mode)
        self.gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", self.mode)
        self.classes = ["normal", "paroxysmal"]
        self.transforms = transforms
        self.class_to_idx = {"label0": 0, "label1": 1}
        cnt = [0, 0]
        self.data = []

        # Use config for input shape if specified
        self.input_shape = self.config.get("data", {}).get("input_shape", [3, 128, 128])

        # Rest of your implementation remains the same
        subfolders = [
            f
            for f in os.listdir(self.gadf_dir)
            if os.path.isdir(os.path.join(self.gadf_dir, f))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(self.gadf_dir, folder)
            gasf_path = os.path.join(self.gasf_dir, folder)

            image_files = [f for f in os.listdir(gadf_path) if f.endswith(".jpeg")]

            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                if not os.path.exists(gadf_image_path):
                    raise FileNotFoundError(f"Missing GADF image: {gadf_image_path}")
                if not os.path.exists(gasf_image_path):
                    raise FileNotFoundError(f"Missing GASF image: {gasf_image_path}")
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in self.class_to_idx:
                    raise ValueError(f"Unknown label: {label_str} in file {image_file}")
                label = self.class_to_idx[label_str]
                if label == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
                self.data.append((gadf_image_path, gasf_image_path, label))

        # Compute stats safely
        self.gasf_mean, self.gasf_std = self._compute_stats(
            [s[1] for s in self.data], transforms
        )
        self.gadf_mean, self.gadf_std = self._compute_stats(
            [s[0] for s in self.data], transforms
        )

        self.pos_weight = cnt[0] / cnt[1]

    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Get the path to the default config
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
            config_path = os.path.join(config_dir, "default.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return config["default_config"]

    # The rest of your methods remain the same
    def _compute_stats(self, image_paths, transform):
        means, stds = [], []
        for img_path in image_paths:
            try:
                # Replace Image.open with read_image
                img = read_image(img_path)

                # Convert to float (0-1 range)
                img = img.float() / 255.0

                if transform:
                    img_tensor = transform(img)
                else:
                    img_tensor = img  # Already a tensor, no need for ToTensor

                means.append(img_tensor.mean(dim=(1, 2)).cpu().numpy())
                stds.append(img_tensor.std(dim=(1, 2)).cpu().numpy())
            except Exception as e:
                print(f"Warning: Failed to compute stats for {img_path}: {str(e)}")
                continue

        if not means:
            print(
                "Warning: No valid images for stats computation, using default mean/std"
            )
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

        return np.mean(means, axis=0), np.mean(stds, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        gadf_path, gasf_path, label = self.data[idx]
        gadf_image = read_image(gadf_path)
        gasf_image = read_image(gasf_path)

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gadf_image = T.functional.normalize(
            gadf_image, mean=self.gadf_mean, std=self.gadf_std
        )
        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        # --------------------GASF + i GADF ------------------------------------------
        # Create a complex tensor using GASF as real part and GADF as imaginary part
        complex_r = torch.complex(gasf_image[0], gadf_image[0])
        complex_g = torch.complex(gasf_image[1], gadf_image[1])
        complex_b = torch.complex(gasf_image[2], gadf_image[2])

        # Stack channels to form the final complex image
        complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)
        # -----------------------0 + i GADF -------------------------------------------
        # # Create a zero tensor with the same shape as gadf_image
        # zero_tensor = torch.zeros_like(gadf_image)

        # # Create a complex tensor with zero real part and GADF as imaginary part
        # complex_r = torch.complex(zero_tensor[0], gadf_image[0])
        # complex_g = torch.complex(zero_tensor[1], gadf_image[1])
        # complex_b = torch.complex(zero_tensor[2], gadf_image[2])

        # # Stack channels to form the final complex image
        # complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)
        # ------------------------GADF + i GADF --------------------------------------
        # # Create a complex tensor using GASF as real part and GADF as imaginary part
        # complex_r = torch.complex(gadf_image[0], gadf_image[0])
        # complex_g = torch.complex(gadf_image[1], gadf_image[1])
        # complex_b = torch.complex(gadf_image[2], gadf_image[2])

        # # Stack channels to form the final complex image
        # complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)
        # ------------------------GADF + i GASF --------------------------------------
        # Create a complex tensor using GADF as real part and GASF as imaginary part
        # complex_r = torch.complex(gadf_image[0], gasf_image[0])
        # complex_g = torch.complex(gadf_image[1], gasf_image
        # complex_b = torch.complex(gadf_image[2], gasf_image[2])
        # Stack channels to form the final complex image
        # complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)
        # ------------------------GASF + i GASF --------------------------------------
        # Create a complex tensor using GASF as real part and GASF as imaginary part
        # complex_r = torch.complex(gasf_image[0], gasf_image[
        # complex_g = torch.complex(gasf_image[1], gasf_image[1])
        # complex_b = torch.complex(gasf_image[2], gasf_image[
        # Stack channels to form the final complex image
        # complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)
        
        return complex_image, label

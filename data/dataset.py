import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
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
        # Existing implementation...
        means, stds = [], []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                if transform:
                    # If transform contains ToTensor(), this already gives us a tensor
                    img_tensor = transform(img)
                else:
                    # Only use separate ToTensor if no transform provided
                    to_tensor = T.ToTensor()
                    img_tensor = to_tensor(img)

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
        gadf_image = Image.open(gadf_path).convert("RGB")
        gasf_image = Image.open(gasf_path).convert("RGB")

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gadf_image = T.functional.normalize(
            gadf_image, mean=self.gadf_mean, std=self.gadf_std
        )
        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        gadf_grayscale = T.functional.rgb_to_grayscale(
            gadf_image, num_output_channels=1
        )
        gasf_grayscale = T.functional.rgb_to_grayscale(
            gasf_image, num_output_channels=1
        )
        zero_chanel = torch.zeros_like(gadf_grayscale)

        modified_image = torch.cat((gadf_grayscale, gasf_grayscale, zero_chanel), dim=0)
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
        # Existing implementation...
        means, stds = [], []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                if transform:
                    # If transform contains ToTensor(), this already gives us a tensor
                    img_tensor = transform(img)
                else:
                    # Only use separate ToTensor if no transform provided
                    to_tensor = T.ToTensor()
                    img_tensor = to_tensor(img)

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
        gadf_image = Image.open(gadf_path).convert("RGB")
        gasf_image = Image.open(gasf_path).convert("RGB")

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        # gadf_image = T.functional.normalize(
        #     gadf_image, mean=self.gadf_mean, std=self.gadf_std
        # )
        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        # gadf_grayscale = T.functional.rgb_to_grayscale(
        #     gadf_image, num_output_channels=1
        # )
        # gasf_grayscale = T.functional.rgb_to_grayscale(
        #     gasf_image, num_output_channels=1
        # )
        # zero_chanel = torch.zeros_like(gadf_grayscale)

        # modified_image = torch.cat((gadf_grayscale, gasf_grayscale, zero_chanel), dim=0)
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
        # Existing implementation...
        means, stds = [], []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                if transform:
                    # If transform contains ToTensor(), this already gives us a tensor
                    img_tensor = transform(img)
                else:
                    # Only use separate ToTensor if no transform provided
                    to_tensor = T.ToTensor()
                    img_tensor = to_tensor(img)

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
        gadf_image = Image.open(gadf_path).convert("RGB")
        gasf_image = Image.open(gasf_path).convert("RGB")

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
        # Existing implementation...
        means, stds = [], []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                if transform:
                    # If transform contains ToTensor(), this already gives us a tensor
                    img_tensor = transform(img)
                else:
                    # Only use separate ToTensor if no transform provided
                    to_tensor = T.ToTensor()
                    img_tensor = to_tensor(img)

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
        gadf_image = Image.open(gadf_path).convert("RGB")
        gasf_image = Image.open(gasf_path).convert("RGB")

        if self.transforms:
            gadf_image = self.transforms(gadf_image)
            gasf_image = self.transforms(gasf_image)

        gadf_image = T.functional.normalize(
            gadf_image, mean=self.gadf_mean, std=self.gadf_std
        )
        gasf_image = T.functional.normalize(
            gasf_image, mean=self.gasf_mean, std=self.gasf_std
        )
        # Create a complex tensor using GADF as real part and GASF as imaginary part
        complex_r = torch.complex(gadf_image[0], gasf_image[0])
        complex_g = torch.complex(gadf_image[1], gasf_image[1])
        complex_b = torch.complex(gadf_image[2], gasf_image[2])

        # Stack channels to form the final complex image
        complex_image = torch.stack([complex_r, complex_g, complex_b], dim=0)

        return complex_image, label

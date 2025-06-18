import os
import yaml
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class ComplexImagePipeline(Pipeline):
    def __init__(
        self, file_list, batch_size, num_threads, device_id, input_size=(128, 128)
    ):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.input_size = input_size
        self.file_list = file_list

    def define_graph(self):
        # Define reader in the graph
        images, labels = fn.readers.file(
            file_list=self.file_list,
            random_shuffle=True,
            name="Reader",
            pad_last_batch=True,
            read_ahead=True,
            file_root=".",  # not used with file_list
        )

        # Split paths and read GASF and GADF separately
        gasf, gadf = fn.split(images, axis=0, num_outputs=2)

        # Define and connect decoders directly in the graph
        gasf = fn.decoders.image(gasf, device="mixed", output_type=types.RGB)
        gadf = fn.decoders.image(gadf, device="mixed", output_type=types.RGB)

        # Apply preprocessing in the graph
        gasf = fn.resize(gasf, resize_x=self.input_size[1], resize_y=self.input_size[0])
        gadf = fn.resize(gadf, resize_x=self.input_size[1], resize_y=self.input_size[0])

        gasf = fn.cast(gasf, dtype=types.FLOAT)
        gadf = fn.cast(gadf, dtype=types.FLOAT)

        gasf = fn.normalize(gasf, mean=[0.5 * 255] * 3, stddev=[0.5 * 255] * 3)
        gadf = fn.normalize(gadf, mean=[0.5 * 255] * 3, stddev=[0.5 * 255] * 3)

        return (gasf, gadf), labels


def load_config(config_path=None):
    if config_path is None:
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"
        )
        config_path = os.path.join(config_dir, "default.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["default_config"]


class DALIWrapper:
    def __init__(
        self,
        file_list,
        batch_size,
        device_id=0,
        num_threads=4,
        input_size=(128, 128),
    ):
        self.pipeline = ComplexImagePipeline(
            file_list=file_list,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            input_size=input_size,
        )
        self.pipeline.build()
        self.iterator = DALIGenericIterator(
            self.pipeline,
            output_map=["gasf_gadf", "label"],
            reader_name="Reader",
            dynamic_shape=False,
            last_batch_policy="PARTIAL",
        )

    def __iter__(self):
        for batch in self.iterator:
            gasf, gadf = batch[0]["gasf_gadf"]
            label = batch[0]["label"].squeeze().long()
            gasf = gasf.permute(0, 3, 1, 2).contiguous()
            gadf = gadf.permute(0, 3, 1, 2).contiguous()
            complex_tensor = torch.complex(gasf, gadf)
            yield complex_tensor, label


def get_dali_dataloaders(config_path=None):
    config = load_config(config_path)
    batch_size = config.get("training", {}).get("batch_size", 32)
    input_size = tuple(
        config.get("data", {}).get("input_shape", [3, 128, 128])[1:]
    )  # Skip channels

    # Make sure these files exist
    train_file = os.path.join(os.path.dirname(__file__), "train_filelist.txt")
    val_file = os.path.join(os.path.dirname(__file__), "val_filelist.txt")
    test_file = os.path.join(os.path.dirname(__file__), "test_filelist.txt")

    # Check if files exist
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DALI file list not found: {file_path}")

    loaders = {
        "train": DALIWrapper(train_file, batch_size, input_size=input_size),
        "val": DALIWrapper(val_file, batch_size, input_size=input_size),
        "test": DALIWrapper(test_file, batch_size, input_size=input_size),
    }
    return loaders


if __name__ == "__main__":
    dataloaders = get_dali_dataloaders()
    for batch in dataloaders["train"]:
        x, y = batch
        print(f"Batch shape: {x.shape}, Labels: {y.shape}")
        break

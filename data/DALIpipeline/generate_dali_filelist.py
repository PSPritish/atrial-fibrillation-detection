import os


def create_dali_filelist_direct(
    data_dir, mode, output_filelist_path, class_to_idx={"label0": 0, "label1": 1}
):
    gadf_dir = os.path.join(data_dir, "GADFC_train_test_val", mode)
    gasf_dir = os.path.join(data_dir, "GASFC_train_test_val", mode)

    with open(output_filelist_path, "w") as f_out:
        subfolders = [
            d for d in os.listdir(gadf_dir) if os.path.isdir(os.path.join(gadf_dir, d))
        ]
        for folder in subfolders:
            gadf_path = os.path.join(gadf_dir, folder)
            gasf_path = os.path.join(gasf_dir, folder)
            image_files = [
                file for file in os.listdir(gadf_path) if file.endswith(".jpeg")
            ]
            for image_file in image_files:
                gadf_image_path = os.path.join(gadf_path, image_file)
                gasf_image_path = os.path.join(gasf_path, image_file)
                label_str = image_file.split("_")[-1].split(".")[0]
                if label_str not in class_to_idx:
                    continue
                label = class_to_idx[label_str]
                f_out.write(f"{gasf_image_path} {gadf_image_path} {label}\n")

    print(f"Filelist created at {output_filelist_path}")


# Example usage
data_dir = "/home/prasad/Desktop/datas2"  # Replace with your actual dataset root
mode = "val"
output_path = f"{mode}_filelist.txt"
create_dali_filelist_direct(data_dir, mode, output_path)

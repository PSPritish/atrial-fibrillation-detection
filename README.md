# Atrial Fibrillation Detection using Complex-Valued ResNet Architectures

## Overview
This project aims to detect atrial fibrillation (AF) using advanced deep learning techniques, specifically leveraging complex-valued ResNet architectures. The detection process utilizes GASF (Gramian Angular Summation Field) and GADF (Gramian Angular Difference Field) transformations to preprocess the input data.

## Project Structure
The project is organized into several key directories and files:

- **config/**: Contains configuration files for different experiments, including model architectures and training strategies.
- **data/**: Manages data processing, including raw and processed datasets.
- **experiments/**: Handles the execution and management of experiments, including tracking metrics and hyperparameters.
- **logs/**: Stores log files for monitoring training progress.
- **models/**: Contains implementations of various neural network architectures and components.
- **notebooks/**: Includes Jupyter notebooks for exploratory data analysis and results visualization.
- **saved_models/**: Directory for saving trained models.
- **scripts/**: Contains scripts for training, evaluating, and exporting models.
- **tests/**: Includes unit tests for ensuring code quality and functionality.
- **utils/**: Utility functions for various tasks, including callbacks and visualization.

## Installation
To set up the project environment, use the following command:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate <your-env-name>
```

## Usage
1. **Data Preparation**: Place your raw data in the `data/raw/` directory. The data processing scripts will handle the transformation into the required format.
2. **Configuration**: Modify the configuration files in the `config/experiments/` directory to set hyperparameters and training strategies.
3. **Training**: Run the training script:

```bash
python scripts/train.py --config config/experiments/resnet18_focal.yaml
```

4. **Evaluation**: After training, evaluate the model using:

```bash
python scripts/evaluate.py --model <model_path>
```

5. **Results Visualization**: Use the Jupyter notebooks in the `notebooks/` directory to visualize the results and metrics.

## Experiment Tracking
The project employs systematic tracking of experiments, including hyperparameters, loss functions, and training strategies. This ensures reproducibility and facilitates comparison across different model architectures.

## Contribution
Contributions to the project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
We acknowledge the contributions of the open-source community and the researchers whose work has inspired this project.
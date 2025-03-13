# Quantization-Free Autoregressive Action Transformer 
An official implementation of the paper **Quantization-Free Autoregressive Action Transformer**.

## Installation

You can install this project using either [Poetry](https://python-poetry.org/) or the `requirements.txt` file.

### Using Poetry

If you prefer to manage dependencies with Poetry, follow these steps:

1. Install Poetry if you haven't already.

2. Navigate to the project directory and install dependencies:
   ```sh
   poetry install
   ```

3. Activate the virtual environment:
   ```sh
   poetry shell
   ```

### Using `requirements.txt`

Alternatively, you can install dependencies using `pip` and `requirements.txt`:

1. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies from `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```
### Installing Kitchen and UR3 Environments

If you are using the Kitchen or UR3 environments, you need to run the setup script to properly install them:
   ```sh
   bash setup.sh
   ```
This will ensure that all necessary dependencies and configurations for these environments are correctly set up.

## Downloading the Datasets

This project uses datasets released in the paper [VQ-BeT](https://arxiv.org/abs/2403.03181). To get started:

1. Download the dataset from the official release: [VQ-BeT Dataset](https://drive.google.com/file/d/1aHb4kV0mpMvuuApBpVGYjAPs6MCNVTNb/view?usp=sharing)
2. Create a new directory named `data` inside your project:
   ```sh
   mkdir data
   ```
3. Move all the folder contents of the dataset directory into the `data` directory.

If you wish to change the dataset paths, feel free to adjust the path constants in `src/qfat/constants.py` 
Once the dataset is in place, you're ready to proceed! 🚀 

## Logging into Weights & Biases

This project uses [Weights & Biases](https://wandb.ai/) (wandb) for experiment tracking and model registry. To set up wandb, follow these steps:

1. Log in to wandb:
   ```sh
   wandb login
   ```

2. Follow the instructions to authenticate with your wandb account.

## Running Training

To start training your model, use the following command:
   ```sh
   python scripts/train.py --config-name [CONFIG_NAME]
   ```

Replace `[CONFIG_NAME]` with one of the valid configuration names found in `configs/training`:

- `kitchen`
- `kitchen_conditional`
- `pusht_image`
- `pusht`
- `ur3`
- `ur3_conditional`
- `ant`

By default, training logs and model checkpoints will be tracked using wandb. You can customize logging settings inside the `config.yaml` or by modifying the training script. You can adjust the model checkpointing frequency or the evaluation frequency using the `n_save_model` and `n_eval` respectively.

Make sure you are logged into wandb before running training to ensure proper experiment tracking! 📊🔥

## Running Inference

Once you have trained models, you can run inference using the following command:
   ```sh
   python scripts/inference.py --config-name [CONFIG_NAME]
   ```

Replace `[CONFIG_NAME]` with one of the valid configuration names found in `configs/inference`:

- `kitchen`
- `kitchen_conditional`
- `pusht_image`
- `pusht`
- `ur3`
- `ur3_conditional`
-  `ant`

To properly run inference, you must add the `training_run_id` of the training run in the corresponding configuration file under `configs/inference`. Additionally, specify the `model_afid` you wish to run inference on.

Example configuration:
```yaml
training_run_id: wandb_username/qfat/o4dz629x
model_id: wandb_username/qfat/checkpoint_epoch_720_o4dz629x:v0
```

## Callbacks

Callbacks allow you to monitor and debug your model during both training and inference. You can add them to the configuration files under `configs/training` and `configs/inference`. The following describes the most important **inference callbacks** and how to implement custom ones:

### Step End Callbacks

To visualize the output distributions over time and log per dimension as a video to wandb:
```yaml
callbacks:
  step_end:
    - _target_: qfat.callbacks.inference_callbacks.QFATOutputStatsLogger
      lim: [Add limits of the action space with a bit of buffer]
```
This helps debug and understand what the model has learned.

### Episode End Callbacks

To log the total reward at the end of each episode:
```yaml
  episode_end:
    - _target_: qfat.callbacks.inference_callbacks.RewardLogger
      reward_reduction: sum  # Options: sum, last, average
```
This allows tracking of model performance over time.

To log task completion success distribution and behavioral entropy:
```yaml
  episode_end:
    - _target_: qfat.callbacks.inference_callbacks.EnvBehaviourLogger
      min_sequence_length: # add minimum task completion sequence length you are interested in 
      max_sequence_length: # add maximum task completion sequence length you are interested in
```
This provides insight into how well different sequence lengths perform.

### Custom Callbacks

More options can be found under `src/qfat/callbacks/inference_callbacks.py` for inference and `src/qfat/callbacks/training_callbacks.py` for training. There are many additional callback options available in these files. If custom functionality is required, implement it there and add it to one of the following stages for inference:

- `step_start`
- `step_end`
- `episode_start`
- `episode_end`

and for training:

- `run_start`
- `run_end`
- `iteration_start`
- `iteration_end`
- `epoch_start`
- `epoch_end`

# The Cocktail Effect
This repo contains the necessary code to recreate the full training component of our ablation study.

## Running the experiments
First create and activate a new conda environment:
```
conda create -n myenv python=3.11.9 
conda activate myenv
```

Next install the requirements:
``` 
pip install -r requirements.txt --no-cache
```
Then run the main training script:
```
. src/train_script.sh
```
This should run the full ablation, creating all $\binom{n}{1} + \binom{n}{2} + \binom{n-1}{1} + \binom{n}{n} = 55$ variations in the cocktails folder, as well as various statistics about each training run and setup.

In order to choose a different model for experimenting, alter the model name in `src/train_script.sh` to the desired model, and update `model_name_or_path` in `training_configs/base_config.yaml`. You also need to update the `template` variable according to the selected model (see <href https://github.com/hiyouga/LLaMA-Factory#supported-models> for more details).
# Forcasting Solar Energetic Particles Fluxes Using Transformer and Comparing to Previous Methods

## Intro
Predicting Solar Energetic Particle (SEP) fluxes is critical in the protection of operating satellites.  [Posner 2007](https://doi.org/10.1029/2006SW000268) first attempted to demonstrate short term  1 hour forecasting on the Comprehensive Suprathermal and Energetic Particle Analyzer (COSTEP) on SOHO  using relativistic electrons travelling 1 AU ahead of nonrelativistc Solar Proton Events (SPE) using a $13 \times 18$ forecasting matrix that maps electron intensity and a calculated electron increase parameter. Continued efforts using advanced methods such as machine learning algorithms more accurately forecasted these storms. [Torres 2025](https://doi.org/10.1029/2024SW003921) uses Neural Networks (NN) and Recurrnet Neural Networks (RNN) to develope an early warning system using COSTEP data. This repo will attempt to reproduce Torres's M1 models with a transformer encoding only achtiecture using results from both papers as a baseline. Can a transformer achtiecture outperform these models 30 mins to 1 hour while providing accurate forecasting up to 2 hours? 

```seu_predict_torres_transformer.py``` performs a hyperparamterization search to find the most efficient model in forecasting. Torres RNN and NN uses two layers: a hidden layer with 30 units and a dense output layer of one unit. Because of the smaller sizes we will be training our transfomer on smaller parameters.

# Data Acquistion

Torres uses a the 5 min averaged data for their forecasting. you can download it  [here](https://soho.nascom.nasa.gov/data/archive.html), afterwards you want to extract it in the `SWLG_Code/code/data/enviromental/ephin-soho/5min_proton_1997-2025` folder (need to clean data folder)

# Config

```
file:
  find_best_model: True
  model_name_use: ""
model:
  n_encoder_layers: 2
  dim_values: [32]
  n_heads: 4
  dropout: 0.1
  train_new_model: False
  seed_experiment: False

training:
  learning_rates: [0.0005]  
  n_epochs: [1000]
  batch_sizes: [32]
  electron_range: 24
  num_workers: 4

data:
  forecasts_out: [6,12]    # 6 = 30min, 12 = 60min
  train_split: 0.8
```

## Configuration Reference

### file
- **find_best_model** — Whether to run model selection across the search results (trivial here since there's one candidate per horizon).
- **model_name_use** — A specific model filename to load by name; empty means none.

### model
- **n_encoder_layers** — Number of stacked transformer encoder layers, set to 2 to match Torres's small scale.
- **dim_values** — The embedding dimension each timestep is projected into before attention, kept small at 32.
- **n_heads** — Number of parallel attention heads (8 dimensions each), letting the model attend to multiple patterns at once.
- **dropout** — Fraction of neurons randomly zeroed during training to limit overfitting, set mild at 0.1.
- **train_new_model** — Whether to train fresh models or skip the search and load prior results; False skips it.
- **seed_experiment** — Whether to retrain the five seeds or evaluate already-saved ones; False evaluates existing models.

### training
- **learning_rates** — Adam's weight-update step size, set to a stable 0.0005.
- **n_epochs** — Maximum training epochs (1000 ceiling), with convergence stopping earlier when loss plateaus.
- **batch_sizes** — Number of windows per gradient update, set to 32 to match Torres-scale training.
- **electron_range** — Input window length in timesteps (24 = 2 hours), matching Torres's electron history window.
- **num_workers** — Parallel CPU processes feeding the GPU, set to 4 for faster data loading.

### data
- **forecasts_out** — Forecast horizons in timesteps: 6 (30 min) and 12 (60 min).
- **train_split** — Fraction used for training versus testing, 0.8 for a chronological 80/20 split.



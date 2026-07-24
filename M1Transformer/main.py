import os
import numpy as np
import pandas as pd
import argparse
import yaml
import torch

from datetime import datetime

from torres.load_data import load_series
from torres.m1 import pair_input_output
from torres.m1 import evaluate
from torres.time_series_classification import score_forecast
from torres.stats import tss_f1

from transformer.m1dataset import M1Dataset
from transformer.m1transformer import M1Transformer 

from transformer.m1transformersin import M1TransformerSin
from transformer.m1transfotrmerRoPE import M1TransformerRoPE


import torch as tc
from torch.utils.data import Dataset, DataLoader
from torch import nn

from linear.linear_regression import run_linear_baseline
        
def train_model(model, train_loader, optimizer, loss_fn, device,
                n_epoch=1000, patience=20, min_delta=1e-4, tag=""):
    """Train one model with Torres-style convergence on training loss."""
    n_batches = len(train_loader)
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        epoch_start_time = datetime.now()
        total_loss = torch.zeros(1, device=device)
        for batch_counter, batch in enumerate(train_loader, start=1):
            x = batch["Current_Flux_Data"].to(device)
            y = batch["Target_Proton_Data"].to(device)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()           
            total_loss += loss.detach()   # stays on GPU, no sync
        avg_loss = (total_loss / n_batches).item()   # only sync ONCE, at epoch end

        # avg_loss = total_loss / n_batches
        epoch_end_time = datetime.now()

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        print(f"{tag} Epoch {epoch+1}/{n_epoch} done in "
              f"{(epoch_end_time - epoch_start_time).total_seconds():.1f}s — "
              f"loss {avg_loss:.4f} | best {best_loss:.4f} | "
              f"patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"{tag} Converged at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss

def test_model(model, test_loader, device, loss_fn):
    model.eval()
    
    predictions = tc.tensor([]).to(device)
    # original    = tc.tensor([]).to(device)
    # losses      = tc.tensor([]).to(device)
    # test_loss   = 0.0

    with torch.no_grad():
        for batch in test_loader:
            x = batch["Current_Flux_Data"].to(device)
            # y = batch["Target_Proton_Data"].to(device)
            pred = model(x).squeeze()
            predictions = tc.cat((predictions, pred.flatten()))
            # loss = loss_fn(y,pred)
            # loss = tc.cat((losses, loss.flatten()))
            # test_loss += loss.item()

    return predictions 

def load_config(path="utils/config.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)

    cfg = {
        
        "n_encoder_layers": config["model"]["n_encoder_layers"],
        "dim_val":       config["model"]["dim_val"],
        "n_heads":          config["model"]["n_heads"],
        "dropout":          config["model"]["dropout"],
        "window_size":           config["model"]["window_size"],

        "learning_rate":   config["training"]["learning_rate"],
        "n_epochs":         config["training"]["n_epochs"],
        "batch_size":      config["training"]["batch_size"],
        "num_workers":      config["training"]["num_workers"],
        "train_new_models": config["training"]["train_new_models"],
        "n_seeds":          config["training"]["n_seeds"],

        "prediction_time":    config["data"]["prediction_time"],
        "train_split":      config["data"]["train_split"],
    }
    return cfg

def create_result_dirs(cfg):
    """Create results/linear/t={pt} and results/transformer/t={pt}/{model_name}
    (one per seed) for every prediction_time in cfg, so evaluate() has
    somewhere to write before it's called."""
    for pt in cfg["prediction_time"]:
        os.makedirs(f"results/linear/t={pt}", exist_ok=True)
        os.makedirs(f"models/t={pt}", exist_ok=True)
        for seed in range(cfg["n_seeds"]):
            model_name = "M1-" + str(seed).zfill(2)
            os.makedirs(f"results/transformer/t={pt}/{model_name}", exist_ok=True)

def create_result_csv(cfg):
    results = {}
    final_results = {}
    for pt in cfg["prediction_time"]:
        
        metrics = {"Average MAE":[],
                   "Average PE": [],
                   "Average O2P lag":[],
                   "Average O2T lag":[],
                   "Average ln10 lag":[],
                    }
        for seed in range(cfg["n_seeds"]):
            model_name = model_name = "M1-" + str(seed).zfill(2)

            with open(f"results/transformer/t={pt}/{model_name}/results.txt") as file:
                
                for line in file:
                    for m in list(metrics):
                        if line.startswith(m):
                            metrics[m].append(float(line.split("=")[1].replace(" ", "")))

        results[f"t={pt}"] = metrics
        
    results_df = pd.DataFrame([])
    

    for  pt in cfg["prediction_time"]:

        data = {}
        for m in list(metrics):
            mean = np.mean(results[f"t={pt}"][m]).round(3)
            std =  np.std(results[f"t={pt}"][m]).round(3)
            results[f"t={pt}"]["RAVG-" + m] = mean
            results[f"t={pt}"]["RSTD-" + m] = std

            data["t="] =  pt
            data[m] = mean
            data["std " + m] = std 


            

        df = pd.DataFrame(data, index = [0])
        results_df = pd.concat([results_df, df])
        print(results_df)
    results_df.to_csv("results/results.csv", index = False)
    return results       

def set_up_model_train_test(data, cfg, prediction_time):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M1TransformerSin(input_size = 3,dim_val = cfg["dim_val"], window_size= cfg["window_size"]).to(device)
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
    train, targets_train, test, targets_test = pair_input_output(data, False, prediction_time)

    train = np.array([instance.T for instance in train])

    train_dataset = M1Dataset(train, targets_train)
    train_loader  = DataLoader(train_dataset, batch_size= cfg["batch_size"], num_workers= cfg["num_workers"], pin_memory= True, persistent_workers= True, shuffle=True)

    test = np.array([instance.T for instance in test])
    targets_test = np.array(list(targets_test.values()))
    test_dataset = M1Dataset(test, targets_test)
    test_loader = DataLoader(test_dataset, batch_size= cfg["batch_size"], num_workers= cfg["num_workers"],pin_memory=True, persistent_workers=True, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    return optimizer, model,  train_loader, test_loader, device, loss_fn

def load_checkpoint_safely(model, path):
    """Load a state dict that may or may not have a torch.compile '_orig_mod.' prefix,
    into a model that may or may not currently be compiled."""
    raw_sd = torch.load(path)

    # detect whether the FILE has the prefix
    file_has_prefix = any(k.startswith("_orig_mod.") for k in raw_sd.keys())

    # detect whether the MODEL currently expects the prefix (i.e., is compiled)
    model_is_compiled = hasattr(model, "_orig_mod")
    target = model._orig_mod if model_is_compiled else model

    if file_has_prefix:
        # strip it so it matches the plain (uncompiled) module's key names
        raw_sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    target.load_state_dict(raw_sd)
    return model


def main():
    data = pd.read_csv('data/data.csv')
    
    cfg = load_config()
    create_result_dirs(cfg)
    
    with open('data/event_timestamps.txt', 'r') as event_file:
        lines = event_file.readlines()
    event_times = [line.split() for line in lines]


    ### TRAINING ###
    for pt in cfg["prediction_time"]:
            # print(f"======== Evaluating Lin Regress, t={pt}  ========")
            # run_linear_baseline(data, pt)
            
            
            # for seed in  range(cfg["n_seeds"]):
            #     optimizer,model, train_loader, test_loader, device, loss_fn = set_up_model_train_test(data, cfg, int(pt))
            #     model_name = "M1-" + str(seed).zfill(2)

            #     if cfg["train_new_models"]:
            #         torch.manual_seed(seed)
            #         model, _ = train_model(model, train_loader, optimizer, loss_fn, device, tag=model_name)
            #         torch.save(model.state_dict(), f"models/t={pt}/{model_name}.pt")

            #     ### TESTING ###
            #     model = load_checkpoint_safely(model, f"models/t={pt}/{model_name}.pt")
            #     predictions  = test_model(model, test_loader, device, loss_fn).detach().cpu().numpy()
            #     np.savetxt(f"results/transformer/t={pt}/{model_name}/predictions.txt", predictions ,delimiter=",")
            #     ### FROM TORRES MAIN FUNCTION - EVALUATION ###

            #     _,_,_, targets_test = pair_input_output(data, False, pt)

            #     event_times_test = []
            #     first_test_event_time = list(targets_test.keys())[0]

            #     for event in event_times:
            #         if event[0] >= first_test_event_time:
            #             event_times_test.append(event) 

            #     ### EVALUTE PREDICTIONS ###
            #     target_times = list(targets_test.keys())
               
            #     predictions = {target_times[i]: predictions[i] for i in range(len(predictions))}

            #     print(f"======== Evaluating {model_name}, t={pt} ========")
            #     evaluate(targets_test, predictions, event_times_test, data, path = f"results/transformer/t={pt}/{model_name}", display= False)

            for app in ["app0", "app1", "app2","app3"]:
                print(app)
                score_forecast(pt, app)

    ### RECORDING RESULTS ###

    create_result_csv(cfg)    
### END ###
if __name__ == "__main__":
        main()
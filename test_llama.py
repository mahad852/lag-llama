import numpy as np
from datasets.ecg_mit import ECG_MIT
import random
import os

from lag_llama.gluon.estimator import LagLlamaEstimator
import torch
from huggingface_hub import snapshot_download
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

context_len = 512
pred_len = 64

batch_size = 16

data_path = "/home/mali2/datasets/ecg/MIT-BIH-splits.npz"

device = torch.device("cuda")

# ckpt = torch.load("lightning_logs/version_1/checkpoints/epoch=46-step=2350.ckpt", map_location="cpu") # Uses GPU since in this Colab we use a GPU.
# ckpt = torch.load("lightning_logs/version_3/checkpoints/epoch=0-step=1562.ckpt", map_location="cpu")
ckpt = torch.load("lightning_logs/version_4/checkpoints/epoch=19-step=62500.ckpt", map_location="cpu")
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

rope_scaling_arguments = {
    "type": "linear",
    "factor": max(1.0, (context_len + pred_len) / estimator_args["context_length"]),
}

estimator = LagLlamaEstimator(
    # ckpt_path="lightning_logs/version_1/checkpoints/epoch=46-step=2350.ckpt",
    ckpt_path="lightning_logs/version_4/checkpoints/epoch=19-step=62500.ckpt",
    prediction_length=pred_len,
    context_length=context_len, # Lag-Llama was trained with a context length of 32, but can work with any context length

    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],
    scaling=estimator_args["scaling"],
    time_feat=estimator_args["time_feat"],
    rope_scaling=None,
    batch_size=batch_size,
    num_parallel_samples=100,
    device=device,
)

lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()
predictor = estimator.create_predictor(transformation, lightning_module)


def get_lag_llama_predictions(dataset, predictor, num_samples=100):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


total_times = []

def build_timeseries(dataset):
    targets = None

    item_ids = []

    for i in range(1, len(dataset.files), 2):
        xy = dataset[dataset.files[i]]
        x, y = xy[:context_len], xy[context_len:context_len + pred_len]

        vals = np.append(x, y)

        if targets is None:
            targets = vals
        else:
            targets = np.append(targets, vals)

        item_ids.extend([str(i)] * (x.shape[0] + y.shape[0]))

    df = pd.DataFrame(data={"item_id" : item_ids, "target" : targets})
    df["item_id"] = df["item_id"].astype("string")
    df["target"] = df["target"].astype("float32")
    df = df.set_index(pd.date_range('1990', freq='3ms', periods=df.shape[0]))

    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    return dataset

dataset = build_timeseries(np.load(data_path))

backtest_dataset = dataset
num_samples = 20 # number of samples sampled from the probability distribution for each timestep

forecasts, tss = get_lag_llama_predictions(dataset, predictor, num_samples)

print(len(forecasts), len(tss))

mse_by_plen = {plen : 0 for plen in range(1, pred_len + 1)}
rmse_by_plen = {plen : 0 for plen in range(1, pred_len + 1)}
mae_by_plen = {plen : 0 for plen in range(1, pred_len + 1)}

total = 0

for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
    total += 1

    median_forecast = np.quantile(forecast.samples, 0.5, axis=0)
    gt = ts.iloc[:, 0].values[-pred_len:]

    for plen in range(1, pred_len + 1):
        mae_by_plen[plen] += mean_absolute_error(gt[:plen], median_forecast[:plen])
        mse_by_plen[plen] += mean_squared_error(gt[:plen], median_forecast[:plen])

    print(f"Iteration: {i} | MSE: {mean_squared_error(gt, median_forecast)}, RMSE: {np.sqrt(mean_squared_error(gt, median_forecast))} MAE: {mean_absolute_error(gt, median_forecast)}")

for plen in range(1, pred_len + 1):
    mse_by_plen[plen] /= total
    rmse_by_plen[plen] = np.sqrt(mse_by_plen[plen])
    mae_by_plen[plen] /= total

print(f"Finished inference. MSE: {mse_by_plen[pred_len]} RMSE: {rmse_by_plen[pred_len]} MAE: {mae_by_plen[pred_len]}")
# evaluator = Evaluator()
# agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
# print(agg_metrics)

if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"LagLlama_V4_{context_len}_{pred_len}.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_plen[p_len]},{rmse_by_plen[p_len]},{mae_by_plen[p_len]}")
        if p_len != pred_len:
            f.write("\n")


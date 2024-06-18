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

if not os.path.exists("lagllama_weights"):
    snapshot_download("time-series-foundation-models/Lag-Llama", local_dir="lagllama_weights")

context_len = 512
pred_len = 64
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/mali2/datasets/ecg/MIT-BIH.npz")

batch_size = 64
total_samples = 100000


indices = random.sample(range(len(ecg_dataset)), total_samples)

device = torch.device("cuda")

ckpt = torch.load("lagllama_weights/lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

rope_scaling_arguments = {
    "type": "linear",
    "factor": max(1.0, (context_len + pred_len) / estimator_args["context_length"]),
}

estimator = LagLlamaEstimator(
    ckpt_path="lagllama_weights/lag-llama.ckpt",
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
    trainer_kwargs = {"max_epochs": 1},
    device=device,
    lr=5e-4,
    num_batches_per_epoch=int(total_samples/batch_size)
)

lightning_module = estimator.create_lightning_module()
transformation = estimator.create_transformation()
# predictor = estimator.create_predictor(transformation, lightning_module)


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

def build_timeseries(dataset: ECG_MIT, indices: list[int]):
    targets = np.zeros((context_len + pred_len) * total_samples)

    item_ids = []

    for i, index in enumerate(indices):
        x, y = dataset[index]
        targets[i * (context_len + pred_len) : (i + 1) * (context_len + pred_len)] = np.append(x, y)
        item_ids.extend([str(index)] * (x.shape[0] + y.shape[0]))

    df = pd.DataFrame(data={"item_id" : item_ids, "target" : targets})
    df["item_id"] = df["item_id"].astype("string")
    df["target"] = df["target"].astype("float32")
    df.set_index(pd.date_range('1990', freq='3ms', periods=df.shape[0]), inplace=True)

    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
    return dataset
    

dataset = build_timeseries(ecg_dataset, indices)

backtest_dataset = dataset
num_samples = 20 # number of samples sampled from the probability distribution for each timestep


predictor = estimator.train(dataset, cache_data=True, shuffle_buffer_length=1000)

# get_lag_llama_predictions(dataset, predictor, num_samples)


if not os.path.exists("logs"):
    os.mkdir("logs")


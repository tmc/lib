"""Produce an offline MLflow 2.22.0 history fixture through the native SDK.

Run in a task-scoped environment with mlflow-skinny==2.22.0. Pass new, explicit
--store and --output directories; this script never contacts a tracking server.
The logged values are test controls, not a convergence or GPU measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow.protos.service_pb2 import GetMetricHistory, Metric
from mlflow.tracking import MlflowClient
from mlflow.utils.proto_json_utils import message_to_json

parser = argparse.ArgumentParser()
parser.add_argument("--store", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if mlflow.__version__ != "2.22.0":
    raise RuntimeError("requires mlflow-skinny==2.22.0")
args.store.mkdir(parents=True)
args.output.mkdir(parents=True, exist_ok=True)
client = MlflowClient(tracking_uri=args.store.resolve().as_uri())
experiment = client.create_experiment("lane108-native-offline")
run_id = client.create_run(experiment).info.run_id
for step, value in [(0, 0.5), (1, 0.3), (3, 0.2)]:
    client.log_metric(run_id, "loss", value, 1700000000000 + step * 10, step=step)
history = client.get_metric_history(run_id, "loss")
response = GetMetricHistory.Response(
    metrics=[
        Metric(key=m.key, value=m.value, timestamp=m.timestamp, step=m.step)
        for m in history
    ]
)
raw = (message_to_json(response) + "\n").encode()
(args.output / "mlflow-native.json").write_bytes(raw)
receipt = {
    "producer": "mlflow-skinny",
    "version": mlflow.__version__,
    "run_id": run_id,
    "transport": "file-store; no endpoint/account",
    "source": "MlflowClient.log_metric/get_metric_history; GetMetricHistory.Response; mlflow.utils.proto_json_utils.message_to_json",
    "sha256": hashlib.sha256(raw).hexdigest(),
    "raw_bytes": len(raw),
    "metrics": len(history),
    "steps": [m.step for m in history],
    "qualification": "native offline producer serialization; controlled metric values, no convergence/checkpoint/GPU proof",
}
(args.output / "mlflow-native-receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt))

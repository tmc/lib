# Entity adapter fixtures

These are synthetic known-answer documents, not captured cluster or GPU data.
They preserve fields of the public producer schemas:

- `podresources-v1.json`: protobuf-JSON lowerCamelCase rendering of Kubernetes
  v1.32.0 PodResources v1 `ListPodResourcesResponse`:
  https://github.com/kubernetes/kubernetes/blob/v1.32.0/staging/src/k8s.io/kubelet/pkg/apis/podresources/v1/api.proto
- `squeue-v0.0.43.json`: selected fields from Slurm 25.05 data_parser v0.0.43
  `JOB_INFO` records:
  https://github.com/SchedMD/slurm/blob/slurm-25.05/src/plugins/data_parser/v0.0.43/parsers.c
  The official `squeue --json` producer is described at
  https://slurm.schedmd.com/squeue.html.

The readers preserve opaque device IDs, pod namespace/name and scheduler TRES
strings. They do not authenticate an external exporter, invent pod UIDs, map
plugin IDs to physical GPU UUIDs, or turn allocation counts into consumption.
SourceRevision must identify the actual supplying producer. Live kubelet,
scheduler, MPS and MIG qualification remains separate.

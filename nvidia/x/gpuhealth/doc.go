// Package gpuhealth checks, from inside a process about to measure, that it
// can still reach an NVIDIA GPU: the device nodes open read-write, NVML
// answers through nvidia-smi, and the expected GPU is the one present.
//
// It exists because a container can lose its GPU in the middle of a
// campaign while every outward sign stays the same. In the observed case the
// /dev/nvidia* nodes kept mode 0666, but opening them failed, NVML exited
// 255 with "Unknown Error", CUDA reported no devices, and CUPTI's next call
// failed with CUPTI_ERROR_UNKNOWN. Permission bits never qualify access, so
// [ProbeNodes] opens each node and records the errno.
//
// [ParseSMIXML] reads nvidia-smi -q -x. A GPU is chosen by UUID and its node
// is /dev/nvidia<minor_number>; the CUDA or NVML enumeration index is never a
// node number. (nvidia-smi --query-gpu=minor_number is not a valid query:
// it exits 2 and prints the error on standard output.) [DockerArgs] maps
// exactly that GPU's nodes into a container instead of relying on --gpus.
//
// [Probe] combines these with the driver version and the process's cgroup
// into a [Report] whose Problems name the likely cause of each failure.
// Probe never creates a CUDA context; a fresh-process cuInit check is the
// cudadoctor command's -smoke mode.
package gpuhealth

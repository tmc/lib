# `nvtelemetry` Kubernetes DaemonSet template

This is a source template, not a deployable or qualified DaemonSet. It contains
two fail-closed placeholders: an immutable image reference and an
authenticated-probe strategy. The latter must be resolved for the selected
listener/configuration and pinned image before an operator may render and
apply the manifest. No cluster, image digest, runtime, or deployment is
selected here.

## Image digest and render guard

The image must use a registry and repository path followed by a lowercase
SHA-256 digest, for example
`registry.example/team/nvtelemetry@sha256:<64 lowercase hex digits>`. The
example is not a real image reference. Render a private copy only after
choosing a real digest and supplying a reviewed authenticated-probe
implementation. The `grep` checks refuse to proceed while either marker or a
floating `latest` tag remains:

```sh
set -eu
image_ref='registry.example/team/nvtelemetry@sha256:REPLACE_WITH_64_HEX_DIGEST'
printf '%s\n' "$image_ref" | grep -Eq '^[a-z0-9]+([.-][a-z0-9]+)*(:[0-9]+)?(/[a-z0-9]+([._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$' || {
  echo 'image_ref must be a repository reference pinned by a lowercase sha256 digest' >&2
  exit 1
}

rendered='daemonset.rendered.yaml'
sed "s|image: \"REQUIRED_IMAGE_DIGEST\"|image: \"$image_ref\"|" daemonset.yaml > "$rendered"
if grep -Eq 'REQUIRED_IMAGE_DIGEST|REQUIRED_AUTHENTICATED_PROBE|:latest([[:space:]]|$)' "$rendered"; then
  echo 'HOLD: resolve image and authenticated probe placeholders before apply' >&2
  exit 1
fi

# Apply only after reviewing the completed probe, ConfigMap, Secret and node
# target mapping for the selected cluster.
kubectl apply -f "$rendered"
```

The example digest deliberately fails validation. The manifest's annotation
`nvidia.com/nvtelemetry-probe-strategy: REQUIRED_AUTHENTICATED_PROBE` is also a
HOLD marker; replacing it alone is insufficient. Add a probe that can reach
the configured listener and authenticate without embedding bearer credentials
in the Pod spec, then replace the marker with the reviewed strategy identity.
The generic template intentionally has no HTTP liveness or readiness probes.

## ConfigMap, UUID selection, and node mapping

The referenced ConfigMap must be named `nvtelemetry-config` and contain
`config.json`. Live serving refuses an empty `device_uuids` allowlist and
constructs its sampler only for the exact listed UUIDs. `NVIDIA_VISIBLE_DEVICES=all`
controls runtime visibility; it does **not** select or prove sampled targets.
For example, the operator-generated ConfigMap needs a configuration shaped
like this, populated with real values after verifying each UUID against the
target node and admitted allocation:

```json
{
  "listen": "0.0.0.0:9464",
  "interval": "1s",
  "max_age": "5s",
  "device_uuids": ["GPU-REPLACE_WITH_VERIFIED_UUID"],
  "allow_remote": true,
  "tls_cert": "/run/nvtelemetry/credentials/tls.crt",
  "tls_key": "/run/nvtelemetry/credentials/tls.key",
  "auth_token_file": "/run/nvtelemetry/credentials/bearer-token"
}
```

The sample UUID is invalid and must not be used. The named Kubernetes Secret
`nvtelemetry-serving-credentials` must supply `tls.crt`, `tls.key`, and
`bearer-token`; the manifest mounts it read-only at the referenced path. The
listener's non-loopback mode requires all three of `allow_remote`, TLS cert/key,
and bearer authentication. Health, readiness, and metrics routes use the same
bearer authentication. Do not weaken that policy to make a kubelet probe pass.

The DaemonSet shares one ConfigMap among all replicas. That exact UUID list must
be valid on every node selected by the operator's verified node label and must
match the intended allocation. If nodes have different UUIDs, create and bind
per-node or per-homogeneous-target configurations using a separately reviewed
rendering/scheduling scheme. This template has no Kubernetes API access and
does not discover allocations or map `nvidia.com/gpu` resource IDs to UUIDs.
Without that external, pinned node→allocation→UUID mapping the workload is
unsupported and sampled-device coverage is unknown.

`runtimeClassName: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`, and
`NVIDIA_DRIVER_CAPABILITIES=utility` describe the required runtime inputs. A
cluster operator must verify the registered RuntimeClass, toolkit/CDI mode,
NVML library injection, all-device visibility, UID/GID device access, and node
placement. This template requests no exclusive `nvidia.com/gpu` resource; a
one-GPU limit would alter scheduling and expose only an allocated device. A
runtime that injects only scheduled resources, lacks the named class, or cannot
provide the required visibility is unsupported here.

## Least privilege and scope

The Pod runs as UID/GID 10001, applies `RuntimeDefault` seccomp, drops all Linux
capabilities, disables privilege escalation and service-account token
automounting, uses a read-only root filesystem, and mounts configuration and
credentials read-only. It does not mount host `/dev`, use host PID/network
namespaces, or request `CAP_PERFMON`, `CAP_SYS_ADMIN`, `CAP_BPF`, or
`NET_ADMIN`. NVML-only collection requires the runtime-injected devices and
library; BPF and CUPTI modes are unsupported by this template and require
separate admission and privilege review.

There are no generic probes because the service defaults to loopback and its
routes require bearer authentication; remote serving additionally requires TLS
and `allow_remote`. A kubelet HTTP probe to the Pod address would not match the
default listener, and an unauthenticated probe would be rejected. The probe
placeholder keeps this template in HOLD until an image-supported, authenticated
strategy is selected and independently reviewed. Do not copy the bearer token
into a probe header.

The separate systemd template in this directory still contains
`CAP_PERFMON`; it was not changed by this Kubernetes-only correction and is not
qualified here for NVML-only operation.

## Qualification status

Host YAML parsing and structural tests do not qualify RuntimeClass behavior,
device injection, node selection, UUID mapping, permissions, probes, Kubernetes
admission/RBAC, NVML/native ABI, physical GPUs, production soak, or deployment.
Retain each as a separate gate. Until the image, authenticated probe, ConfigMap,
Secret, and node-target mapping are rendered and reviewed, deployment status is
HOLD.

# `nvtelemetry` Kubernetes DaemonSet template

This is a source template, not a deployable or qualified DaemonSet. It contains
fail-closed placeholders for an immutable image reference and the TLS server
name used by its authenticated exec probes. The image must contain this
repository's `nvtelemetry probe` command. No cluster, image digest, runtime,
certificate, or deployment is selected here.

## Image digest and render guard

The image must use a registry and repository path followed by a lowercase
SHA-256 digest, for example
`registry.example/team/nvtelemetry@sha256:<64 lowercase hex digits>`. The
example is not a real image reference. Render a private copy only after
choosing a real digest and the DNS name present in the probe certificate SAN.
The checks refuse to proceed while either marker or a floating `latest` tag
remains:

```sh
set -eu
image_ref='registry.example/team/nvtelemetry@sha256:REPLACE_WITH_64_HEX_DIGEST'
printf '%s\n' "$image_ref" | grep -Eq '^[a-z0-9]+([.-][a-z0-9]+)*(:[0-9]+)?(/[a-z0-9]+([._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$' || {
  echo 'image_ref must be a repository reference pinned by a lowercase sha256 digest' >&2
  exit 1
}
tls_server_name='REPLACE_WITH_CERTIFICATE_DNS_NAME'
test "${#tls_server_name}" -le 253 || {
  echo 'tls_server_name must be at most 253 characters' >&2
  exit 1
}
printf '%s\n' "$tls_server_name" | grep -Eq '^([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)(\.([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?))*$' || {
  echo 'tls_server_name must be a DNS name present in the serving certificate SAN' >&2
  exit 1
}

rendered='daemonset.rendered.yaml'
sed -e "s|image: \"REQUIRED_IMAGE_DIGEST\"|image: \"$image_ref\"|" \
    -e "s|REQUIRED_TLS_SERVER_NAME|$tls_server_name|g" daemonset.yaml > "$rendered"
if grep -Eq 'REQUIRED_IMAGE_DIGEST|REQUIRED_TLS_SERVER_NAME|REPLACE_WITH_CERTIFICATE_DNS_NAME|:latest([[:space:]]|$)' "$rendered"; then
  echo 'HOLD: resolve image and probe TLS server name before apply' >&2
  exit 1
fi

# Apply only after reviewing the completed probe, ConfigMap, Secret and node
# target mapping for the selected cluster.
kubectl apply -f "$rendered"
```

The example digest and TLS name deliberately fail validation. The probes run
the same `nvtelemetry` binary as the server. They read the bearer token only
from `auth_token_file` in the mounted config, require the explicitly mounted CA
file and certificate name, and connect to the configured numeric listener using
loopback for wildcard binds. They use verified HTTPS, do not follow redirects,
and have fixed request and dial deadlines. The bearer token never appears in
the Pod spec or command arguments. Liveness checks `/healthz`; readiness checks
the existing `/readyz` snapshot-freshness predicate.

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
`nvtelemetry-serving-credentials` must supply `tls.crt`, `tls.key`,
`bearer-token`, and `ca.crt`; `ca.crt` must contain the trust root that issued
the serving certificate. The manifest mounts it read-only at the referenced
path. The configured TLS server name must match a certificate DNS SAN. The
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

The exec probes do not change the listener's production authentication. The
operator must render the DNS SAN name and provide the CA and bearer token in the
named Secret; the ConfigMap must point `auth_token_file` at that mounted token.
Keep the explicit nonempty `device_uuids` list and the external node→allocation→
UUID mapping described above. Rendering these values is not evidence that the
selected image contains this CLI, the Secret and ConfigMap match, the
certificate is valid, or the runtime injects usable NVML devices. Do not copy
the bearer token into a probe argument.

The separate systemd template in this directory still contains
`CAP_PERFMON`; it was not changed by this Kubernetes-only correction and is not
qualified here for NVML-only operation.

## Qualification status

Host YAML parsing and structural tests do not qualify RuntimeClass behavior,
device injection, node selection, UUID mapping, permissions, probes, Kubernetes
admission/RBAC, NVML/native ABI, physical GPUs, production soak, or deployment.
Retain each as a separate gate. Until the image, probe TLS name, ConfigMap,
Secret, and node-target mapping are rendered and reviewed, deployment status is
HOLD.

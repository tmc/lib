package main

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestDaemonSetManifestLeastPrivilege(t *testing.T) {
	manifest, err := os.ReadFile(filepath.Join("packaging", "daemonset.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(manifest)

	for _, want := range []string{
		"kind: DaemonSet",
		"runtimeClassName: nvidia",
		"automountServiceAccountToken: false",
		"runAsNonRoot: true",
		"readOnlyRootFilesystem: true",
		"allowPrivilegeEscalation: false",
		"seccompProfile:\n          type: RuntimeDefault",
		"capabilities:\n              drop:\n                - ALL",
		"- key: nvidia.com/gpu",
		"- name: NVIDIA_VISIBLE_DEVICES\n              value: all",
		"- name: NVIDIA_DRIVER_CAPABILITIES\n              value: utility",
		"image: \"REQUIRED_IMAGE_DIGEST\"",
		"nvidia.com/nvtelemetry-probe-strategy: nvtelemetry-authenticated-exec-v1",
		"-kind=liveness",
		"-kind=readiness",
		"-ca-file=/run/nvtelemetry/credentials/ca.crt",
		"-server-name=REQUIRED_TLS_SERVER_NAME",
		"livenessProbe:\n            exec:",
		"readinessProbe:\n            exec:",
		"secretName: nvtelemetry-serving-credentials",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("daemonset.yaml is missing required manifest text %q", want)
		}
	}

	for _, forbidden := range []string{
		"CAP_PERFMON",
		"CAP_SYS_ADMIN",
		"CAP_BPF",
		"NET_ADMIN",
		"privileged: true",
		"hostPID: true",
		"hostNetwork: true",
		"hostIPC: true",
		"              add:",
		"hostPath:",
		"mountPath: /dev",
		"path: /dev",
	} {
		if strings.Contains(s, forbidden) {
			t.Errorf("daemonset.yaml contains forbidden broad privilege or device mount %q", forbidden)
		}
	}

	if regexp.MustCompile(`(?m)^\s*image:\s*\S+:latest(?:\s|$)`).MatchString(s) {
		t.Error("daemonset.yaml must not use a floating latest image")
	}
	if strings.Contains(s, "REQUIRED_AUTHENTICATED_PROBE") {
		t.Error("generic probe hold marker must be replaced by the binary exec probes")
	}
	if regexp.MustCompile(`(?m)^\s+nvidia\.com/gpu:\s*\d+\s*$`).MatchString(s) {
		t.Error("node agent must not request an exclusive NVIDIA GPU allocation")
	}
}

func TestREADMEProvidesFailClosedImageRenderGuard(t *testing.T) {
	manifest, err := os.ReadFile(filepath.Join("packaging", "daemonset.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	readme, err := os.ReadFile(filepath.Join("packaging", "README.md"))
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(string(manifest), "REQUIRED_IMAGE_DIGEST") {
		t.Fatal("checked-in template must retain its unresolved digest placeholder")
	}
	if !strings.Contains(string(manifest), `image: "REQUIRED_IMAGE_DIGEST"`) {
		t.Fatal("image placeholder must remain a quoted YAML string")
	}

	readmeText := string(readme)
	for _, want := range []string{
		"/team/nvtelemetry@sha256:<64 lowercase hex digits>",
		`image_ref='registry.example/team/nvtelemetry@sha256:REPLACE_WITH_64_HEX_DIGEST'`,
		"@sha256:[0-9a-f]{64}$",
		`image: \"REQUIRED_IMAGE_DIGEST\"|image: \"$image_ref\"`,
		"REQUIRED_IMAGE_DIGEST|REQUIRED_TLS_SERVER_NAME|REPLACE_WITH_CERTIFICATE_DNS_NAME|:latest",
		"HOLD: resolve image and probe TLS server name before apply",
		"tls_server_name must be a DNS name present in the serving certificate SAN",
		"tls_server_name must be at most 253 characters",
		"Liveness checks `/healthz`",
		"the existing `/readyz` snapshot-freshness predicate",
		"bearer-token",
		"ca.crt",
		"kubectl apply -f \"$rendered\"",
		"Live serving refuses an empty `device_uuids` allowlist",
		"RuntimeClass",
		"node→allocation→UUID mapping",
	} {
		if !strings.Contains(readmeText, want) {
			t.Errorf("README is missing fail-closed render/deployment guard text %q", want)
		}
	}
	if strings.Index(readmeText, "HOLD: resolve image and probe TLS server name") > strings.Index(readmeText, "kubectl apply -f") {
		t.Error("README deployment guard must appear before kubectl apply")
	}
}

func TestDaemonSetProbeMutationsAreDetected(t *testing.T) {
	manifest, err := os.ReadFile(filepath.Join("packaging", "daemonset.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(manifest)
	mutations := []struct {
		name string
		from string
		to   string
	}{
		{"remove liveness auth config", "-kind=liveness", "-kind=wrong"},
		{"remove readiness probe", "readinessProbe:", "readinessProbe-disabled:"},
		{"remove CA mount argument", "-ca-file=/run/nvtelemetry/credentials/ca.crt", "-ca-file=/tmp/ca.crt"},
		{"remove certificate name", "-server-name=REQUIRED_TLS_SERVER_NAME", "-server-name=localhost"},
		{"remove Secret mount", "mountPath: /run/nvtelemetry/credentials", "mountPath: /run/other"},
		{"replace Secret", "secretName: nvtelemetry-serving-credentials", "secretName: other"},
		{"add unauthenticated HTTP probe", "livenessProbe:", "livenessProbe:\n            httpGet:\n              path: /healthz\n              port: 9464\n            exec:"},
	}
	for _, mutation := range mutations {
		t.Run(mutation.name, func(t *testing.T) {
			mutant := strings.Replace(s, mutation.from, mutation.to, 1)
			if mutant == s {
				t.Fatal("mutation did not change manifest")
			}
			if err := validateDaemonSetProbeContract(mutant); err == nil {
				t.Fatal("mutated manifest unexpectedly passed probe contract")
			}
		})
	}
}

func validateDaemonSetProbeContract(s string) error {
	for _, want := range []string{
		"livenessProbe:\n            exec:",
		"readinessProbe:\n            exec:",
		"-kind=liveness",
		"-kind=readiness",
		"-ca-file=/run/nvtelemetry/credentials/ca.crt",
		"-server-name=REQUIRED_TLS_SERVER_NAME",
		"nvidia.com/nvtelemetry-probe-strategy: nvtelemetry-authenticated-exec-v1",
		"mountPath: /run/nvtelemetry/credentials\n              readOnly: true",
		"secretName: nvtelemetry-serving-credentials",
		"automountServiceAccountToken: false",
	} {
		if !strings.Contains(s, want) {
			return errors.New("probe contract is missing a required field")
		}
	}
	for _, want := range []string{
		"-config=/etc/nvtelemetry/config.json",
		"-ca-file=/run/nvtelemetry/credentials/ca.crt",
		"-server-name=REQUIRED_TLS_SERVER_NAME",
	} {
		if strings.Count(s, want) != 2 {
			return errors.New("both probes must use the same protected configuration")
		}
	}
	if strings.Contains(s, "httpGet:") {
		return errors.New("unauthenticated HTTP probes are not supported")
	}
	return nil
}

package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestNVMLServiceCapabilities(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("packaging", "nvtelemetry.service"))
	if err != nil {
		t.Fatal(err)
	}
	if err := validateNVMLServiceTemplate(string(data)); err != nil {
		t.Fatal(err)
	}
}

func TestNVMLServiceDirectiveControls(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("packaging", "nvtelemetry.service"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, tt := range []struct{ name, from, to string }{
		{"ambient perfmon", "AmbientCapabilities=", "AmbientCapabilities=CAP_PERFMON"},
		{"bounding perfmon", "CapabilityBoundingSet=", "CapabilityBoundingSet=CAP_PERFMON"},
		{"ambient admin", "AmbientCapabilities=", "AmbientCapabilities=CAP_SYS_ADMIN"},
		{"bounding bpf", "CapabilityBoundingSet=", "CapabilityBoundingSet=CAP_BPF"},
		{"complement set", "CapabilityBoundingSet=", "CapabilityBoundingSet=~CAP_SYS_ADMIN"},
		{"commented reset", "AmbientCapabilities=", "# AmbientCapabilities="},
		{"missing reset", "CapabilityBoundingSet=\n", ""},
		{"continued capability", "AmbientCapabilities=", "AmbientCapabilities=\\\n CAP_PERFMON"},
		{"duplicate capability", "AmbientCapabilities=", "AmbientCapabilities=\nAmbientCapabilities=CAP_BPF"},
		{"root user", "User=nvtelemetry", "User=root"},
		{"disabled no-new-privileges", "NoNewPrivileges=true", "NoNewPrivileges=false"},
		{"privileged command", "ExecStart=/usr/local/bin", "ExecStart=+/usr/local/bin"},
		{"wrong section", "[Service]", "[Unit]"},
		{"private devices", "PrivateDevices=false", "PrivateDevices=true"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if strings.Count(s, tt.from) != 1 {
				t.Fatal("control must replace one actual directive")
			}
			if err := validateNVMLServiceTemplate(strings.Replace(s, tt.from, tt.to, 1)); err == nil {
				t.Fatal("changed active service directive was accepted")
			}
		})
	}
	// A reset mentioned in comments or another section cannot substitute for
	// a real [Service] assignment; a later section cannot add capabilities.
	for _, text := range []string{
		strings.Replace(s, "AmbientCapabilities=\n", "", 1) + "\n# AmbientCapabilities=\n",
		strings.Replace(s, "AmbientCapabilities=\n", "", 1) + "\n[Unit]\nAmbientCapabilities=\n",
		s + "\n[Service]\nAmbientCapabilities=CAP_SYS_ADMIN\n",
	} {
		if err := validateNVMLServiceTemplate(text); err == nil {
			t.Fatal("inactive reset or appended capability bypassed directive checks")
		}
	}
	for _, text := range []string{
		s + "\n# AmbientCapabilities=CAP_PERFMON\n; CapabilityBoundingSet=CAP_SYS_ADMIN\n",
		strings.ReplaceAll(s, "\n", "\r\n"),
		strings.Replace(s, "AmbientCapabilities=", "  AmbientCapabilities =  ", 1),
	} {
		if err := validateNVMLServiceTemplate(text); err != nil {
			t.Fatalf("comment/whitespace changed active semantics: %v", err)
		}
	}
}

// This is a strict guard for the checked-in template, not a systemd parser or
// an evaluator of installed units/drop-ins. Duplicate settings and continued
// lines require deliberate review rather than silently changing this contract.
func validateNVMLServiceTemplate(text string) error {
	service := make(map[string][]string)
	section := ""
	scanner := bufio.NewScanner(strings.NewReader(text))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasSuffix(line, "\\") {
			return errors.New("continued service directives require review")
		}
		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") {
				return errors.New("malformed unit section")
			}
			section = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
			continue
		}
		if section != "Service" {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return errors.New("malformed service directive")
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		service[key] = append(service[key], value)
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	for key, want := range map[string][]string{
		"AmbientCapabilities": {""}, "CapabilityBoundingSet": {""},
		"User": {"nvtelemetry"}, "Group": {"nvtelemetry"},
		"ExecStart":       {"/usr/local/bin/nvtelemetry serve -config /etc/nvtelemetry/config.json"},
		"ExecReload":      {"/bin/kill -HUP $MAINPID"},
		"NoNewPrivileges": {"true"}, "ProtectSystem": {"strict"},
		"ProtectHome": {"true"}, "PrivateTmp": {"true"}, "PrivateDevices": {"false"},
		"ProtectKernelTunables": {"true"}, "ProtectControlGroups": {"true"},
		"RestrictSUIDSGID": {"true"}, "LockPersonality": {"true"}, "MemoryDenyWriteExecute": {"true"},
		"DeviceAllow": {"/dev/nvidiactl rw", "/dev/nvidia-uvm rw", "/dev/nvidia-uvm-tools rw", "/dev/nvidia-modeset rw", "char-nvidia* rw"},
	} {
		if !slices.Equal(service[key], want) {
			return fmt.Errorf("service directive %s = %q, want %q", key, service[key], want)
		}
	}
	return nil
}

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

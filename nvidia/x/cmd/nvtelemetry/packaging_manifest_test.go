package main

import (
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
		"nvidia.com/nvtelemetry-probe-strategy: REQUIRED_AUTHENTICATED_PROBE",
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
	if !strings.Contains(s, "REQUIRED_AUTHENTICATED_PROBE") || strings.Contains(s, "livenessProbe:") || strings.Contains(s, "readinessProbe:") {
		t.Error("generic probes must remain on HOLD until an authenticated strategy is supplied")
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
		"REQUIRED_IMAGE_DIGEST|REQUIRED_AUTHENTICATED_PROBE|:latest",
		"HOLD: resolve image and authenticated probe placeholders before apply",
		"kubectl apply -f \"$rendered\"",
		"Live serving refuses an empty `device_uuids` allowlist",
		"RuntimeClass",
		"node→allocation→UUID mapping",
	} {
		if !strings.Contains(readmeText, want) {
			t.Errorf("README is missing fail-closed render/deployment guard text %q", want)
		}
	}
	if strings.Index(readmeText, "HOLD: resolve image and authenticated probe placeholders") > strings.Index(readmeText, "kubectl apply -f") {
		t.Error("README deployment guard must appear before kubectl apply")
	}
}

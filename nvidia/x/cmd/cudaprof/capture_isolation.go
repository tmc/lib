package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

func captureProcessIdentity(pid int) isolation.Process {
	host, _ := os.Hostname()
	p := isolation.Process{Host: host, PID: pid}
	// /proc starttime is generation identity within this host boot. Boot ID
	// prevents a reboot from making an old PID/start pair look current.
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return p
	}
	end := strings.LastIndexByte(string(b), ')')
	if end < 0 {
		return p
	}
	fields := strings.Fields(string(b)[end+1:])
	if len(fields) < 20 {
		return p
	}
	if _, err := strconv.ParseUint(fields[19], 10, 64); err != nil {
		return p
	}
	boot, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
	if err != nil {
		return p
	}
	p.Start = strings.TrimSpace(string(boot)) + ":" + fields[19]
	return p
}

func attachCaptureIsolation(result *captureResult, path string) {
	if path == "" && result.Isolation != nil {
		return
	}
	out := isolation.Evaluate(nil, "", result.IsolationBinding)
	result.Isolation = &out
	if path == "" {
		return
	}
	f, err := os.Open(path)
	if err != nil {
		out.Reasons = []string{"observer receipt unavailable: " + err.Error()}
		return
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, isolation.MaxBytes+1))
	if err != nil || len(raw) > isolation.MaxBytes {
		out.Reasons = []string{"observer receipt unreadable or exceeds size limit"}
		return
	}
	result.isolationRaw = raw
	receipt, digest, err := isolation.Read(bytes.NewReader(raw))
	if err != nil {
		out.SHA256 = digest
		out.Reasons = []string{err.Error()}
		return
	}
	// A profiler stop is not proven final GPU completion; device identity and
	// completion coverage remain unavailable until independently observed.
	out = isolation.Evaluate(receipt, digest, result.IsolationBinding)
	result.Isolation = &out
}

// captureIsolation validates retained bytes against separately retained capture
// binding. It never trusts a claimed status disposition alone.
func captureIsolation(cap *cudacapture.Capture) isolation.Result {
	var status struct {
		Status struct {
			Binding        isolation.Binding `json:"isolation_binding"`
			Isolation      *isolation.Result `json:"gpu_isolation"`
			ManifestSHA256 string            `json:"isolation_binding_manifest_sha256"`
		} `json:"status"`
	}
	if err := json.Unmarshal(cap.Files["capture-status.json"], &status); err != nil {
		return isolation.Evaluate(nil, "", isolation.Binding{})
	}
	if raw := cap.Files[isolation.ManifestAttachment]; len(raw) > 0 {
		m, digest, err := isolation.ReadManifest(bytes.NewReader(raw))
		result := isolation.Evaluate(nil, "", status.Status.Binding)
		if err != nil {
			result.Reasons = []string{err.Error()}
			return result
		}
		if status.Status.ManifestSHA256 != digest {
			result.Reasons = []string{"isolation binding manifest hash mismatch"}
			return result
		}
		result, err = evaluateIsolationManifest(cap, m)
		if err != nil {
			result = isolation.Evaluate(nil, "", status.Status.Binding)
			result.Reasons = []string{err.Error()}
			return result
		}
		if status.Status.Isolation != nil && status.Status.Isolation.SHA256 != "" && status.Status.Isolation.SHA256 != result.SHA256 {
			result.State = "unchecked"
			result.Reasons = []string{"retained observer receipt hash mismatch"}
		}
		return result
	}
	raw := cap.Files[isolation.Attachment]
	if len(raw) == 0 {
		return isolation.Evaluate(nil, "", status.Status.Binding)
	}
	r, digest, err := isolation.Read(bytes.NewReader(raw))
	out := isolation.Evaluate(r, digest, status.Status.Binding)
	if status.Status.Isolation != nil && status.Status.Isolation.SHA256 != "" && status.Status.Isolation.SHA256 != digest {
		out.State = "unchecked"
		out.Reasons = []string{"retained observer receipt hash mismatch"}
		return out
	}
	if err != nil {
		out.Reasons = []string{err.Error()}
	}
	return out
}

func printIsolation(w io.Writer, result isolation.Result) {
	state := result.State
	if state == "clear" {
		state = "sampled clear"
	}
	if state == "" {
		state = "unchecked"
	}
	fmt.Fprintf(w, "GPU isolation: %s\n", state)
	for _, reason := range result.Reasons {
		fmt.Fprintf(w, "  %s\n", reason)
	}
	if result.State == "clear" {
		fmt.Fprintln(w, "  brief activity between samples cannot be ruled out")
	}
}

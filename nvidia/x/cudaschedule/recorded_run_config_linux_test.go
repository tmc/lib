//go:build linux

package cudaschedule

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func recordedTestConfig() recordedConfig {
	digest := strings.Repeat("a", 64)
	return recordedConfig{
		Version: 1, Nonce: digest,
		Controller: recordedProcess{PID: 10, Parent: 2, Group: 10, Start: 42, Executable: digest},
		Directory:  "/owned", Helper: "/owned/helper", HelperDigest: digest,
		Worker: "/owned/worker", WorkerDigest: digest, Args: []string{}, WorkDir: "/owned",
		PTXPath: "/owned/source.ptx", PTXDigest: digest, LockScript: "/owned/gpu-lock", LockDigest: recordedLockDigest,
		LockDirectory: "/actual-lock", Label: "source-reviewed-worker",
		Limits:   recordedLimits{Queue: time.Second, Admission: time.Second, Runtime: time.Second, Cleanup: time.Second, Lease: 3 * time.Second},
		Expected: time.Second, OutputLimit: 1024,
	}
}

func TestRecordedConfig(t *testing.T) {
	valid := recordedTestConfig()
	body, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := readRecordedConfig(body); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(*recordedConfig)
	}{
		{"lock bypass", func(c *recordedConfig) { c.LockDigest = strings.Repeat("b", 64) }},
		{"raw pid only", func(c *recordedConfig) { c.Controller.Start = 0 }},
		{"foreign worker", func(c *recordedConfig) { c.Worker = "/foreign/worker" }},
		{"relative dir", func(c *recordedConfig) { c.WorkDir = "." }},
		{"NUL argument", func(c *recordedConfig) { c.Args = []string{"\x00"} }},
		{"oversized argument", func(c *recordedConfig) { c.Args = []string{strings.Repeat("x", 4097)} }},
		{"unbounded output", func(c *recordedConfig) { c.OutputLimit = 0 }},
		{"fractional lock queue", func(c *recordedConfig) { c.Limits.Queue = time.Millisecond }},
		{"fractional lock lease", func(c *recordedConfig) { c.Limits.Lease += time.Millisecond }},
		{"no estimate", func(c *recordedConfig) { c.Expected = 0 }},
		{"label newline", func(c *recordedConfig) { c.Label = "claim\nreceipt" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := valid
			tt.change(&c)
			data, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := readRecordedConfig(data); err == nil {
				t.Fatal("accepted invalid configuration")
			}
		})
	}
	for _, bad := range []string{
		strings.Replace(string(body), `"version":1`, `"version":1,"version":1`, 1),
		strings.Replace(string(body), `"version":1`, `"Version":1`, 1),
		strings.Replace(string(body), `"version":1`, `"version":1,"CPUOnly":true`, 1),
		string(body) + `{}`,
		strings.Replace(string(body), `"args":[]`, `"args":null`, 1),
	} {
		if _, err := readRecordedConfig([]byte(bad)); err == nil {
			t.Fatalf("accepted wire alias: %s", bad)
		}
	}
}

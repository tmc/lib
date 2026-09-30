package nativecontrol_test

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestUnifiedMemoryCollection(t *testing.T) {
	bin := controlBinary(t, "-DCUPTI_API_VERSION=28", "-DCUDAPROF_TEST_ALLOW_UM=1")
	for _, tt := range []struct {
		name, result, state string
		count               int
		env                 []string
	}{
		{"supported", "0", "collected", 2, nil},
		{"native refusal", "28", "configuration_refused", 0, nil},
		{"layout refusal", "0", "unsupported_layout", 0, []string{"CUDAPROF_TEST_UM_STRUCT_SIZE=80"}},
		{"enable refusal", "0", "activity_enable_refused", 0, []string{"CUDAPROF_TEST_UM_ENABLE_RESULT=1"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			cmd := exec.Command(bin, "memory")
			cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex, "CUDAPROF_UNIFIED_MEMORY=1", "CUDAPROF_MEMORY_POLICY=1", "CUDAPROF_TEST_UM_RESULT="+tt.result)
			cmd.Env = append(cmd.Env, tt.env...)
			if b, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("control: %v: %s", err, b)
			}
			paths, err := filepath.Glob(filepath.Join(dir, "process-*", "capture.journal"))
			if err != nil || len(paths) != 1 {
				t.Fatalf("journals: %v %v", paths, err)
			}
			data, err := os.ReadFile(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			records, err := timeline.Decode(map[string][]byte{"native": data}, 1000, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if len(records.UnifiedMemory) != tt.count || len(records.Activities) != 1 {
				t.Fatalf("memory/device records = %d/%d", len(records.UnifiedMemory), len(records.Activities))
			}
			if len(records.MemoryPolicies) != 3 || len(records.Calls) != 0 {
				t.Fatalf("policy/call records = %d/%d", len(records.MemoryPolicies), len(records.Calls))
			}
			good, bad, prefetch := records.MemoryPolicies[0], records.MemoryPolicies[1], records.MemoryPolicies[2]
			if good.Value == nil || *good.Value != -1 || !good.ResultKnown || good.Result != 0 || bad.Value != nil || bad.Result != 1 || prefetch.Event != "prefetch_request" || prefetch.Value != nil {
				t.Fatalf("policies = %+v", records.MemoryPolicies)
			}
			if tt.count != 0 {
				migration, fault := records.UnifiedMemory[0], records.UnifiedMemory[1]
				if !migration.Valid || migration.Value != 4096 || !migration.EndValid || migration.End != 150 || !migration.SourceValid || !migration.DestinationValid {
					t.Fatalf("migration = %+v", migration)
				}
				if !fault.Valid || fault.EndValid || fault.End != fault.Start || fault.RawEnd != 1 || fault.Value != 0xfeed || fault.SourceValid || fault.DestinationValid {
					t.Fatalf("cpu fault = %+v", fault)
				}
			}
			decoder := wire.NewDecoder(bytes.NewReader(data), wire.Limits{})
			found := false
			for {
				record, err := decoder.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				if record.Kind != wire.KindCapability {
					continue
				}
				var capability struct {
					UM struct {
						State     string `json:"state"`
						Results   []int  `json:"native_configuration_results"`
						Residency bool   `json:"residency_observed"`
					} `json:"unified_memory"`
				}
				if err := json.Unmarshal(record.Payload, &capability); err != nil {
					t.Fatal(err)
				}
				if capability.UM.State != tt.state || capability.UM.Residency || len(capability.UM.Results) != 4 {
					t.Fatalf("capability = %+v", capability)
				}
				found = true
			}
			if !found {
				t.Fatal("missing capability")
			}
		})
	}
}

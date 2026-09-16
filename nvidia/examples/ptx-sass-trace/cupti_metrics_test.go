package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// stubCUPTI replaces the CUPTI seams with succeeding stubs for one test and
// restores them afterwards, so the diagnostics can be driven on a host with no
// GPU. ordinal is the device index every stub expects to be handed.
func stubCUPTI(t *testing.T, ordinal uint32) {
	t.Helper()
	oldIndex, oldChip := deviceIndex, chipName
	oldMetrics, oldAvail, oldOS := supportedMetrics, counterAvailability, hostOS
	t.Cleanup(func() {
		deviceIndex, chipName = oldIndex, oldChip
		supportedMetrics, counterAvailability, hostOS = oldMetrics, oldAvail, oldOS
	})
	deviceIndex = func(*cuda.Device) (uint32, error) { return ordinal, nil }
	chipName = func(uint32) (string, error) { return "GH100", nil }
	supportedMetrics = func(string) ([]cuptisass.Metric, error) { return nil, nil }
	counterAvailability = func(uint32) ([]byte, error) { return nil, nil }
	hostOS = "linux"
}

func TestSassMetricsCensus(t *testing.T) {
	dis := &cudasass.Disassembly{
		Functions: []*cudasass.Function{
			{
				Name: "ds4_q8_0_matvec",
				Instructions: []cudasass.Instruction{
					{Address: 0x00, Opcode: "MOV"},
					{Address: 0x10, Opcode: "LDG.E"},
					{Address: 0x20, Opcode: "SHFL.BFLY"},
					{Address: 0x30, Opcode: "RET"},
				},
			},
			{
				Name: "ds4_iq2_simd",
				Instructions: []cudasass.Instruction{
					{Address: 0x00, Opcode: "IDP4A"},
					{Address: 0x10, Opcode: "RET"},
				},
			},
		},
	}

	// The census is static, so it must survive a run that has no device at all
	// and must never be published as if a collector had produced it.
	res, data, err := collectSassMetrics(nil, dis)
	if err != nil {
		t.Fatalf("collectSassMetrics: %v", err)
	}
	if res.Census["ds4_q8_0_matvec"] != 4 {
		t.Errorf("expected census 4 for ds4_q8_0_matvec, got %d", res.Census["ds4_q8_0_matvec"])
	}
	if res.Census["ds4_iq2_simd"] != 2 {
		t.Errorf("expected census 2 for ds4_iq2_simd, got %d", res.Census["ds4_iq2_simd"])
	}
	if res.SassCollection != collectionNotRun {
		t.Errorf("sass collection = %q, want %q", res.SassCollection, collectionNotRun)
	}

	var parsed sassMetricsResult
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal json data: %v", err)
	}
	if parsed.Status == "" {
		t.Error("expected non-empty status in json output")
	}
	// The JSON keys carry the distinction to anyone reading the bundle
	// without the Go types in front of them.
	for _, key := range []string{"static_instruction_census", "sass_collection", "discovery_status"} {
		if !strings.Contains(string(data), key) {
			t.Errorf("sass metrics json missing key %q:\n%s", key, data)
		}
	}
}

func TestPMSamplingDiagnostics(t *testing.T) {
	res, data, err := collectPMSampling(nil)
	if err != nil {
		t.Fatalf("collectPMSampling: %v", err)
	}
	if res == nil || len(data) == 0 {
		t.Fatal("expected non-nil result and non-empty json")
	}
	if res.Sampling != collectionNotRun {
		t.Errorf("pm sampling = %q, want %q", res.Sampling, collectionNotRun)
	}
	for _, key := range []string{"pm_sampling", "availability_status", "availability_size_bytes"} {
		if !strings.Contains(string(data), key) {
			t.Errorf("pm sampling json missing key %q:\n%s", key, data)
		}
	}
}

func TestCUPTIDeviceIndexRejectsMissingDevice(t *testing.T) {
	if _, err := cuptiDeviceIndex(nil); err == nil {
		t.Fatal("cuptiDeviceIndex(nil) = nil error, want error")
	}
}

// TestDiagnosticsUseSuppliedDevice pins the defect these diagnostics had: both
// queried device ordinal 0 whatever device they were handed.
func TestDiagnosticsUseSuppliedDevice(t *testing.T) {
	const ordinal = 3
	stubCUPTI(t, ordinal)

	var chipIndex, availIndex uint32 = 99, 99
	chipName = func(index uint32) (string, error) {
		chipIndex = index
		return "GH100", nil
	}
	counterAvailability = func(index uint32) ([]byte, error) {
		availIndex = index
		return make([]byte, 16), nil
	}

	sass, _, err := collectSassMetrics(nil, nil)
	if err != nil {
		t.Fatalf("collectSassMetrics: %v", err)
	}
	if chipIndex != ordinal {
		t.Errorf("chip name queried for device %d, want %d", chipIndex, ordinal)
	}
	if sass.DeviceIndex != ordinal {
		t.Errorf("sass result device index = %d, want %d", sass.DeviceIndex, ordinal)
	}

	pm, _, err := collectPMSampling(nil)
	if err != nil {
		t.Fatalf("collectPMSampling: %v", err)
	}
	if availIndex != ordinal {
		t.Errorf("counter availability queried for device %d, want %d", availIndex, ordinal)
	}
	if pm.DeviceIndex != ordinal {
		t.Errorf("pm result device index = %d, want %d", pm.DeviceIndex, ordinal)
	}
}

// TestStatusNeverClaimsCollection checks the reported vocabulary against what
// each run actually established. Discovery and a static census must never be
// reportable as collected hardware data.
func TestStatusNeverClaimsCollection(t *testing.T) {
	tests := []struct {
		name     string
		noDevice bool
		notLinux bool
		chip     func(uint32) (string, error)
		metrics  func(string) ([]cuptisass.Metric, error)
		avail    func(uint32) ([]byte, error)
		wantSass string
		wantPM   string
	}{
		{
			name:     "no device",
			noDevice: true,
			wantSass: statusNoDevice,
			wantPM:   statusNoDevice,
		},
		{
			name:     "non linux host",
			notLinux: true,
			wantSass: statusSkippedNonLinux,
			wantPM:   statusSkippedNonLinux,
		},
		{
			name:     "query failed",
			chip:     func(uint32) (string, error) { return "", errors.New("chip query failed") },
			avail:    func(uint32) ([]byte, error) { return nil, errors.New("no availability") },
			wantSass: statusChipQueryFailed,
			wantPM:   statusAvailQueryFailed,
		},
		{
			name:     "discovery failed",
			metrics:  func(string) ([]cuptisass.Metric, error) { return nil, errors.New("metric discovery failed") },
			avail:    func(uint32) ([]byte, error) { return make([]byte, 8), nil },
			wantSass: statusDiscoveryFailed,
			wantPM:   statusAvailabilityShown,
		},
		{
			name: "discovery succeeded",
			metrics: func(string) ([]cuptisass.Metric, error) {
				return []cuptisass.Metric{{Name: "sass__inst_executed"}}, nil
			},
			avail:    func(uint32) ([]byte, error) { return make([]byte, 8), nil },
			wantSass: statusMetricsListed,
			wantPM:   statusAvailabilityShown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stubCUPTI(t, 0)
			if tt.noDevice {
				deviceIndex = func(*cuda.Device) (uint32, error) {
					return 0, errors.New("no device supplied")
				}
			}
			if tt.notLinux {
				hostOS = "darwin"
			}
			if tt.chip != nil {
				chipName = tt.chip
			}
			if tt.metrics != nil {
				supportedMetrics = tt.metrics
			}
			if tt.avail != nil {
				counterAvailability = tt.avail
			}

			sass, sassJSON, err := collectSassMetrics(nil, nil)
			if err != nil {
				t.Fatalf("collectSassMetrics: %v", err)
			}
			if sass.Status != tt.wantSass {
				t.Errorf("sass status = %q, want %q", sass.Status, tt.wantSass)
			}
			if sass.SassCollection != collectionNotRun {
				t.Errorf("sass collection = %q, want %q", sass.SassCollection, collectionNotRun)
			}

			pm, pmJSON, err := collectPMSampling(nil)
			if err != nil {
				t.Fatalf("collectPMSampling: %v", err)
			}
			if pm.Status != tt.wantPM {
				t.Errorf("pm status = %q, want %q", pm.Status, tt.wantPM)
			}
			if pm.Sampling != collectionNotRun {
				t.Errorf("pm sampling = %q, want %q", pm.Sampling, collectionNotRun)
			}

			// A status a reader could take as "we sampled" or "we collected"
			// is the whole defect, so reject that spelling outright.
			for _, s := range []string{sass.Status, pm.Status} {
				for _, banned := range []string{"active", "collected", "sampled"} {
					if strings.Contains(s, banned) {
						t.Errorf("status %q contains %q and overclaims", s, banned)
					}
				}
			}
			for _, b := range [][]byte{sassJSON, pmJSON} {
				if !json.Valid(b) {
					t.Errorf("diagnostic json is invalid: %s", b)
				}
			}
		})
	}
}

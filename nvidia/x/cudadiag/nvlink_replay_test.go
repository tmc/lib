package cudadiag_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// Every device method here is a host control. This does not qualify NVML,
// physical endpoints, library ABI or a transport-cause diagnosis.
type nvlinkReplayDevice struct{ clock *interchangeClock }

func (*nvlinkReplayDevice) UUID() (string, error) { return "GPU-protocol", nil }
func (*nvlinkReplayDevice) NvLinkState(uint32) (nvml.EnableState, error) {
	return nvml.FeatureEnabled, nil
}
func (*nvlinkReplayDevice) NvLinkVersion(uint32) (uint32, error) { return 4, nil }
func (*nvlinkReplayDevice) NvLinkRemoteDeviceType(_ uint32, typ *nvml.IntNvLinkDeviceType) error {
	*typ = nvml.NvlinkDeviceTypeGpu
	return nil
}
func (*nvlinkReplayDevice) NvLinkRemotePciInfo(uint32) (nvml.PciInfo, error) {
	var pci nvml.PciInfo
	copy(pci.BusId[:], "00000000:03:00.0")
	pci.Bus = 3
	return pci, nil
}
func (d *nvlinkReplayDevice) NvLinkErrorCounter(_ uint32, counter nvml.NvLinkErrorCounter) (uint64, error) {
	d.clock.now++
	switch counter {
	case nvml.NvlinkErrorDlReplay:
		return 1<<53 + 1, nil
	case nvml.NvlinkErrorDlRecovery:
		return 0, nil
	default:
		return 999, nvml.ErrNotSupported
	}
}

func nvlinkReplayFixture(t *testing.T) ([]nvtelemetry.Reading, cudadiag.TelemetryOptions, []byte) {
	t.Helper()
	readings, opt := mixedProtocol()
	clock := &interchangeClock{now: 100000}
	config := nvtelemetry.NVLinkCounterConfig{Topology: nvtelemetry.NVLinkConfig{NodeID: "protocol-node", SourceID: "nvlink-host-control", NVMLVersion: "fake-NVML-1", TopologyEpoch: "topology-1", Links: []uint32{0}, Peers: []nvtelemetry.NVLinkPeer{{BusID: "00000000:03:00.0", UUID: "GPU-peer"}}, ValidFromNanos: 0, ValidUntilNanos: 1000000}, Counters: []nvml.NvLinkErrorCounter{nvml.NvlinkErrorDlReplay, nvml.NvlinkErrorDlRecovery, nvml.NvlinkErrorDlCrcFlit}}
	r, err := nvtelemetry.NewNVLinkCounterReader(&nvlinkReplayDevice{clock: clock}, clock, config)
	if err != nil {
		t.Fatal(err)
	}
	s, err := r.Read(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	counters, err := nvtelemetry.NVLinkCounterReadings(raw, "synthetic")
	if err != nil {
		t.Fatal(err)
	}
	transform := []byte("synthetic host NVLink raw-count identity transformation control v1")
	transformHash := fmt.Sprintf("%x", sha256.Sum256(transform))
	rawHash := fmt.Sprintf("%x", sha256.Sum256(raw))
	opt.RawArtifacts[rawHash] = append([]byte(nil), raw...)
	opt.RawArtifacts[transformHash] = transform
	for i := range counters {
		o := counters[i].Observation
		o.Assignment = &opt.Assignment
		o.Workload = &opt.Workload
		o.OpID = opt.OpID
		o.Observer = opt.Observer
		o.TransformSHA256 = transformHash
		uncertainty := int64(10)
		o.UncertaintyNS = &uncertainty
		if o.CounterEpoch != "" {
			t.Fatal("reader invented counter epoch")
		}
	}
	opt.ExpectedSources[rawHash] = *counters[0].Observation
	return append(readings, counters...), opt, raw
}

func TestNVLinkCounterReplayDiagnosis(t *testing.T) {
	readings, opt, _ := nvlinkReplayFixture(t)
	stream := collectMixed(t, readings, opt) // real canonical Writer through collector
	_, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(stream), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	last := samples[0].Readings[len(samples[0].Readings)-3:]
	if last[0].Number.Uint == nil || *last[0].Number.Uint != 1<<53+1 || last[0].Number.Float != nil {
		t.Fatal("strict replay rounded raw integer")
	}
	if last[1].Number.Uint == nil || *last[1].Number.Uint != 0 || !last[1].IsAvailable() {
		t.Fatal("observed zero lost")
	}
	if last[2].Number != nil || last[2].IsAvailable() {
		t.Fatal("unsupported counter became zero")
	}
	out, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(stream), opt)
	if err != nil {
		t.Fatal(err)
	}
	var seen int
	for _, observation := range out.Observations {
		if !strings.HasPrefix(observation.Reading.Name, "nvlink_dl_") {
			continue
		}
		seen++
		if observation.Finding.Admission.Admitted || observation.Finding.Admission.Status == "admitted" || observation.Finding.Value != nil {
			t.Fatalf("unknown reset epoch or lossy value admitted %+v", observation)
		}
		if observation.Reading.IsAvailable() {
			if observation.Finding.Admission.Reason != "missing counter epoch" {
				t.Fatalf("positive evidence links failed before epoch rule: %s", observation.Finding.Admission.Reason)
			}
		} else if !strings.Contains(observation.Finding.Admission.Reason, "unavailable reading") {
			t.Fatalf("unavailable status lost %+v", observation)
		}
	}
	if seen != 3 {
		t.Fatalf("raw findings missing: %d", seen)
	}
	if strings.Contains(out.Diagnosis.PrimaryFinding.Rule, "nvlink") || strings.Contains(out.Diagnosis.PrimaryFinding.Rule, "congestion") {
		t.Fatal("raw count fabricated transport cause")
	}
}

func TestNVLinkCounterReplayArtifactIdentity(t *testing.T) {
	for _, change := range []string{"uuid", "peer", "topology"} {
		t.Run(change, func(t *testing.T) {
			readings, opt, raw := nvlinkReplayFixture(t)
			stream := collectMixed(t, readings, opt)
			var snapshot nvtelemetry.NVLinkCounterSnapshot
			if err := json.Unmarshal(raw, &snapshot); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "uuid":
				snapshot.Before.DeviceUUID = "GPU-other"
			case "peer":
				snapshot.Before.Links[0].PeerUUID = "GPU-other"
			case "topology":
				snapshot.Before.TopologyEpoch = "other"
			}
			changed, err := json.Marshal(snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := nvtelemetry.NVLinkCounterReadings(changed, "synthetic"); err == nil {
				t.Fatal("inconsistent endpoint raw evidence admitted")
			}
			originalHash := fmt.Sprintf("%x", sha256.Sum256(raw))
			opt.RawArtifacts[originalHash] = changed
			out, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(stream), opt)
			if err != nil {
				t.Fatal(err)
			}
			for _, observation := range out.Observations {
				if observation.Reading.Name != nvtelemetry.MetricNVLinkDLReplayErrors {
					continue
				}
				if observation.Finding.Admission.Admitted || observation.Finding.Admission.Reason != "missing or corrupt raw artifact" {
					t.Fatalf("changed independent artifact admitted %+v", observation)
				}
			}
		})
	}
}

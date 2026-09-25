package advise

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

// Callback IDs from cupti/enums.gen.go (CUPTI cupti_driver_cbid.h and
// cupti_runtime_cbid.h).
const (
	cbidInit                    = 1   // cuInit
	cbidMemAllocV2              = 243 // cuMemAlloc_v2
	cbidMemcpyHtoDV2            = 276 // cuMemcpyHtoD_v2
	cbidMemcpyHtoDAsyncV2       = 277 // cuMemcpyHtoDAsync_v2
	cbidMemcpyDtoDV2            = 280 // cuMemcpyDtoD_v2
	cbidGraphLaunch             = 514 // cuGraphLaunch
	cbidGraphInstantiateWithFlg = 643 // cuGraphInstantiateWithFlags
	cbidRuntimeMemcpy           = 31  // cudaMemcpy_v3020
)

// hostAPI appends one API record with the fields the host-cost audit reads.
func hostAPI(b *journaltest.Builder, domain string, cbid uint32, start, end uint64, thread, correlation uint32) *journaltest.Builder {
	b.Append(wire.KindAPI, map[string]any{
		"kind": "api", "domain": domain, "callback_id": cbid, "start": start, "end": end,
		"process_id": 4242, "thread_id": thread, "correlation": correlation,
	})
	return b
}

// memcpy appends one copy activity naming the driver and runtime correlation
// of the call that issued it.
func memcpy(b *journaltest.Builder, start, end, bytes uint64, kind uint8, stream, correlation, runtimeCorrelation uint32) *journaltest.Builder {
	b.Append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "memcpy", "valid": true, "start": start, "end": end,
		"bytes": bytes, "copy_kind": kind, "device": 0, "stream": stream,
		"correlation": correlation, "runtime_correlation": runtimeCorrelation,
	})
	return b
}

// loop writes n iterations of a 1 ms step: a kernel, then whatever body adds
// for iteration i starting at the given host time.
func loop(n int, body func(b *journaltest.Builder, i int, at uint64)) []byte {
	b := journaltest.New()
	for i := range n {
		at := uint64(1_000_000 + i*1_000_000)
		corr := uint32(1000 + i)
		b.Kernel(at+100_000, at+200_000, corr)
		b.API(at, at+10_000, corr)
		body(b, i, at+300_000)
	}
	return b.Finalize()
}

func hostCostItems(t *testing.T, journal []byte) []Item {
	t.Helper()
	report := &Report{}
	auditHostAPICost(testCapture(map[string][]byte{"capture.journal": journal}, journaltest.CompleteStatus()), report)
	return report.Items
}

func TestAuditHostAPICost(t *testing.T) {
	const ms = 1_000_000
	tests := []struct {
		name    string
		journal []byte
		want    []string // one substring per expected finding, in order
		host    []int64
		sev     []Severity
	}{
		{
			// The planted control: a synchronous 8 KiB upload every step.
			name: "planted per-iteration sync copy",
			journal: loop(5, func(b *journaltest.Builder, i int, at uint64) {
				corr := uint32(1 + i)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+ms, 1, corr)
				memcpy(b, at+100, at+2_100, 8192, 1, 7, corr, 0)
			}),
			want: []string{"Synchronous copy cuMemcpyHtoD_v2 repeats 5 times (8192 B host-to-device on stream 7)"},
			host: []int64{5 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			// The quiet control: the same upload issued async on stream 7.
			name: "async copy on a non-default stream is quiet",
			journal: loop(5, func(b *journaltest.Builder, i int, at uint64) {
				corr := uint32(1 + i)
				hostAPI(b, "driver", cbidMemcpyHtoDAsyncV2, at, at+5_000, 1, corr)
				memcpy(b, at+100, at+2_100, 8192, 1, 7, corr, 0)
			}),
		},
		{
			name: "device-to-device sync copy is quiet",
			journal: loop(5, func(b *journaltest.Builder, i int, at uint64) {
				corr := uint32(1 + i)
				hostAPI(b, "driver", cbidMemcpyDtoDV2, at, at+ms, 1, corr)
				memcpy(b, at+100, at+2_100, 8192, 8, 0, corr, 0)
			}),
		},
		{
			name: "one sync copy is not per-iteration",
			journal: loop(1, func(b *journaltest.Builder, i int, at uint64) {
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+ms, 1, 1)
				memcpy(b, at+100, at+2_100, 8192, 1, 0, 1, 0)
			}),
		},
		{
			// A per-iteration upload whose size grows each step is still one
			// repeated copy.
			name: "sync copy of varying size",
			journal: loop(4, func(b *journaltest.Builder, i int, at uint64) {
				corr := uint32(1 + i)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+ms, 1, corr)
				memcpy(b, at+100, at+2_100, uint64(1024<<i), 1, 7, corr, 0)
			}),
			want: []string{"cuMemcpyHtoD_v2 repeats 4 times (1024-8192 B host-to-device on stream 7)"},
			host: []int64{4 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			// One runtime copy the driver split in two reports both halves.
			name: "runtime copy issuing two driver copies",
			journal: loop(2, func(b *journaltest.Builder, i int, at uint64) {
				rt, d1, d2 := uint32(100+i), uint32(200+i), uint32(300+i)
				hostAPI(b, "runtime", cbidRuntimeMemcpy, at, at+ms, 1, rt)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at+1_000, at+2_000, 1, d1)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at+3_000, at+4_000, 1, d2)
				memcpy(b, at+1_500, at+2_500, 4096, 1, 7, d1, 0)
				memcpy(b, at+3_500, at+4_500, 4096, 1, 7, d2, 0)
			}),
			want: []string{"Synchronous copy cudaMemcpy repeats 2 times (8192 B host-to-device on stream 7)"},
			host: []int64{2 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			// A runtime-only capture: the copy's correlation names the
			// runtime call itself, as on the ds4 C prefill capture.
			name: "runtime call joined by its own correlation",
			journal: loop(2, func(b *journaltest.Builder, i int, at uint64) {
				rt := uint32(100 + i)
				hostAPI(b, "runtime", cbidRuntimeMemcpy, at, at+ms, 1, rt)
				memcpy(b, at+2_000, at+4_000, 4096, 2, 0, rt, 0)
			}),
			want: []string{"Synchronous copy cudaMemcpy repeats 2 times (4096 B device-to-host on stream 0)"},
			host: []int64{2 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			// A runtime copy on thread 1 does not swallow a driver allocation
			// it encloses: containment is only within one class.
			name: "runtime call does not enclose another class",
			journal: loop(2, func(b *journaltest.Builder, i int, at uint64) {
				rt, drv := uint32(100+i), uint32(200+i)
				hostAPI(b, "runtime", cbidRuntimeMemcpy, at, at+ms, 1, rt)
				hostAPI(b, "driver", cbidMemAllocV2, at+1_000, at+ms-1_000, 1, drv)
			}),
			want: []string{"Synchronous copy cudaMemcpy repeats 2 times", "Allocation cuMemAlloc_v2 repeats 2 times"},
			host: []int64{2 * ms, 2 * (ms - 2_000)},
			sev:  []Severity{SeverityCost, SeverityCost},
		},
		{
			name: "allocations and instantiates per replay",
			journal: loop(4, func(b *journaltest.Builder, i int, at uint64) {
				c := uint32(10 * (i + 1))
				hostAPI(b, "driver", cbidMemAllocV2, at, at+2*ms/10, 1, c)
				hostAPI(b, "driver", cbidGraphInstantiateWithFlg, at+300_000, at+600_000, 1, c+1)
				hostAPI(b, "driver", cbidGraphLaunch, at+600_000, at+610_000, 1, c+2)
			}),
			want: []string{"Graph instantiated 4 times for 4 graph launches", "Allocation cuMemAlloc_v2 repeats 4 times"},
			host: []int64{4 * 300_000, 4 * 200_000},
			sev:  []Severity{SeverityCost, SeverityCost},
		},
		{
			// One instantiate amortized over ten replays is the intended use.
			name: "instantiate once, replay many is quiet",
			journal: loop(10, func(b *journaltest.Builder, i int, at uint64) {
				c := uint32(10 * (i + 1))
				if i == 0 {
					hostAPI(b, "driver", cbidGraphInstantiateWithFlg, at, at+300_000, 1, c)
				}
				hostAPI(b, "driver", cbidGraphLaunch, at+300_000, at+310_000, 1, c+1)
			}),
		},
		{
			// cudaMemcpy makes the cuMemcpyHtoD_v2 it encloses: one copy, and
			// its activity names the runtime call's correlation.
			name: "runtime call enclosing its driver call counts once",
			journal: loop(3, func(b *journaltest.Builder, i int, at uint64) {
				rt, drv := uint32(100+i), uint32(200+i)
				hostAPI(b, "runtime", cbidRuntimeMemcpy, at, at+ms, 1, rt)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at+1_000, at+ms-1_000, 1, drv)
				memcpy(b, at+2_000, at+4_000, 4096, 1, 0, drv, rt)
			}),
			want: []string{"Synchronous copy cudaMemcpy repeats 3 times (4096 B host-to-device on stream 0)"},
			host: []int64{3 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			name: "copy with no activity is UNREAD, not zero",
			journal: loop(3, func(b *journaltest.Builder, i int, at uint64) {
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+ms, 1, uint32(1+i))
			}),
			want: []string{"bytes, direction and stream UNREAD"},
			host: []int64{3 * ms},
			sev:  []Severity{SeverityCost},
		},
		{
			// 3 us of copies in a 100 ms host API window: small in host time
			// and in share, so it ranks as context.
			name: "negligible copies demote to context",
			journal: loop(3, func(b *journaltest.Builder, i int, at uint64) {
				corr := uint32(1 + i)
				hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+1_000, 1, corr)
				memcpy(b, at+100, at+200, 64, 1, 0, corr, 0)
				if i == 2 {
					hostAPI(b, "driver", cbidInit, at+100*ms, at+100*ms+1_000, 1, 99)
				}
			}),
			want: []string{"cuMemcpyHtoD_v2 repeats 3 times"},
			host: []int64{3_000},
			sev:  []Severity{SeverityContext},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			items := hostCostItems(t, tc.journal)
			if len(items) != len(tc.want) {
				for _, it := range items {
					t.Logf("finding: %s", it.Finding)
				}
				t.Fatalf("got %d findings, want %d", len(items), len(tc.want))
			}
			for i, it := range items {
				if !strings.Contains(it.Finding, tc.want[i]) {
					t.Errorf("finding %d = %q, want it to contain %q", i, it.Finding, tc.want[i])
				}
				if it.HostTimeNanos != tc.host[i] {
					t.Errorf("finding %d HostTimeNanos = %d, want %d", i, it.HostTimeNanos, tc.host[i])
				}
				if it.DeviceTimeNanos != 0 {
					t.Errorf("finding %d DeviceTimeNanos = %d, want 0: host time is not device time", i, it.DeviceTimeNanos)
				}
				if it.Severity != tc.sev[i] {
					t.Errorf("finding %d severity = %s, want %s", i, it.Severity, tc.sev[i])
				}
				if !strings.HasPrefix(it.Location.CallSite, "UNREAD") {
					t.Errorf("finding %d call site = %q, want UNREAD", i, it.Location.CallSite)
				}
			}
		})
	}
}

// TestAuditHostAPICostWithoutAPIRecords: a capture with no journal, or a
// journal with no API records, reports nothing rather than zero host time.
func TestAuditHostAPICostWithoutAPIRecords(t *testing.T) {
	for name, files := range map[string]map[string][]byte{
		"no journal":     nil,
		"no api records": {"capture.journal": journaltest.New().Kernel(1_000, 2_000, 1).Finalize()},
	} {
		report := &Report{}
		auditHostAPICost(testCapture(files, journaltest.CompleteStatus()), report)
		if len(report.Items) != 0 {
			t.Errorf("%s: got %d findings, want none", name, len(report.Items))
		}
	}
}

// TestHostCostRanksByHostTime: a host finding competes on its host time. It
// sorts above a smaller device finding of the same tier, and the device
// demotion does not touch it.
func TestHostCostRanksByHostTime(t *testing.T) {
	report := &Report{Items: []Item{
		{Severity: SeverityCost, Finding: "device", DeviceTimeNanos: 1_000_000, ShareOfProfile: 50},
		{Severity: SeverityCost, Finding: "host", HostTimeNanos: 5_000_000},
		{Severity: SeverityCost, Finding: "small host", HostTimeNanos: 100_000},
	}}
	finalizeAndRankFindings(report, nil, 2_000_000)
	var got []string
	for _, it := range report.Items {
		got = append(got, it.Finding+":"+string(it.Severity))
	}
	want := []string{"host:Cost", "device:Cost", "small host:Cost"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("ranked %v, want %v", got, want)
	}
}

// TestAnalyzeCaptureReportsHostCost runs the audit through Analyze and
// the Markdown renderer: the header states host time, not device time, and
// the location states the call site is UNREAD.
func TestAnalyzeCaptureReportsHostCost(t *testing.T) {
	journal := loop(5, func(b *journaltest.Builder, i int, at uint64) {
		corr := uint32(1 + i)
		hostAPI(b, "driver", cbidMemcpyHtoDV2, at, at+1_000_000, 1, corr)
		memcpy(b, at+100, at+2_100, 8192, 1, 7, corr, 0)
	})
	report, err := Analyze(testCapture(map[string][]byte{"capture.journal": journal}, journaltest.CompleteStatus()))
	if err != nil {
		t.Fatal(err)
	}
	md := report.MarkdownWithLimit(0)
	for _, want := range []string{
		"(host 5.00 ms over 5 calls, ",
		"device copy time 0.01 ms) Dispatch: Synchronous copy cuMemcpyHtoD_v2 repeats 5 times",
		"- **Location**: call site UNREAD: the journal records API calls without a host stack",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown missing %q:\n%s", want, md)
		}
	}
}

// TestHostAPIName checks the names and classes read from callback IDs,
// including the graph-node calls whose names contain Memcpy or MemAlloc but
// which neither copy nor allocate.
func TestHostAPIName(t *testing.T) {
	tests := []struct {
		domain string
		cbid   uint32
		name   string
		class  hostAPIClass
	}{
		{"driver", cbidMemcpyHtoDV2, "cuMemcpyHtoD_v2", hostAPISyncCopy},
		{"driver", cbidMemcpyHtoDAsyncV2, "cuMemcpyHtoDAsync_v2", hostAPIOther},
		{"driver", 397, "cuMemcpyHtoD_v2_ptds", hostAPISyncCopy},
		{"driver", cbidMemAllocV2, "cuMemAlloc_v2", hostAPIAlloc},
		{"driver", cbidGraphInstantiateWithFlg, "cuGraphInstantiateWithFlags", hostAPIInstantiate},
		{"driver", cbidGraphLaunch, "cuGraphLaunch", hostAPIGraphLaunch},
		{"runtime", cbidRuntimeMemcpy, "cudaMemcpy", hostAPISyncCopy},
		{"runtime", 41, "cudaMemcpyAsync", hostAPIOther},
		{"runtime", 20, "cudaMalloc", hostAPIAlloc},
		{"driver", cbidInit, "cuInit", hostAPIOther},
		{"nvtx", 1, "", hostAPIOther},
	}
	for _, tc := range tests {
		name, class := hostAPIName(tc.domain, tc.cbid)
		if name != tc.name || class != tc.class {
			t.Errorf("hostAPIName(%s, %d) = %q, %d, want %q, %d", tc.domain, tc.cbid, name, class, tc.name, tc.class)
		}
	}
	// No graph call other than instantiate and launch is a copy or an
	// allocation, whatever its name contains.
	for _, domain := range []string{"driver", "runtime"} {
		for id := uint32(0); id < 1200; id++ {
			name, class := hostAPIName(domain, id)
			if strings.Contains(name, "Graph") && (class == hostAPISyncCopy || class == hostAPIAlloc) {
				t.Errorf("hostAPIName(%s, %d) = %q classified %d, a graph call", domain, id, name, class)
			}
		}
	}
}

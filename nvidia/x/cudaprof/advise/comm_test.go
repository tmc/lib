package advise

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestCommKind(t *testing.T) {
	tests := []struct {
		ev   timeline.Event
		want string
	}{
		{timeline.Event{Subtype: "kernel", Name: "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"}, "NCCL"},
		{timeline.Event{Subtype: "kernel", Name: "ncclKernel_AllGather_RING_LL_Sum_int8_t"}, "NCCL"},
		{timeline.Event{Subtype: "kernel", Name: "ncclDevKernel_AllReduce_Sum_bf16_NVLS_SIMPLE(ncclDevKernelArgsStorage<4096ul>)"}, "NCCL"},
		{timeline.Event{Subtype: "kernel", Name: "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"}, "NCCL"},
		{timeline.Event{Subtype: "kernel", Name: "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)"}, "NCCL"},
		{timeline.Event{Subtype: "kernel", Name: "nvshmemi_proxy_progress_kernel"}, "NVSHMEM"},
		{timeline.Event{Subtype: "kernel", Name: "rankgroup_allreduce_f32"}, "collective"},
		{timeline.Event{Subtype: "kernel", Name: "nvshmemi_transfer_rma_kernel"}, "NVSHMEM"},
		{timeline.Event{Subtype: "kernel", Name: "multimem_collectives"}, "NVLink multicast"},
		{timeline.Event{Subtype: "kernel", Name: "allreduce_residual_rmsnorm"}, "collective"},
		{timeline.Event{Subtype: "kernel", Name: "eg_reduce_scatter_bf16"}, "collective"},
		{timeline.Event{Subtype: "memcpy", Name: "memcpy", CopyKind: uint8(cupti.ActivityMemcpyKindPtop)}, "peer copy"},
		{timeline.Event{Subtype: "memcpy", Name: "memcpy", CopyKind: uint8(cupti.ActivityMemcpyKindDtod)}, ""},
		{timeline.Event{Subtype: "kernel", Name: "gemm_bf16_tn"}, ""},
		{timeline.Event{Subtype: "kernel", Name: "rmsnorm_reduce"}, ""},
		{timeline.Event{Subtype: "kernel", Name: "ds4_moe_expert_dispatch"}, ""},
	}
	for _, tt := range tests {
		if got := commKind(tt.ev); got != tt.want {
			t.Errorf("commKind(%s %q) = %q, want %q", tt.ev.Subtype, tt.ev.Name, got, tt.want)
		}
	}
}

func TestSummarizeCommunication(t *testing.T) {
	// Stream 1 computes 0-100 and 150-200. Stream 2 all-reduces 50-180,
	// so 100-150 is exposed. Stream 1 then idles 200-250 and all-reduces
	// 250-300 with nothing else running.
	streams := map[uint32][]timeline.Event{
		1: {
			{Subtype: "kernel", Name: "gemm", Start: 0, End: 100},
			{Subtype: "kernel", Name: "gemm", Start: 150, End: 200},
			{Subtype: "kernel", Name: "ncclDevKernel_AllReduce", Start: 250, End: 300},
		},
		2: {
			{Subtype: "kernel", Name: "ncclDevKernel_AllReduce", Start: 50, End: 180},
		},
	}
	s, ok := summarizeCommunication(streams)
	if !ok {
		t.Fatal("no communication found")
	}
	want := commSummary{span: 300, comm: 180, compute: 150, exposed: 100, idleBeforeComm: 50}
	if s.span != want.span || s.comm != want.comm || s.compute != want.compute || s.exposed != want.exposed || s.idleBeforeComm != want.idleBeforeComm {
		t.Errorf("summary = span %d comm %d compute %d exposed %d idleBeforeComm %d; want %+v", s.span, s.comm, s.compute, s.exposed, s.idleBeforeComm, want)
	}
	if len(s.kernels) != 1 || s.kernels[0].count != 2 || s.kernels[0].exposed != 100 {
		t.Errorf("kernels = %+v, want one kernel, 2 launches, 100 exposed", s.kernels)
	}
	if _, ok := summarizeCommunication(map[uint32][]timeline.Event{1: {{Subtype: "kernel", Name: "gemm", Start: 0, End: 10}}}); ok {
		t.Error("compute-only timeline reported communication")
	}
}

// commCapture is a capture of one rank: a GEMM, then an exposed all-reduce
// after an idle gap, repeated.
func commCapture() *cudacapture.Capture {
	var samples []*profile.Sample
	start := uint64(1_000_000)
	for i := 0; i < 4; i++ {
		samples = append(samples,
			microKernelSample("gemm_bf16_tn", 7, start, start+300_000, 2*i+1),
			microKernelSample("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 7, start+1_400_000, start+2_000_000, 2*i+2))
		start += 2_000_000
	}
	return &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     samples,
	}}
}

func TestAuditCommunication(t *testing.T) {
	report, err := Analyze(commCapture())
	if err != nil {
		t.Fatal(err)
	}
	var exposed, idle *Item
	for i := range report.Items {
		it := &report.Items[i]
		switch {
		case it.Category == CategoryCommunication:
			exposed = it
		case strings.HasPrefix(it.Finding, "The device was idle"):
			idle = it
		}
		if strings.Contains(it.Finding, "ncclDevKernel") && it.Category != CategoryCommunication {
			t.Errorf("communication kernel reported as %s: %s", it.Category, it.Finding)
		}
	}
	if exposed == nil {
		t.Fatalf("no exposed communication finding; items:\n%s", report.Markdown())
	}
	if exposed.DeviceTimeNanos != 2_400_000 || exposed.EvidenceLevel != EvidenceObserved {
		t.Errorf("exposed = %d ns, %s; want 2400000 ns observed", exposed.DeviceTimeNanos, exposed.EvidenceLevel)
	}
	if idle == nil {
		t.Fatalf("no idle finding; items:\n%s", report.Markdown())
	}
	if !strings.Contains(idle.Finding, "peer wait, not an unexplained stall") {
		t.Errorf("idle finding does not separate peer wait:\n%s", idle.Finding)
	}
}

package cudabpf_test

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabpf"
)

func schedObject() cudabpf.ObjectSpec {
	return cudabpf.ObjectSpec{
		Name: "gpu_sched_policy.bpf.o",
		Programs: []cudabpf.ProgramSpec{
			{Name: "on_task_init", Section: "struct_ops/on_task_init", Member: "on_task_init"},
			{Name: "on_bind", Section: "struct_ops/on_bind", Member: "on_bind"},
		},
	}
}

func schedSpec() cudabpf.StructOpsSpec {
	return cudabpf.StructOpsSpec{
		Table:   cudabpf.SchedOpsTable,
		MapName: "gpu_sched_policy",
		Programs: map[string]string{
			"on_task_init": "on_task_init",
			"on_bind":      "on_bind",
		},
	}
}

func schedBridge(t *testing.T, k cudabpf.Kfuncs, p cudabpf.SchedPolicy) *cudabpf.Bridge {
	t.Helper()
	br := cudabpf.NewBridge(newFakeLoader(patchedBTF()), k, cudabpf.Config{})
	if _, err := br.Attach(context.Background(), schedObject(), schedSpec()); err != nil {
		t.Fatalf("Attach: %v", err)
	}
	br.SetSchedPolicy(p)
	return br
}

// TestSchedulingAttributeRangeValidation covers the rule that every value is
// checked against the driver's accepted range before the kfunc call, and that
// validation refuses rather than clamps. A tenant asking for more than it may
// have must not silently receive the maximum.
func TestSchedulingAttributeRangeValidation(t *testing.T) {
	limits := cudabpf.DefaultLimits()
	tests := []struct {
		name string
		attr cudabpf.SchedAttributes
		ok   bool
	}{
		{"in range", cudabpf.SchedAttributes{TimesliceUS: 1000, SetTimeslice: true}, true},
		{"at minimum", cudabpf.SchedAttributes{TimesliceUS: limits.MinTimesliceUS, SetTimeslice: true}, true},
		{"at maximum", cudabpf.SchedAttributes{TimesliceUS: limits.MaxTimesliceUS, SetTimeslice: true}, true},
		{"below minimum", cudabpf.SchedAttributes{TimesliceUS: 0, SetTimeslice: true}, false},
		{"above maximum", cudabpf.SchedAttributes{TimesliceUS: limits.MaxTimesliceUS + 1, SetTimeslice: true}, false},
		{"interleave high", cudabpf.SchedAttributes{Interleave: cudabpf.InterleaveHigh, SetInterleave: true}, true},
		{"interleave above the defined levels", cudabpf.SchedAttributes{Interleave: cudabpf.InterleaveLevel(3), SetInterleave: true}, false},
		{"nothing set", cudabpf.SchedAttributes{}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.attr.Validate(limits)
			if (err == nil) != tt.ok {
				t.Fatalf("Validate = %v, want ok=%t", err, tt.ok)
			}
		})
	}

	// Refusal, not clamping: an out-of-range request applies nothing at
	// all, and in particular does not apply the in-range half of a pair.
	k := &fakeKfuncs{}
	br := schedBridge(t, k, &tenantPolicy{byTSG: map[uint64]cudabpf.SchedAttributes{
		1: {
			TimesliceUS: limits.MaxTimesliceUS * 10, SetTimeslice: true,
			Interleave: cudabpf.InterleaveLow, SetInterleave: true,
		},
	}})
	got, out := br.OnTaskInit(&cudabpf.TaskInitContext{TSGID: 1})
	if !out.Fallback {
		t.Fatalf("an out-of-range request was not refused: %+v", out)
	}
	if got.SetTimeslice || got.SetInterleave {
		t.Errorf("attributes applied after a refusal: %+v", got)
	}
	if len(k.timeslices) != 0 || len(k.interleave) != 0 {
		t.Errorf("kfuncs called after a refusal: %v %v", k.timeslices, k.interleave)
	}
	if got.TimesliceUS == limits.MaxTimesliceUS {
		t.Error("the request was clamped to the maximum instead of refused")
	}
}

// TestSchedulingAttributesDifferByTenantBeforeValidation records what a host
// test can and cannot show here.
//
// It shows that the policy layer routes different attribute values to
// different tenants and validates each independently. It does not show that
// the tenants received different shares of the GPU: no host can produce that
// evidence, and an observed priority differentiation under contention is a
// Linux-only gate needing the kernel/driver/GPU identity, a contended
// workload and an oracle outside the policy path. This test asserts the
// routing and says nothing about QoS.
func TestSchedulingAttributesDifferByTenantBeforeValidation(t *testing.T) {
	limits := cudabpf.DefaultLimits()
	k := &fakeKfuncs{}
	br := schedBridge(t, k, &tenantPolicy{byTSG: map[uint64]cudabpf.SchedAttributes{
		1: {TimesliceUS: 500, SetTimeslice: true, Interleave: cudabpf.InterleaveLow, SetInterleave: true},
		2: {TimesliceUS: 5000, SetTimeslice: true, Interleave: cudabpf.InterleaveHigh, SetInterleave: true},
		// The tenant that tries to raise its own priority past the
		// validated range. It gets nothing, not the maximum.
		3: {TimesliceUS: limits.MaxTimesliceUS + 1, SetTimeslice: true},
	}})

	a1, out1 := br.OnTaskInit(&cudabpf.TaskInitContext{TSGID: 1})
	a2, out2 := br.OnTaskInit(&cudabpf.TaskInitContext{TSGID: 2})
	a3, out3 := br.OnTaskInit(&cudabpf.TaskInitContext{TSGID: 3})

	if !out1.Submitted || !out2.Submitted {
		t.Fatalf("in-range tenants were not applied: %+v %+v", out1, out2)
	}
	if a1.TimesliceUS == a2.TimesliceUS || a1.Interleave == a2.Interleave {
		t.Errorf("two tenants received the same attributes: %v and %v", a1, a2)
	}
	if out3.Fallback == false {
		t.Errorf("the over-asking tenant was not refused: %+v", out3)
	}
	if a3.SetTimeslice {
		t.Errorf("the over-asking tenant received %v", a3)
	}
	want := []uint64{500, 5000}
	if len(k.timeslices) != len(want) {
		t.Fatalf("timeslice calls = %v, want %v", k.timeslices, want)
	}
	for i := range want {
		if k.timeslices[i] != want[i] {
			t.Errorf("timeslice call %d = %d, want %d", i, k.timeslices[i], want[i])
		}
	}
}

// TestBindRejectionPath names the form it asserts: the kfunc
// bpf_nv_gpu_reject_bind, not a direct write to the allow member.
//
// Both forms exist in the interface. The kfunc is non-sleepable and its whole
// effect is to write that member, so the member is what the driver reads and
// the call is how a verified program is permitted to write it. This package
// therefore calls the kfunc and never assigns Allow itself; the test asserts
// both halves, because "we called the kfunc" and "we did not write the field"
// are separate claims and only the pair rules out the other mechanism.
//
// The determination comes from the interface sources. No target available to
// this lane registers nv_gpu_sched_ops, so it has not been observed on a
// running driver.
func TestBindRejectionPath(t *testing.T) {
	if cudabpf.BindRejection != "bpf_nv_gpu_reject_bind" {
		t.Fatalf("BindRejection = %q; this test asserts the kfunc form", cudabpf.BindRejection)
	}
	k := &fakeKfuncs{}
	br := schedBridge(t, k, &tenantPolicy{reject: map[uint64]string{7: "tenant over quota"}})

	admitted := &cudabpf.BindContext{TSGID: 1, Allow: 1}
	if d, out := br.OnBind(admitted); d.Reject || out.Submitted {
		t.Errorf("an admitted bind was rejected: %+v %+v", d, out)
	}
	if admitted.Allow != 1 {
		t.Error("the policy layer cleared allow on an admitted bind")
	}

	rejected := &cudabpf.BindContext{TSGID: 7, Allow: 1}
	d, out := br.OnBind(rejected)
	if !d.Reject || !out.Submitted {
		t.Fatalf("the rejection was not submitted: %+v %+v", d, out)
	}
	if k.rejects != 1 {
		t.Errorf("bpf_nv_gpu_reject_bind called %d times, want 1", k.rejects)
	}
	if rejected.Allow != 0 {
		t.Error("allow was not cleared; the kfunc's effect did not reach the context")
	}
	if out.Reason != "tenant over quota" {
		t.Errorf("reason = %q", out.Reason)
	}
}

// TestPriorityAttributeRejectedWhenSchedHookAbsent covers the case a caller
// is most likely to get wrong: the scheduler table is not registered, so
// there is nowhere for a priority to go. The request must be refused by name
// rather than accepted and dropped.
func TestPriorityAttributeRejectedWhenSchedHookAbsent(t *testing.T) {
	// A target with the memory table and not the scheduler one.
	b := patchedBTF()
	b.tables = []string{cudabpf.MemOpsTable, "tcp_congestion_ops"}

	l := newFakeLoader(b)
	br := cudabpf.NewBridge(l, &fakeKfuncs{}, cudabpf.Config{})
	_, err := br.Attach(context.Background(), schedObject(), schedSpec())
	if err == nil {
		t.Fatal("attached the scheduler table on a target that does not register it")
	}
	if !strings.Contains(err.Error(), cudabpf.SchedOpsTable) {
		t.Errorf("error = %v, want it to name %s", err, cudabpf.SchedOpsTable)
	}
	if l.attaches != 0 {
		t.Errorf("registered %d times despite the absent table", l.attaches)
	}

	// The memory table on the same target still attaches: absence is
	// per-table, and one missing hook does not disable the package.
	if _, err := br.Attach(context.Background(), memObject(), memSpec()); err != nil {
		t.Errorf("the memory table was refused because the scheduler table is absent: %v", err)
	}

	// Discovery says the same thing without being asked to attach.
	r, err := cudabpf.Discover(context.Background(), l)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if r.TableAvailable(cudabpf.SchedOpsTable) {
		t.Error("the scheduler table was reported available")
	}
	if !r.TableAvailable(cudabpf.MemOpsTable) {
		t.Error("the memory table was reported unavailable")
	}
}

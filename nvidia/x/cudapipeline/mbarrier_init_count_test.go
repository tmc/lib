package cudapipeline_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func initCountWaits() []string {
	return []string{
		"mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.test_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.try_wait.shared::cta.b64 %p, [bar], %state;",
		"mbarrier.test_wait.shared::cta.b64 %p, [bar], %state;",
	}
}

func TestMBarrierInitCountRefusal(t *testing.T) {
	for _, tc := range []struct {
		name, prefix, operand, kind string
		configured                  int
	}{
		{"unknown register", "", ", %count", "unverified barrier arrival count", 0},
		{"unknown expression", "", ", %count+1", "unverified barrier arrival count", 0},
		{"unknown symbol", "", ", count_param", "unverified barrier arrival count", 0},
		{"explicit defeats fallback", "", ", %count", "unverified barrier arrival count", 1},
		{"zero", "", ", 0", "invalid barrier arrival count", 0},
		{"negative", "", ", -1", "invalid barrier arrival count", 0},
		{"zero defeats fallback", "", ", 0", "invalid barrier arrival count", 1},
		{"register zero", "mov.u32 %count, 0;\n", ", %count", "invalid barrier arrival count", 0},
		{"register negative", "mov.s32 %count, -2;\n", ", %count", "invalid barrier arrival count", 0},
		{"missing", "", "", "unverified barrier arrival count", 0},
		{"negative fallback", "", "", "invalid barrier arrival count", -1},
		{"invalidated numeric fact", "mov.u32 %count, 1;\nadd.u32 %count, %other, 1;\n", ", %count", "unverified barrier arrival count", 0},
		{"predicated numeric write", "mov.u32 %count, 1;\n@%p mov.u32 %count, 2;\n", ", %count", "unverified barrier arrival count", 0},
		{"unresolved copy clobbers", "mov.u32 %count, 1;\nmov.u32 %count, %unknown;\n", ", %count", "unverified barrier arrival count", 0},
		{"unresolved source later set", "mov.u32 %count, %unknown;\nmov.u32 %unknown, 1;\n", ", %count", "unverified barrier arrival count", 0},
	} {
		for _, wait := range initCountWaits() {
			t.Run(tc.name+"/"+wait, func(t *testing.T) {
				body := tc.prefix + "mbarrier.init.shared::cta.b64 [bar]" + tc.operand + ";\n" +
					"mbarrier.arrive.shared::cta.b64 %state, [bar];\n" + wait + "\n" +
					"mbarrier.arrive.shared::cta.b64 %state, [bar];\n" + wait + "\nret;"
				k := parseControlKernel(t, body)
				var opts []cudapipeline.Option
				if tc.configured != 0 {
					opts = append(opts, cudapipeline.WithBarrierArrivalCount("bar", tc.configured))
				}
				tracker := cudapipeline.NewBarrierTracker(opts...)
				violations, err := tracker.ValidateKernel(k)
				if err != nil || len(violations) != 5 {
					t.Fatalf("violations=%v err=%v, want init refusal and four uninitialized uses", violations, err)
				}
				init := k.Instructions[len(k.Instructions)-6]
				first := violations[0]
				if first.Violation != tc.kind || first.Line != init.Line || first.Opcode != init.Opcode || first.Barrier != "bar" || first.Phase != 0 {
					t.Fatalf("unanchored init diagnostic: %+v", first)
				}
				for _, v := range violations[1:] {
					if v.Violation != cudapipeline.ErrUninitializedBarrier || v.Barrier != "bar" || v.Phase != 0 {
						t.Fatalf("unknown initialization became a modeled phase: %v", violations)
					}
				}
				b := tracker.Barrier("bar")
				if b == nil || b.Initialized || b.ExpectedArrivals != 0 || b.PendingArrivals != 0 || b.Phase != 0 || b.Arrived || len(b.Events) != 0 {
					t.Fatalf("invented initialization, arrival count or phase: %+v", b)
				}
			})
		}
	}
	// A configured zero is distinct from no configuration, but neither can
	// supply a valid positive count when the PTX count is omitted.
	tracker := cudapipeline.NewBarrierTracker(cudapipeline.WithBarrierArrivalCount("bar", 0))
	v, err := tracker.ValidateKernel(parseControlKernel(t, "mbarrier.init.shared::cta.b64 [bar];\nret;"))
	if err != nil || len(v) != 1 || v[0].Violation != "invalid barrier arrival count" {
		t.Fatalf("zero fallback admitted: %v, %v", v, err)
	}
	// Missing the entire operand list must also fail even without a later use.
	v, err = cudapipeline.ValidateKernel(&cudaptx.Kernel{Body: true, Instructions: []cudaptx.Instruction{
		{Line: 17, Opcode: "mbarrier.init.shared::cta.b64"},
	}})
	if err != nil || len(v) != 1 || v[0].Line != 17 || v[0].Violation != "unverified barrier arrival count" {
		t.Fatalf("empty initialization admitted: %v, %v", v, err)
	}
}

func TestMBarrierInitCountResolved(t *testing.T) {
	for _, tc := range []struct {
		name, prefix, address, operand string
		count, fallback                int
	}{
		{"immediate one", "", "bar", ", 1", 1, 0},
		{"immediate two", "", "bar", ", 2", 2, 0},
		{"hex", "", "bar", ", 0x2", 2, 0},
		{"register", "mov.u32 %count, 2;\n", "bar", ", %count", 2, 0},
		{"self copy", "mov.u32 %count, 2;\nmov.u32 %count, %count;\n", "bar", ", %count", 2, 0},
		{"numeric copy", "mov.u32 %source, 2;\nmov.u32 %count, %source;\n", "bar", ", %count", 2, 0},
		{"numeric copy chain", "mov.u32 %source, 2;\nmov.u32 %copy, %source;\nmov.u32 %count, %copy;\n", "bar", ", %count", 2, 0},
		{"copy before source clobber", "mov.u32 %source, 2;\nmov.u32 %count, %source;\nmov.u32 %source, 1;\n", "bar", ", %count", 2, 0},
		{"copy before unknown source clobber", "mov.u32 %source, 2;\nmov.u32 %count, %source;\nadd.u32 %source, %other, 1;\n", "bar", ", %count", 2, 0},
		{"address alias", "mov.u64 %bar, bar;\nmov.u32 %count, 2;\n", "%bar", ", %count", 2, 0},
		{"missing with positive fallback", "", "bar", "", 2, 2},
		{"alias fallback", "mov.u64 %bar, bar;\n", "%bar", "", 2, 2},
		{"explicit count overrides fallback", "", "bar", ", 2", 2, 1},
	} {
		for _, wait := range initCountWaits() {
			t.Run(tc.name+"/"+wait, func(t *testing.T) {
				body := tc.prefix + "mbarrier.init.shared::cta.b64 [" + tc.address + "]" + tc.operand + ";\n"
				for phase := 0; phase < 2; phase++ {
					for arrival := 0; arrival < tc.count; arrival++ {
						body += "mbarrier.arrive.shared::cta.b64 %state, [bar];\n"
					}
					phaseWait := wait
					if phase == 1 && strings.Contains(wait, ".parity.") {
						phaseWait = strings.Replace(wait, ", 0;", ", 1;", 1)
					}
					body += phaseWait + "\n"
				}
				body += "ret;"
				var opts []cudapipeline.Option
				if tc.fallback != 0 {
					opts = append(opts, cudapipeline.WithBarrierArrivalCount("bar", tc.fallback))
				}
				tracker := cudapipeline.NewBarrierTracker(opts...)
				violations, err := tracker.ValidateKernel(parseControlKernel(t, body))
				if err != nil || len(violations) != 0 {
					t.Fatalf("known count refused: %v, %v", violations, err)
				}
				b := tracker.Barrier("bar")
				if b == nil || !b.Initialized || b.ExpectedArrivals != tc.count || b.PendingArrivals != tc.count || b.Phase != 2 || b.Arrived {
					t.Fatalf("known count or later phase changed: %+v", b)
				}
				var inits, waits int
				for _, event := range b.Events {
					if event.Kind == cudapipeline.EventInit {
						inits++
						if event.Count != tc.count {
							t.Fatal("init event has wrong resolved count", event)
						}
					}
					if event.Kind == cudapipeline.EventWait {
						waits++
					}
				}
				if inits != 1 || waits != 2 {
					t.Fatalf("events: inits=%d waits=%d", inits, waits)
				}
			})
		}
	}
}

func TestMBarrierInitCountCausalControls(t *testing.T) {
	// If a copied count followed the source's later write (or init silently
	// became one), this one-arrival wait would be falsely clean.
	for _, init := range []string{
		"mbarrier.init.shared::cta.b64 [bar], 2;\n",
		"mov.u32 %source, 2;\nmov.u32 %count, %source;\nmov.u32 %source, 1;\nmbarrier.init.shared::cta.b64 [bar], %count;\n",
		"mov.u32 %count, 2;\nmbarrier.init.shared::cta.b64 [bar], %count;\nmov.u32 %count, 1;\n",
	} {
		for _, wait := range initCountWaits() {
			tracker := cudapipeline.NewBarrierTracker()
			violations, err := tracker.ValidateKernel(parseControlKernel(t, init+"mbarrier.arrive.shared::cta.b64 %state, [bar];\n"+wait+"\nret;"))
			if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrMissingArrive || violations[0].Expected != 2 || violations[0].Actual != 1 {
				t.Fatalf("underarrival became clean or unresolved: %v, %v", violations, err)
			}
		}
	}
	// Unknown reinitialization must not inherit a prior valid count. A later
	// actual valid init may recover, without erasing earlier diagnostics.
	for _, invalidate := range []bool{false, true} {
		body := "mbarrier.init.shared::cta.b64 [bar], 1;\n"
		if invalidate {
			body += "mbarrier.inval.shared::cta.b64 [bar];\n"
		}
		body += "mbarrier.init.shared::cta.b64 [bar], %count;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\nmbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;\n"
		tracker := cudapipeline.NewBarrierTracker()
		violations, err := tracker.ValidateKernel(parseControlKernel(t, body+"ret;"))
		if err != nil || tracker.Barrier("bar").Initialized || tracker.Barrier("bar").Phase != 0 {
			t.Fatalf("unknown reinit inherited known state: %v, %v", violations, err)
		}
		want := 3
		if !invalidate {
			want++ // Preserve the adopted duplicate-initialization diagnostic.
		}
		if len(violations) != want {
			t.Fatalf("lost reinit/use diagnostics: %v", violations)
		}
		body += "mbarrier.init.shared::cta.b64 [bar], 2;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\nmbarrier.arrive.shared::cta.b64 %state, [bar];\nmbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;\nret;"
		violations, err = tracker.ValidateKernel(parseControlKernel(t, body))
		if err != nil || len(violations) != want || !tracker.Barrier("bar").Initialized || tracker.Barrier("bar").ExpectedArrivals != 2 || tracker.Barrier("bar").Phase != 1 {
			t.Fatalf("known reinit failed recovery or erased errors: %v, %v", violations, err)
		}
	}
	// A distinct valid barrier and later kernel validation cannot acquire the
	// refused barrier's state, nor erase its diagnostic from this kernel.
	tracker := cudapipeline.NewBarrierTracker()
	bad := "mbarrier.init.shared::cta.b64 [bad], %unknown;\n"
	good := "mbarrier.init.shared::cta.b64 [bar], 1;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\nmbarrier.test_wait.parity.shared::cta.b64 %p, [bar], 0;\nret;"
	v, err := tracker.ValidateKernel(parseControlKernel(t, bad+good))
	if err != nil || len(v) != 1 || v[0].Barrier != "bad" || tracker.Barrier("bar").Phase != 1 || tracker.Barrier("bad").Initialized {
		t.Fatalf("barrier isolation: %v, %v", v, err)
	}
	v, err = tracker.ValidateKernel(parseControlKernel(t, good))
	if err != nil || len(v) != 0 || tracker.Barrier("bad") != nil {
		t.Fatalf("kernel reset retained refused state: %v, %v", v, err)
	}
}

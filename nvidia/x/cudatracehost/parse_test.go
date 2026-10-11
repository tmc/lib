package cudatracehost

import (
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"testing"
)

func TestParse(t *testing.T) {
	o, err := Parse(" app-123 [001] .... 12.000003: cuda0_in: (0x12) a0=0xdead a1=1024")
	if err != nil || o.TID != 123 || o.Timestamp != 12000003000 || o.Args[0] != 0xdead || !o.ArgsKnown[1] {
		t.Fatalf("%+v %v", o, err)
	}
	for _, s := range []string{"garbage", "app-0 [001] .... 12.0: cuda0_in:", "app-1 [001] .... 12.1234567891: cuda0_in:", "app-1 [001] .... 12.0: cuda0_in: a0=bad", "app-1 [001] .... 12.0: cuda0_in: a0=1 a0=2"} {
		if _, err := Parse(s); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
}
func target() Target {
	return Target{Name: "cuda0", Arguments: 1, Identity: cudaprobe.Identity{Process: cudaprobe.Process{PID: 1, Generation: 2}, BuildID: "fixture", Library: "/fixture", Version: "fixture", Symbol: "cudaFree", ABI: "linux-amd64-sysv-v1"}}
}
func TestCorrelation(t *testing.T) {
	c, err := NewCorrelator([]Target{target()}, 2)
	if err != nil {
		t.Fatal(err)
	}
	entry := Observation{Name: "cuda0_in", TID: 4, Timestamp: 5, Args: [11]uint64{0x123}, ArgsKnown: [11]bool{true}}
	if cap, err := c.Observe(entry); err != nil || cap != nil {
		t.Fatalf("%v %v", cap, err)
	}
	cap, err := c.Observe(Observation{Name: "cuda0_out", TID: 4, Timestamp: 8, ReturnKnown: true})
	if err != nil {
		t.Fatal(err)
	}
	event, err := cudaprobe.Decode(*cap)
	if err != nil || !event.Decoded || event.Address != 0x123 || event.Sequence != 1 {
		t.Fatalf("%+v %v", event, err)
	}
	if !c.Incomplete {
		t.Fatal("late attach claimed complete")
	}
	c.Observe(entry)
	c.Gap()
	if _, err := c.Observe(Observation{Name: "cuda0_out", TID: 4, Timestamp: 8, ReturnKnown: true}); err == nil {
		t.Fatal("return bridged loss")
	}
}
func TestBoundAndMissingArgument(t *testing.T) {
	c, _ := NewCorrelator([]Target{target()}, 1)
	if _, err := c.Observe(Observation{Name: "cuda0_in", TID: 1}); err == nil {
		t.Fatal("absent argument became zero")
	}
	entry := Observation{Name: "cuda0_in", TID: 1, ArgsKnown: [11]bool{true}}
	c.Observe(entry)
	if _, err := c.Observe(entry); err == nil || c.count != 0 {
		t.Fatal("overflow retained outstanding calls")
	}
}

func TestCompletedThreadsReleaseStacks(t *testing.T) {
	c, err := NewCorrelator([]Target{target()}, 1)
	if err != nil {
		t.Fatal(err)
	}
	for tid := uint32(1); tid <= 10000; tid++ {
		entry := Observation{Name: "cuda0_in", TID: tid, Timestamp: 1, ArgsKnown: [11]bool{true}}
		if _, err := c.Observe(entry); err != nil {
			t.Fatal(err)
		}
		if len(c.stacks) != 1 || c.count != 1 {
			t.Fatalf("pending stacks=%d calls=%d", len(c.stacks), c.count)
		}
		if _, err := c.Observe(Observation{Name: "cuda0_out", TID: tid, Timestamp: 2, ReturnKnown: true}); err != nil {
			t.Fatal(err)
		}
		if len(c.stacks) != 0 || c.count != 0 {
			t.Fatalf("completed stacks=%d calls=%d", len(c.stacks), c.count)
		}
	}
}

func TestPendingThreadsAndNestedFramesBounded(t *testing.T) {
	c, err := NewCorrelator([]Target{target()}, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, tid := range []uint32{1, 2, 1} {
		if _, err := c.Observe(Observation{Name: "cuda0_in", TID: tid, Timestamp: 1, ArgsKnown: [11]bool{true}}); err != nil {
			t.Fatal(err)
		}
	}
	// Returning an inner frame must preserve the outer one, and clear its data.
	frames := c.stacks[1]
	if _, err := c.Observe(Observation{Name: "cuda0_out", TID: 1, Timestamp: 2, ReturnKnown: true}); err != nil {
		t.Fatal(err)
	}
	if len(c.stacks) != 2 || len(c.stacks[1]) != 1 || frames[1].target.Name != "" {
		t.Fatalf("nested stack retained popped frame: %+v", c.stacks)
	}
	for _, tid := range []uint32{3, 4} {
		_, err = c.Observe(Observation{Name: "cuda0_in", TID: tid, Timestamp: 3, ArgsKnown: [11]bool{true}})
		if (err != nil) != (tid == 4) {
			t.Fatalf("tid=%d err=%v", tid, err)
		}
	}
	if c.count != 0 || len(c.stacks) != 0 {
		t.Fatal("overflow retained pending stacks")
	}
	if _, err := c.Observe(Observation{Name: "cuda0_out", TID: 2, Timestamp: 4, ReturnKnown: true}); err == nil {
		t.Fatal("paired return across overflow")
	}
}

func TestDuplicateReturnRejected(t *testing.T) {
	for _, fields := range []string{"ret=0 ret=1", "ret=0 ret=0"} {
		if _, err := Parse("app-1 [001] .... 12.0: cuda0_out: " + fields); err == nil {
			t.Fatalf("accepted %s", fields)
		}
	}
}

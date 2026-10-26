package cudabpfdev

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

func TestInvocationObservationLifecycle(t *testing.T) {
	for _, candidate := range []bool{false, true} {
		t.Run(fmt.Sprint(candidate), func(t *testing.T) {
			l, v := invocationFixture(t, candidate)
			r := l.ObservationReader()
			calls := len(v.f.calls)
			before, err := r.ReadObservation()
			if err != nil || !before.Ready || before.Used || before.Pending || before.Closed || before.Complete || before.TimestampNano != 0 || before.Selection.Candidate != candidate {
				t.Fatalf("initial observation=%+v error=%v", before, err)
			}
			if before.Geometry != l.geometry || !reflect.DeepEqual(before.Schema, l.schema) || before.Original != l.originalKernel || before.Selection != l.selection {
				t.Fatal("owner descriptor lost")
			}
			before.Schema.Params[0].Name = "changed"
			again, _ := r.ReadObservation()
			if again.Schema.Params[0].Name == "changed" || len(v.f.calls) != calls {
				t.Fatal("read aliased schema or called operations")
			}
			if err := l.Launch(context.Background(), uint64(0x1000)); err != nil {
				t.Fatal(err)
			}
			calls = len(v.f.calls)
			after, _ := r.ReadObservation()
			if !after.Used || after.Pending || !after.Complete || after.TimestampNano <= 0 || !before.SameInvocation(after) || len(v.f.calls) != calls {
				t.Fatalf("completion=%+v", after)
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			calls = len(v.f.calls)
			closed, _ := r.ReadObservation()
			if !closed.Closed || !closed.Complete || closed.TimestampNano != after.TimestampNano || len(v.f.calls) != calls {
				t.Fatal("closed receipt lost or read called operations")
			}
			other, _ := invocationFixture(t, candidate)
			defer other.Close()
			o, _ := other.ObservationReader().ReadObservation()
			if before.SameInvocation(o) {
				t.Fatal("same-image owners interchangeable")
			}
		})
	}
}

func TestInvocationObservationFailures(t *testing.T) {
	for _, fail := range []string{"enter", "stream", "launch", "wait", "restore"} {
		t.Run(fail, func(t *testing.T) {
			l, v := invocationFixture(t, false)
			v.f.fail = fail
			if err := l.Launch(context.Background(), uint64(0x1000)); err == nil {
				t.Fatal("missing launch failure")
			}
			calls := len(v.f.calls)
			o, err := l.ObservationReader().ReadObservation()
			if err != nil || !o.Used || o.Complete || o.TimestampNano != 0 || len(v.f.calls) != calls {
				t.Fatalf("failed receipt=%+v error=%v", o, err)
			}
			if (fail == "launch" || fail == "wait") && !o.Pending {
				t.Fatal("lost pending state")
			}
			v.f.fail = ""
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			o, _ = l.ObservationReader().ReadObservation()
			if !o.Closed || o.Complete {
				t.Fatal("close invented success")
			}
		})
	}
	var r InvocationReader
	if _, err := r.ReadObservation(); err == nil {
		t.Fatal("zero reader available")
	}
	var l *LoadedInjection
	if _, err := l.ObservationReader().ReadObservation(); err == nil {
		t.Fatal("nil owner available")
	}
}

func ExampleLoadedInjection_ObservationReader() {
	var owner *LoadedInjection
	_, err := owner.ObservationReader().ReadObservation()
	fmt.Println(err != nil)
	// Output: true
}

func ExampleInvocationReader_ReadObservation() {
	var reader InvocationReader
	_, err := reader.ReadObservation()
	fmt.Println(err != nil)
	// Output: true
}

func ExampleInvocationObservation_SameInvocation() {
	var observation InvocationObservation
	fmt.Println(observation.SameInvocation(observation))
	// Output: false
}

package cuptirange

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

func graphEvent(name string, handle driver.GraphExec) *cupticallback.Event {
	e := launchEvent("")
	e.Api.FunctionName = name
	data := new(cupti.CallbackData)
	// Use pointer storage at the generated ABI offset so checkptr can retain it.
	*(*unsafe.Pointer)(unsafe.Add(unsafe.Pointer(data), unsafe.Offsetof(data.FunctionParams))) = unsafe.Pointer(&handle)
	e.Raw = unsafe.Pointer(data)
	return e
}

func graphTracker() *kernelNames {
	return &kernelNames{context: 7, limit: 8, nativeKernelReplay: true, graphs: map[driver.GraphExec]kernelGraph{
		42: {symbols: []string{"a", "b"}, validate: func() (driver.GraphExec, []string, error) { return 42, []string{"a", "b"}, nil }},
	}}
}

func TestGraphKernelIdentityNativeProof(t *testing.T) {
	k := graphTracker()
	for range 2 {
		if err := k.begin(); err != nil {
			t.Fatal(err)
		}
		k.callback(launchEvent("before"))
		k.callback(graphEvent("cuGraphLaunch", 42))
		k.callback(launchEvent("after"))
		if err := k.end(); err != nil {
			t.Fatal(err)
		}
	}
	ranges := []RangeInfo{{Name: "0"}, {Name: "1"}, {Name: "2"}, {Name: "3"}}
	if err := k.resolve(ranges); err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"before", "a", "b", "after"} {
		if ranges[i].Name != want {
			t.Fatalf("range %d=%+v, want %s", i, ranges[i], want)
		}
	}
	if err := k.checkCount(3); err == nil {
		t.Fatal("accepted missing native graph range")
	}
}

func TestGraphKernelIdentityRefusals(t *testing.T) {
	for _, name := range []string{"user-replay", "unregistered", "missing-params", "capacity", "duplicate-internal", "mutation-before-run", "invalidated-proof"} {
		t.Run(name, func(t *testing.T) {
			k := graphTracker()
			if name == "mutation-before-run" {
				k.callback(graphEvent("cuGraphExecKernelNodeSetParams", 42))
				if err := k.begin(); err == nil {
					t.Fatal("accepted mutated graph")
				}
				return
			}
			if err := k.begin(); err != nil {
				t.Fatal(err)
			}
			e := graphEvent("cuGraphLaunch", 42)
			switch name {
			case "user-replay":
				k.nativeKernelReplay = false
			case "unregistered":
				e = graphEvent("cuGraphLaunch", 43)
			case "missing-params":
				e.Raw = nil
			case "capacity":
				k.limit = 1
			case "invalidated-proof":
				g := k.graphs[42]
				g.validate = func() (driver.GraphExec, []string, error) { return 0, nil, errors.New("updated") }
				k.graphs[42] = g
			}
			k.callback(e)
			if name == "duplicate-internal" {
				k.callback(launchEvent("a"))
				k.callback(launchEvent("b"))
				if err := k.end(); err != nil {
					t.Fatal(err)
				}
				if err := k.checkCount(2); err == nil {
					t.Fatal("silently deduplicated ambiguous internal callbacks")
				}
				return
			}
			if err := k.end(); err == nil {
				t.Fatal("accepted unproved graph trace")
			}
		})
	}
}

func TestGraphMutationIgnoresOtherExecAndGetters(t *testing.T) {
	k := graphTracker()
	k.callback(graphEvent("cuGraphExecDestroy", 43))
	k.callback(graphEvent("cuGraphExecGetFlags", 42))
	if err := k.begin(); err != nil {
		t.Fatal(err)
	}
	k.callback(graphEvent("cuGraphLaunch", 42))
	if err := k.end(); err != nil {
		t.Fatal(err)
	}
	if err := k.checkCount(2); err != nil {
		t.Fatal(err)
	}
}

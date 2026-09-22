package cudagraph

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCaptureContextOwnership(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"entry", "begin", "end", "end owner", "restore", "success"} {
		t.Run(stage, func(t *testing.T) {
			var calls []string
			inside := false
			dev := &cuda.Device{}
			graph, err := captureGraph(dev, func(fn func() error) error {
				calls = append(calls, "entry")
				if stage == "entry" {
					return failure
				}
				inside = true
				err := fn()
				inside = false
				calls = append(calls, "restore")
				if stage == "restore" {
					return failure
				}
				return err
			}, func() error {
				if !inside {
					t.Fatal("begin outside context")
				}
				calls = append(calls, "begin")
				if stage == "begin" {
					return failure
				}
				return nil
			}, func() (driver.Graph, error) {
				if !inside {
					t.Fatal("end outside context")
				}
				calls = append(calls, "end")
				if stage == "end" {
					return 0, failure
				}
				if stage == "end owner" {
					return 7, failure
				}
				return 7, nil
			}, func() {
				if !inside {
					t.Fatal("callback outside context")
				}
				calls = append(calls, "callback")
			})
			want := []string{"entry", "begin", "callback", "end", "restore"}
			if stage == "entry" {
				want = []string{"entry"}
			}
			if stage == "begin" {
				want = []string{"entry", "begin", "restore"}
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls=%v, want %v", calls, want)
			}
			if errors.Is(err, failure) != (stage != "success") {
				t.Fatalf("error=%v", err)
			}
			wantOwner := stage == "end owner" || stage == "restore" || stage == "success"
			if (graph != nil) != wantOwner {
				t.Fatalf("owner=%v", graph)
			}
			if graph != nil && (graph.raw != 7 || graph.device != dev) {
				t.Fatal("wrong graph identity")
			}
		})
	}
}

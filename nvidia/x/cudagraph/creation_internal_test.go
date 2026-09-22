package cudagraph

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCreationOwnership(t *testing.T) {
	failure := errors.New("injected failure")
	for _, kind := range []string{"graph", "exec"} {
		for _, stage := range []string{"entry", "create", "restore", "success"} {
			t.Run(kind+"/"+stage, func(t *testing.T) {
				calls := 0
				run := func(fn func() error) error {
					if stage == "entry" {
						return failure
					}
					if err := fn(); err != nil {
						return err
					}
					if stage == "restore" {
						return failure
					}
					return nil
				}
				dev := &cuda.Device{}
				var err error
				var owned bool
				if kind == "graph" {
					var g *Graph
					g, err = newGraph(dev, run, func() (driver.Graph, error) {
						calls++
						if stage == "create" {
							return 0, failure
						}
						return 7, nil
					})
					owned = g != nil
					if owned {
						if g.raw != 7 || g.device != dev {
							t.Fatal("lost graph identity")
						}
						if err := g.destroy(func(fn func() error) error { return fn() }, func() error { return nil }); err != nil {
							t.Fatal(err)
						}
					}
				} else {
					storage := &launchArgs{}
					g := &Graph{raw: 1, device: dev, retained: []*launchArgs{storage}}
					var exec *Exec
					exec, err = g.instantiate([]string{"kernel"}, run, func() (driver.GraphExec, error) {
						calls++
						if stage == "create" {
							return 0, failure
						}
						return 9, nil
					})
					owned = exec != nil
					if owned {
						if exec.raw != 9 || exec.device != dev || len(exec.retained) != 1 || exec.retained[0] != storage || len(exec.kernelOrder) != 1 {
							t.Fatal("lost executable identity or storage")
						}
						if err := exec.destroy(func(fn func() error) error { return fn() }, func() error { return nil }); err != nil {
							t.Fatal(err)
						}
					}
				}
				if owned != (stage == "restore" || stage == "success") {
					t.Fatalf("owner returned = %v", owned)
				}
				if errors.Is(err, failure) != (stage != "success") {
					t.Fatalf("error = %v", err)
				}
				if (calls == 0) != (stage == "entry") {
					t.Fatalf("create calls = %d", calls)
				}
			})
		}
	}
}

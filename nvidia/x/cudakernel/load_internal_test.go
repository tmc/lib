package cudakernel

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestLoadContextOwnership(t *testing.T) {
	failure := errors.New("injected")
	for _, stage := range []string{"entry", "load", "restore", "success"} {
		t.Run(stage, func(t *testing.T) {
			calls := 0
			mod, err := loadInContext(func(fn func() error) error {
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
			}, func() (driver.Module, error) {
				calls++
				if stage == "load" {
					return 0, failure
				}
				return 7, nil
			})
			want := driver.Module(7)
			if stage == "entry" || stage == "load" {
				want = 0
			}
			if mod != want {
				t.Fatalf("module = %v, want %v", mod, want)
			}
			if errors.Is(err, failure) != (stage != "success") {
				t.Fatalf("error = %v", err)
			}
			if (calls == 0) != (stage == "entry") {
				t.Fatalf("load calls = %d", calls)
			}
		})
	}
}

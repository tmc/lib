//go:build linux

package cuptipcsampling_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// Run in a fresh process under the hardware lease. This enables PC sampling
// exactly once and collects no workload or performance sample.
func TestNativeCollectorExclusion(t *testing.T) {
	if os.Getenv("CUPTI_PC_OWNERSHIP_TEST") != "1" {
		t.Skip("CUPTI_PC_OWNERSHIP_TEST=1 required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	err = dev.WithContext(func() error {
		ctx, err := driver.NewContextGetCurrent()
		if err != nil {
			return err
		}
		lease, err := cuptiowner.AcquireDevice(0)
		if err != nil {
			return err
		}
		disable, rejected := cuptipcsampling.Enable(cupti.Context(ctx))
		lease.Release()
		if disable != nil {
			defer disable()
			t.Fatal("conflicting enable returned native ownership")
		}
		if rejected == nil || !strings.Contains(rejected.Error(), "already collecting") {
			t.Fatalf("conflict error=%v", rejected)
		}
		disable, err = cuptipcsampling.Enable(cupti.Context(ctx))
		if disable != nil {
			defer func() {
				if err := disable(); err != nil {
					t.Error(err)
				}
			}()
		}
		if err != nil {
			return err
		}
		if other, err := cuptiowner.AcquireDevice(0); err == nil {
			other.Release()
			t.Fatal("metric collector admitted during native PC sampling")
		}
		inventory, err := cuptipcsampling.StallReasons(cupti.Context(ctx), 1024)
		if err != nil {
			return err
		}
		if len(inventory.Reasons) == 0 {
			t.Fatal("native PC sampling returned no reason inventory")
		}
		if err := disable(); err != nil {
			return err
		}
		other, err := cuptiowner.AcquireDevice(0)
		if err != nil {
			return err
		}
		other.Release()
		t.Logf("native exclusion and release passed: CUPTI %d, %d stall reasons", inventory.CUPTIVersion, len(inventory.Reasons))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

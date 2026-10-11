//go:build linux

package cuptipcsampling_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestNativeStallReasons(t *testing.T) {
	d := cudatest.Require(t)
	defer d.Close()

	err := d.WithContext(func() error {
		ctx, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		ctxHandle := ctx
		disable, err := cuptipcsampling.Enable(ctxHandle)
		if disable != nil {
			defer func() {
				if disErr := disable(); disErr != nil {
					t.Errorf("disable: %v", disErr)
				}
			}()
		}
		if err != nil {
			return err
		}
		inventory, err := cuptipcsampling.StallReasons(ctxHandle, 1024)
		if err != nil {
			return err
		}
		if len(inventory.Reasons) == 0 {
			t.Fatal("expected non-empty stall reason inventory")
		}
		data, err := json.Marshal(inventory)
		if err != nil {
			return err
		}
		t.Logf("native reason inventory (version %d, %d reasons): %s", inventory.CUPTIVersion, len(inventory.Reasons), data)
		return nil
	})
	if errors.Is(err, errors.ErrUnsupported) {
		t.Skipf("installed CUPTI has no qualified stall reason layout: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
}

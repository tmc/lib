//go:build linux

package cuptipcsampling_test

import (
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestNativeStallReasons(t *testing.T) {
	d := cudatest.Require(t)
	defer d.Close()

	err := d.WithContext(func() error {
		ctx, err := driver.NewContextGetCurrent()
		if err != nil {
			return err
		}
		ctxHandle := cupti.Context(ctx)
		disable, err := cuptipcsampling.Enable(ctxHandle)
		if err != nil {
			return err
		}
		defer func() {
			if disErr := disable(); disErr != nil {
				t.Errorf("disable: %v", disErr)
			}
		}()
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
	if err != nil {
		t.Fatal(err)
	}
}

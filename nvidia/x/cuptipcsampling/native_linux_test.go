//go:build linux

package cuptipcsampling_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func TestNativeStallReasons(t *testing.T) {
	if os.Getenv("CUPTI_PC_REASON_TEST") != "1" {
		t.Skip("CUPTI_PC_REASON_TEST=1 required")
	}
	d, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	}()
	err = d.WithContext(func() error {
		ctx, err := driver.NewContextGetCurrent()
		if err != nil {
			return err
		}
		inventory, err := cuptipcsampling.StallReasons(cupti.Context(ctx), 1024)
		if err != nil {
			return err
		}
		data, err := json.Marshal(inventory)
		if err != nil {
			return err
		}
		t.Logf("native reason inventory: %s", data)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

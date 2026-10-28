//go:build linux

package cuptipcsampling

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestNativeSessionsShareContext runs, on one context, the sequence the
// autotune collector runs per candidate: enable for the reason inventory,
// disable, then a started and closed session. Collection forces a garbage
// collection first, so a buffer the package released while CUPTI still holds
// its address is reclaimed before the next enable uses it.
func TestNativeSessionsShareContext(t *testing.T) {
	if os.Getenv("CUPTI_PC_REPEAT_TEST") != "1" {
		t.Skip("CUPTI_PC_REPEAT_TEST=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("CUPTI_PC_REPEAT_TEST=1 requires an inherited gpu-lock: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	defer driver.ProfilerStop()
	err = dev.WithContext(func() error {
		ctx, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		for i := range 5 {
			runtime.GC()
			disable, err := Enable(ctx)
			if err != nil {
				return fmt.Errorf("session %d: enable for inventory: %w", i, err)
			}
			inv, err := StallReasons(ctx, 1<<10)
			if err = errors.Join(err, disable()); err != nil {
				return fmt.Errorf("session %d: inventory: %w", i, err)
			}
			config := Config{
				Period: 12, WorkerSleepMS: 10,
				MaxReasons:   len(inv.Reasons),
				ScratchBytes: 8 << 20, HardwareBytes: 8 << 20, MaxNativeBytes: 64 << 20,
				PCsPerBuffer: 512, MaxOutputBytes: 8 << 20, MaxReads: 16,
				MaxNames: 4096, MaxNameBytes: 256, MaxTotalNameBytes: 4096 * 256,
			}
			for _, r := range inv.Reasons {
				config.ReasonIDs = append(config.ReasonIDs, r.Index)
			}
			runtime.GC()
			s, err := Start(ctx, config)
			if err != nil {
				if s != nil {
					err = errors.Join(err, s.Close())
				}
				return fmt.Errorf("session %d: start: %w", i, err)
			}
			if err := errors.Join(s.Stop(), s.Close()); err != nil {
				return fmt.Errorf("session %d: stop and close: %w", i, err)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

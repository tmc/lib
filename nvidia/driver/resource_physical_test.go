//go:build linux

package driver_test

import (
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestNativeResourceSplitArray(t *testing.T) {
	if os.Getenv("CUDA_DRIVER_RESOURCE_TEST") != "1" {
		t.Skip("set CUDA_DRIVER_RESOURCE_TEST=1 for native resource-array validation")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	input, err := dev.GetDevResource(driver.DevResourceTypeSm)
	if err != nil {
		t.Fatal(err)
	}
	if input.Type != driver.DevResourceTypeSm {
		t.Fatalf("input resource type = %v", input.Type)
	}
	// cuda.h CUdevSmResource fields smCount and minSmPartitionSize occupy
	// the first two uint32 words of the resource union. They are output-only.
	total, minimum := input.AnonymousUnion[0], input.AnonymousUnion[1]
	if minimum == 0 || total < 2*minimum {
		t.Fatalf("two-group validation unavailable: %d SMs, minimum partition %d", total, minimum)
	}
	var available uint32
	if err := driver.DevSmResourceSplitByCount(nil, &available, &input, nil, 0, minimum); err != nil {
		t.Fatal(err)
	}
	if available < 2 {
		t.Fatalf("two-group validation unavailable: split query returned %d groups", available)
	}
	var output struct {
		Before [4]uint64
		Groups [2]driver.DevResource
		After  [4]uint64
	}
	const guard = uint64(0xa55ac33c5aa53cc3)
	for i := range output.Before {
		output.Before[i], output.After[i] = guard, guard
	}
	var remainder driver.DevResource
	count := uint32(len(output.Groups))
	if err := driver.DevSmResourceSplitByCount(&output.Groups[0], &count, &input, &remainder, 0, minimum); err != nil {
		t.Fatal(err)
	}
	for i := range output.Before {
		if output.Before[i] != guard || output.After[i] != guard {
			t.Fatalf("resource array guard %d overwritten", i)
		}
	}
	if count != 2 {
		t.Fatalf("split wrote %d groups, want 2", count)
	}
	var assigned uint32
	for i, group := range output.Groups {
		if group.Type != driver.DevResourceTypeSm || group.AnonymousUnion[0] < minimum {
			t.Fatalf("group %d: type %v, SM count %d, minimum %d", i, group.Type, group.AnonymousUnion[0], minimum)
		}
		if group.AnonymousUnion[0] != output.Groups[0].AnonymousUnion[0] {
			t.Fatalf("groups are not symmetrical: %d and %d SMs", output.Groups[0].AnonymousUnion[0], group.AnonymousUnion[0])
		}
		assigned += group.AnonymousUnion[0]
	}
	var remaining uint32
	if remainder.Type == driver.DevResourceTypeSm {
		remaining = remainder.AnonymousUnion[0]
	} else if remainder.Type != driver.DevResourceTypeInvalid {
		t.Fatalf("remainder type = %v", remainder.Type)
	}
	if assigned+remaining != total {
		t.Fatalf("SM accounting: groups %d + remainder %d != input %d", assigned, remaining, total)
	}
	// Resource queries and splitting allocate no descriptors or contexts.
	t.Logf("two resource slots: input=%d, groups=%d+%d, remainder=%d, available=%d; guards intact", total, output.Groups[0].AnonymousUnion[0], output.Groups[1].AnonymousUnion[0], remaining, available)
}

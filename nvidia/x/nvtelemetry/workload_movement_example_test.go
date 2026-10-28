package nvtelemetry_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadMovementContext() {
	var b bytes.Buffer
	if err := nvtelemetry.WriteMovementContext(&b, movementFixture(), nvtelemetry.WorkloadLimits{}); err != nil {
		fmt.Println(err)
		return
	}
	_, c, err := nvtelemetry.ReadMovementContext(&b, movementSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	t := c.Transfers[0]
	fmt.Println(t.ID, t.Direction, *t.Bytes, t.Unit, t.State, t.Path.Mechanism)
	// Output: t1 peer_to_peer 9007199254740993 bytes completed nvlink
}

func ExampleWriteMovementContext() {
	c := movementFixture()
	c.Transfers[0].Unit = "KiB"
	fmt.Println(nvtelemetry.WriteMovementContext(&bytes.Buffer{}, c, nvtelemetry.WorkloadLimits{}))
	// Output: movement transfer "t1": bytes require exact unit bytes
}

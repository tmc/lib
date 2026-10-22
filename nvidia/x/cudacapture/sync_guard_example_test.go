package cudacapture_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func ExampleSyncArbiter() {
	arb := cudacapture.NewSyncArbiter(cudacapture.SyncAllow, nil)

	restoreForbid := arb.Forbid(cudacapture.SyncReject)
	err1 := arb.CheckSync("cuStreamSynchronize")

	// Temporarily allow synchronization for a legitimate event query.
	restoreAllow := arb.Allow()
	err2 := arb.CheckSync("cuEventElapsedTime")
	restoreAllow()

	err3 := arb.CheckSync("cuMemcpyDtoH")
	restoreForbid()

	err4 := arb.CheckSync("cuCtxSynchronize")
	fmt.Printf("forbid=%v allow_override=%v after_allow=%v restored=%v violations=%d\n",
		errors.Is(err1, cudacapture.ErrForbiddenSync),
		err2 == nil,
		errors.Is(err3, cudacapture.ErrForbiddenSync),
		err4 == nil,
		len(arb.Violations()))

	// Output:
	// forbid=true allow_override=true after_allow=true restored=true violations=2
}

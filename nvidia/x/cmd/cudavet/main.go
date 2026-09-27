// Command cudavet reports CUDA hazards that a Go compiler cannot see.
//
// Checks:
//
//	cargalign          kernel parameter layouts that disagree with cudakernel.Schema
//	unpinned           Go memory handed to asynchronous DMA without a runtime.Pinner
//	leak               handles created and never released
//
//	contextthread      a context made current without runtime.LockOSThread
//	unlockdestroy      a context released on a different thread from the one it is current on
//	graphshare         a graph used concurrently from two goroutines
//
//	devicederef        a device address turned into a Go pointer
//	devptruintptr      a DevicePtr passed as uintptr to a binding that types device memory
//	memallocptr        a Go pointer type stored in device memory
//	useafterdestroy    a handle used after it is released
//	sizeunits          a legacy BufferSize in elements allocated as bytes
//	sizetrunc          a size_t result narrowed without a bounds check
//	uintptrlive        the address of Go memory stored as a uintptr and not kept alive
//
//	asynchost          a host buffer written or freed while an async copy uses it
//	readbeforesync     a host buffer read before its async copy has finished
//	launcherr          a launch whose first synchronizing call discards its error
//	streammismatch     device memory written on one stream and used on another
//	querydone          the done result of Query discarded
//	profilerexit       a deferred ProfilerStop skipped by os.Exit or log.Fatal
//
//	errcompare         a binding error compared with == instead of errors.Is
//	uncheckedsync      a discarded error of a Synchronize or release
//	settererr          a discarded error of a setter, which keeps the old value
//	devinfo            a cuSOLVER devInfo buffer never checked
//	zeroversion        a version used after its error was discarded
//	legacycupti        a CUPTI function that always fails since CUDA 13.0
//
//	modecount          cuTENSOR modes that disagree with the descriptor's extents
//	firstelem          &s[0] of a slice that may be empty, or with a mismatched count
//	scalartype         alpha and beta of a different type from the compute type
//	pointermode        a device Scalar in host pointer mode, or a host Scalar in device mode
//	halfbits           a float converted to uint16 and passed to Float16BitsScalar as bits
//	rawenum            an integer literal where a named constant exists
//	fftdirection       a cuFFT direction written as -1 or 1
//	stringnul          a string argument with a NUL byte
//	jitlog             a whole JIT log buffer converted to a string
//	linkoptions        LinkCreate option buffers that die before LinkComplete
//
//	arch               a GPU architecture fixed in compile options
//	gridtrunc          a grid size that rounds down
//	blocklimit         a constant block or shared memory size over the hardware limit
//
//	groupbalance       an nccl.GroupStart not closed on every path
//	collectivediverge  a collective called on only some ranks
//
// Usage:
//
//	cudavet [flags] package...
//
// The checks take the flags of golang.org/x/tools/go/analysis/multichecker,
// so a single check runs as, for example,
//
//	cudavet -cargalign ./...
//
// cargalign computes no offsets of its own. Every offset it reports comes
// from [cudakernel.NewSchema], so that kernel parameter alignment has one
// definition and not two.
package main

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/multichecker"
)

var analyzers = []*analysis.Analyzer{
	cargalignAnalyzer,
	unpinnedAnalyzer,
	leakAnalyzer,

	contextThreadAnalyzer,
	unlockDestroyAnalyzer,
	graphShareAnalyzer,

	deviceDerefAnalyzer,
	devptrUintptrAnalyzer,
	memallocPtrAnalyzer,
	useAfterDestroyAnalyzer,
	sizeUnitsAnalyzer,
	sizeTruncAnalyzer,
	uintptrLiveAnalyzer,

	asyncHostAnalyzer,
	readBeforeSyncAnalyzer,
	launchErrAnalyzer,
	streamMismatchAnalyzer,
	queryDoneAnalyzer,
	profilerExitAnalyzer,

	errCompareAnalyzer,
	uncheckedSyncAnalyzer,
	setterErrAnalyzer,
	devInfoAnalyzer,
	zeroVersionAnalyzer,
	legacyCuptiAnalyzer,

	modeCountAnalyzer,
	firstElemAnalyzer,
	scalarTypeAnalyzer,
	pointerModeAnalyzer,
	halfBitsAnalyzer,
	rawEnumAnalyzer,
	fftDirectionAnalyzer,
	stringNULAnalyzer,
	jitLogAnalyzer,
	linkOptionsAnalyzer,

	archAnalyzer,
	gridTruncAnalyzer,
	blockLimitAnalyzer,

	groupBalanceAnalyzer,
	collectiveDivergeAnalyzer,
}

func main() {
	multichecker.Main(analyzers...)
}

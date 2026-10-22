//go:build linux

package main

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// cudaSourceFS is the source-of-truth for the CUDA fragments compiled by the
// runtime. Keeping the files in one embedded filesystem makes the logical
// filename used by #line directives derive from the embedded path itself.
//
//go:embed *.cu
var cudaSourceFS embed.FS

//go:embed cuda_probe.cu
var cudaProbeSource string

// cudaSourceChunk gives each embedded translation-unit fragment its own
// logical filename. Without these directives NVRTC reports line numbers in
// the concatenated string, which cannot be mapped back to a file on disk.
func cudaSourceChunk(name, source string) string {
	return fmt.Sprintf("#line 1 %q\n%s\n", name, strings.TrimSuffix(source, "\n"))
}

// cudaMathFiles is ordered by declaration dependencies. These are embedded
// source fragments for one NVRTC translation unit, not independently linked
// CUDA compilation units. cuda_q80_quant.cu is intentionally absent: it is
// compiled as a separate module with its own options.
var cudaMathFiles = [...]string{
	"cuda_probe.cu",
	"cuda_experts.cu",
	"cuda_experts_q4.cu",
	"cuda_experts_q4_batch.cu",
	"cuda_compressed.cu",
	"cuda_q80_layout.cu",
	"cuda_q80_matvec.cu",
	"cuda_q81.cu",
	"cuda_f16_default.cu",
	"cuda_experts_q81.cu",
	"cuda_hc.cu",
	"cuda_experts_gather.cu",
	"cuda_moe_d2r.cu",
	"cuda_attention_hmma.cu",
	"cuda_dense_d2r.cu",
}

func cudaMathSourceFromFS() string {
	chunks := make([]string, 0, len(cudaMathFiles))
	for _, name := range cudaMathFiles {
		data, err := cudaSourceFS.ReadFile(name)
		if err != nil {
			panic(fmt.Sprintf("ds4-min: embedded CUDA source %q: %v", name, err))
		}
		chunks = append(chunks, cudaSourceChunk(name, string(data)))
	}
	return strings.Join(chunks, "")
}

var cudaMathSource = cudaMathSourceFromFS()

// cudaSourceRoot is the directory NVRTC is told the embedded CUDA sources
// live in. It is absolute so the PTX line table names the same files from any
// working directory; NVRTC joins a relative name to the working directory.
const cudaSourceRoot = "/ds4-min/"

// cudaMathCompilationSource is the small NVRTC translation unit. The CUDA
// fragments are supplied as virtual headers below, so NVRTC can preserve their
// logical file names without requiring them to exist on the host filesystem.
func cudaMathCompilationSource() string {
	var b strings.Builder
	for _, name := range cudaMathFiles {
		fmt.Fprintf(&b, "#include %q\n", cudaSourceRoot+name)
	}
	return b.String()
}

func cudaMathHeaderOptions() []nvrtcx.Option {
	opts := make([]nvrtcx.Option, 0, len(cudaMathFiles))
	for _, name := range cudaMathFiles {
		data, err := cudaSourceFS.ReadFile(name)
		if err != nil {
			panic(fmt.Sprintf("ds4-min: embedded CUDA source %q: %v", name, err))
		}
		opts = append(opts, nvrtcx.WithHeader(cudaSourceRoot+name, string(data)))
	}
	return opts
}

// cudaMathCompileOptions are the NVRTC options for the math module on a
// device of compute capability major.minor.
func cudaMathCompileOptions(major, minor int) []nvrtcx.Option {
	return append(cudaMathHeaderOptions(),
		nvrtcx.WithName(cudaSourceRoot+"ds4_math.cu"),
		nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)),
		nvrtcx.WithFlag("--use_fast_math"),
		nvrtcx.WithFlag("--generate-line-info"),
		nvrtcx.WithFlag("--extra-device-vectorization"),
	)
}

const (
	cudaMemHostRegisterDeviceMap = 2
	cudaMaxRegisteredModelRange  = 8 << 20
)

type cudaModelRange struct {
	host       uintptr
	device     driver.DevicePtr
	bytes      uint
	fileOffset uint64
	view       []byte
}

type cudaModelTensor struct {
	name       string
	typeID     uint32
	dimensions []uint64
	device     driver.DevicePtr
	fileOffset uint64
	bytes      uint64
}

// cudaFoundation owns the resources needed to begin a CUDA model runtime.
// It deliberately does not implement sessionBackend until token evaluation is
// available on CUDA.
type cudaFoundation struct {
	closeOps                                *cudaFoundationCloseOperations
	closing, closeDrained, residentReleased bool
	stateIdentity                           *modelStateIdentityCache
	device                                  *cuda.Device
	workStream                              *cuda.Stream
	streamingExperts                        *cudaStreamingExperts
	file                                    *os.File
	model                                   []byte
	// releaseMapping drops each tensor's pages from the model mapping once
	// its artifact is built. See cudaReleaseMappedArtifacts.
	releaseMapping  int
	releasedBytes   uint64
	releasedTensors int
	// traffic counts the weight bytes each matvec launch reads, when a caller
	// has opened a ledger. See cuda_weight_traffic_linux.go.
	traffic *cudaWeightTraffic
	name    string
	major   int
	minor   int
	hmm     bool
	gguf    *gguf.File
	mapped  cudaModelRange

	// resident holds residentSet's spans in device memory. When it is set,
	// modelRange serves every range from it and none from the file mapping.
	// See loadResidentSet.
	resident           cudamem.Slice[byte]
	residentStageClose func() error
	residentClose      func() error
	residentOps        *cudaResidentOperations
	residentSet        cudaResidentSet

	// direct is a second read-only descriptor on the model opened O_DIRECT,
	// so artifact staging reads never enter the page cache. It is nil unless
	// DS4_CUDA_DIRECT_STAGING selects it, and it is closed for good the first
	// time a read returns an error that means this filesystem will not serve
	// direct reads. directAlign is the alignment those reads must honour.
	// See stagedBytes for why the two byte counters exist.
	direct        *os.File
	directAlign   int
	directBytes   uint64
	bufferedBytes uint64

	// disableLibraryGEMM selects the hand-written and per-token arms of every
	// projection that now goes through cuBLAS. It lives here rather than on the
	// runtime because cudaHCBatch reaches only as far as the foundation.
	disableLibraryGEMM bool

	probeTensor                    string
	probeHost                      uint16
	probeDevice                    uint16
	probeCols                      uint32
	probeRows                      uint32
	probeBytes                     uint64
	probeOffset                    uint64
	probePtr                       driver.DevicePtr
	matrixMod                      driver.Module
	quantQ80Mod                    driver.Module
	mathCompiled                   *nvrtcx.CompiledModule
	matrixFn                       driver.Function
	quantQ80Fn                     driver.Function
	quantQ80StridedFn              driver.Function
	quantQ81Fn                     driver.Function
	quantQ81BatchFn                driver.Function
	q81MatvecFn                    driver.Function
	q81MatvecAlignedFn             driver.Function
	q80QuantizedAlignedFn          driver.Function
	q80QuantizedBlockDiagAlignedFn driver.Function
	q81MatvecBatchFn               driver.Function
	q80QuantizedFn                 driver.Function
	q80QuantizedBatchFn            driver.Function
	q80QuantizedSmallFn            driver.Function
	q80QuantizedBlockDiagFn        driver.Function
	q80BlockDiagBatchFn            driver.Function
	rmsFn                          driver.Function
	rmsPairFn                      driver.Function
	f16Fn                          driver.Function
	f16DefaultFn                   driver.Function
	f16DefaultBatchFn              driver.Function
	f16PairOrderedFn               driver.Function
	f16PairOrderedBatchFn          driver.Function
	f32ToF16Fn                     driver.Function
	hcPreFn                        driver.Function
	hcHalfFn                       driver.Function
	hcSplitFn                      driver.Function
	hcCollapseFn                   driver.Function
	hcExpandFn                     driver.Function

	hcPostFn                        driver.Function
	rowFn                           driver.Function
	rowBatchFn                      driver.Function
	rawAttentionFn                  driver.Function
	rawAttentionPrepareFn           driver.Function
	rawAttentionEvalFn              driver.Function
	disableRawAttentionFused        bool
	routerHashFn                    driver.Function
	routerHashBatchFn               driver.Function
	routerBiasedFn                  driver.Function
	routerBiasedBatchFn             driver.Function
	swigluFn                        driver.Function
	swigluQuantQ80Fn                driver.Function
	swigluQuantQ81Fn                driver.Function
	sumVectorsFn                    driver.Function
	sumRoutedSharedFn               driver.Function
	sumRoutedSharedBatchFn          driver.Function
	iq2SelectedFn                   driver.Function
	q2KSelectedFn                   driver.Function
	quantQ8KFn                      driver.Function
	iq2SelectedQ8KFn                driver.Function
	q2KSelectedQ8KFn                driver.Function
	q4KSelectedQ8KFn                driver.Function
	iq2SelectedQ81Fn                driver.Function
	iq2SelectedQ81ScalarFn          driver.Function
	iq2SelectedQ81AlignedFn         driver.Function
	q2KSelectedQ81Fn                driver.Function
	q2KSelectedQ81ScalarFn          driver.Function
	q2KSelectedQ81AlignedFn         driver.Function
	moeGroupCountFn                 driver.Function
	moeGroupPrefixFn                driver.Function
	moeGroupFillFn                  driver.Function
	moeIDsHelperFn                  driver.Function
	iq2GatherQ81Fn                  driver.Function
	q2KGatherQ81Fn                  driver.Function
	quantMMQFn                      driver.Function
	quantMMQBlocksFn                driver.Function
	d2rMakeIdsSrcFn                 driver.Function
	d2rBuild32Fn                    driver.Function
	d2rBuild64Fn                    driver.Function
	gateupD2RFn                     driver.Function
	gateupD2RMidFn                  driver.Function
	downD2RFn                       driver.Function
	repackIQ2Fn                     driver.Function
	repackQ2KFn                     driver.Function
	disableD2RExperts               bool
	disableDenseD2R                 bool
	compressorPoolFn                driver.Function
	compressorAttentionFn           driver.Function
	compressorIndexerFn             driver.Function
	compressorPoolBatchFn           driver.Function
	compressorStateBatchFn          driver.Function
	compressorAttentionBatchFn      driver.Function
	compressorIndexerBatchFn        driver.Function
	mixedAttentionFn                driver.Function
	mixedAttentionPrepareFn         driver.Function
	mixedAttentionScoresFn          driver.Function
	mixedAttentionNormalizeFn       driver.Function
	mixedAttentionValuesFn          driver.Function
	mixedAttentionFusedFn           driver.Function
	mixedAttentionTileScoresFn      driver.Function
	mixedAttentionFinalizeFn        driver.Function
	mixedAttentionPrepareBatchFn    driver.Function
	mixedAttentionPrepareKVBatchFn  driver.Function
	headRMSRoPEBatchFn              driver.Function
	mixedAttentionTileScoresBatchFn driver.Function
	mixedAttentionFinalizeBatchFn   driver.Function
	mixedAttentionCommitRawFn       driver.Function
	disableMixedAttentionFused      bool
	disableCompressorBatch          bool
	disableAttnQKVD2R               bool
	disableRawAttentionBatch        bool
	disableChunkedArena             bool
	indexerQueryFn                  driver.Function
	indexerScoreFn                  driver.Function
	indexerSelectFn                 driver.Function
	fillU32Fn                       driver.Function
	tokentileDenseBuildFn           driver.Function
	tokentileUnionBuildFn           driver.Function
	tokentileRawMirrorFn            driver.Function
	tokentileCompMirrorFn           driver.Function
	tokentileHMMAFn                 driver.Function
	repackQ80Fn                     driver.Function
	denseQ8D2RFn                    driver.Function
	denseQ8D2RBlockDiagFn           driver.Function

	q4KPairGatherFn driver.Function
	q4KDownGatherFn driver.Function

	// denseQ8D2RLaunches counts ds4_dense_q8_d2r launches. A shape-envelope
	// arm that routes sites off this kernel is only an arm if the count
	// moves, so the count is the arm's own firing witness.
	denseQ8D2RLaunches uint64

	phaseMark func(name string, position, layer int) error
}

func (r *cudaFoundation) stream() *cuda.Stream {
	if r != nil && r.workStream != nil {
		return r.workStream
	}
	return r.device.DefaultStream()
}

func (r *cudaFoundation) markCUDAPhase(name string, position, layer int) error {
	// The host span is driven here rather than at each call site, and
	// before the hook check, so it does not depend on a profiler being
	// installed.
	markPhaseSpan(name)
	if r == nil || r.phaseMark == nil {
		return nil
	}
	return r.phaseMark(name, position, layer)
}

func newCUDAFoundation(path string) (_ *cudaFoundation, err error) {
	return newCUDAFoundationWithOptions(path, cudaRuntimeOptions{})
}

func newCUDAFoundationWithOptions(path string, options cudaRuntimeOptions) (_ *cudaFoundation, err error) {
	return newCUDAFoundationWithOperations(path, options, defaultCUDAFoundationConstructorOperations())
}

// cudaReleaseMappedArtifacts drops a tensor's pages from the model mapping
// once its repacked artifact is on the device, from
// DS4_CUDA_RELEASE_MAPPED_ARTIFACTS:
//
//	1  madvise(MADV_DONTNEED) -- drops this process's page-table entries
//	2  and posix_fadvise(POSIX_FADV_DONTNEED) -- also drops the page cache
//
// It tests the one claim standing between the two-layer expert budget and
// full residency. cudaMoEArtifactLayers records that the artifacts and the
// mapping spend a single budget between them, that the limit is a working
// set rather than a footprint, and that repacking everything is close to
// memory-neutral -- but only once the mapping is released, which no failing
// run had done. Every run that collapsed held both copies.
//
// Mode 1 is MEASURED AND ESTABLISHES NOTHING about either correctness or
// throughput. Two sweeps, one draw per cell, dense=43 throughout:
//
//	moe  release   sweep 1            sweep 2
//	  2      off   559.92 ok
//	  8      off   579.02 ok
//	  8       on   576.97 ok          173.53 ok, then 496.83 ok
//	 16       on                      101.62 WRONG DIGEST
//	 21       on                       41.72 ok
//	 32       on                       56.05 ok
//	 43      off   128.41 NON-FINITE   117.41 ok
//	 43       on   458.81 ok           390.45 WRONG DIGEST
//
// The 43-layer cells flipped between sweeps in BOTH directions. Sweep 1 read
// as a clean dissociation -- off corrupts, on is correct -- and sweep 2
// reversed it exactly. So this gate does not fix the high-layer corruption,
// and it is not shown to fail to either: both arms fail intermittently. The
// 8/on cell drew 173.53 and 496.83 on the same configuration, 2.9x apart and
// both correct, so the throughput column is not a rate anywhere in it.
//
// The corruption signature is the documented one: a wrong logit digest with
// the continuation tokens still [15342, 201]. The digest is the gate; the
// tokens are not.
//
// Anything measured in this regime needs repeated alternating draws per
// cell, a median rather than a mean given the low mode, and the digest as
// the acceptance test. One draw per cell produced a confident finding and
// its exact reversal within the hour.
//
// The one effect that looks like signal and is still one draw per arm:
// building 43 layers took 84.00s with release off and 18.87s with it on.
//
// Mode 2 exists because mode 1 shrinks the wrong working set if the binding
// resource is the page cache: MADV_DONTNEED drops page-table entries and
// leaves the model's bytes cached, and on an integrated part the cache
// shares one pool with device memory, so it frees nothing the artifacts can
// spend. Whether that is what binds is unmeasured.
//
// Both modes are safe by construction rather than by argument. The model is
// mapped PROT_READ MAP_SHARED, so dropping a range is transparent: a reader
// that still reaches a released tensor re-faults the same bytes from the
// same file. It gets the right answer slowly, never a wrong one. That is why
// this can be measured against the logit digest without first proving the
// artifact covers every reader.
func cudaReleaseMappedArtifacts() int {
	switch os.Getenv("DS4_CUDA_RELEASE_MAPPED_ARTIFACTS") {
	case "1":
		return 1
	case "2":
		return 2
	}
	// The C-compatible artifact mode replaces the mapped source tensors with
	// their load-time layouts.  Releasing both the mapping and its page cache
	// is part of that contract; without it we keep two copies of the weights.
	if os.Getenv("DS4_CUDA_LOAD_TIME_ARTIFACTS") == "1" {
		return 2
	}
	// So does the resident model, where the artifacts are the only copy of
	// the experts any kernel reads. See loadResidentSet.
	if cudaResidentModel() {
		return 2
	}
	return 0
}

// cudaWarmModelDefault is whether newCUDAFoundation faults the whole model
// mapping in before the first kernel when DS4_CUDA_WARM_MODEL says nothing.
// Flipping this one line stops warming by default; see warmModel for what it
// is for and what it costs.
const cudaWarmModelDefault = true

func cudaWarmModel() bool {
	switch os.Getenv("DS4_CUDA_WARM_MODEL") {
	case "0":
		return false
	case "1":
		return true
	}
	return cudaWarmModelDefault
}

// cudaWarmModelSelected gives the CLI an explicit override while preserving
// the existing environment/default behavior when the flag is absent.
func cudaWarmModelSelected(set, value bool) bool {
	if set {
		return value
	}
	return cudaWarmModel()
}

// cudaDirectStagingDefault is whether artifact staging reads bypass the page
// cache when DS4_CUDA_DIRECT_STAGING says nothing. Flipping this one line
// ships direct staging; see openDirectModel.
const cudaDirectStagingDefault = false

func cudaDirectStaging() bool {
	switch os.Getenv("DS4_CUDA_DIRECT_STAGING") {
	case "0":
		return false
	case "1":
		return true
	}
	return cudaDirectStagingDefault
}

// openDirectModel opens a second read-only descriptor on the model with
// O_DIRECT, so the bytes artifact staging reads never enter the page cache.
// The C oracle reads its artifact sources this way and caches nothing, where
// buffered staging pulls all 72.56 GiB of expert and dense source through the
// cache -- which on an integrated part is the same physical pool the artifacts
// are being allocated from.
//
// O_DIRECT is a property of the filesystem, not of the kernel, so every
// failure here is expected and none is fatal: a nil descriptor means staging
// reads through the buffered one, exactly as it does today.
func openDirectModel(path string, info os.FileInfo) (*os.File, int) {
	align := 512
	if st, ok := info.Sys().(*syscall.Stat_t); ok && int(st.Blksize) > align {
		align = int(st.Blksize)
	}
	if align > cudaDirectAlign {
		fmt.Fprintf(os.Stderr, "ds4: direct staging off: block size %d exceeds stage alignment %d\n", align, cudaDirectAlign)
		return nil, 0
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_DIRECT, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ds4: direct staging off: open O_DIRECT: %v\n", err)
		return nil, 0
	}
	fmt.Fprintf(os.Stderr, "ds4: direct staging on: align=%d\n", align)
	return os.NewFile(uintptr(fd), path), align
}

// posixFadviseDontNeed drops the page cache for the file's byte range. There
// is no syscall wrapper for fadvise64, only the trap number.
func posixFadviseDontNeed(fd int, offset, length uint64) error {
	const posixFadvDontNeed = 4
	if _, _, e := syscall.Syscall6(syscall.SYS_FADVISE64, uintptr(fd),
		uintptr(offset), uintptr(length), posixFadvDontNeed, 0, 0); e != 0 {
		return e
	}
	return nil
}

// releaseMapped drops the model mapping's pages for t. The range is rounded
// INWARD to whole pages: a partial page at either end also holds a
// neighbouring tensor's bytes, and that neighbour may have no artifact.
func (r *cudaFoundation) releaseMapped(t *cudaModelTensor) error {
	if r.releaseMapping == 0 || t.bytes == 0 {
		return nil
	}
	page := uint64(syscall.Getpagesize())
	lo := (t.fileOffset + page - 1) &^ (page - 1)
	hi := (t.fileOffset + t.bytes) &^ (page - 1)
	if hi <= lo || hi > uint64(len(r.model)) {
		return nil
	}
	if err := syscall.Madvise(r.model[lo:hi], syscall.MADV_DONTNEED); err != nil {
		return fmt.Errorf("cuda: release %s from mapping: %w", t.name, err)
	}
	if r.releaseMapping >= 2 {
		if err := posixFadviseDontNeed(int(r.file.Fd()), lo, hi-lo); err != nil {
			return fmt.Errorf("cuda: drop %s from page cache: %w", t.name, err)
		}
	}
	r.releasedBytes += hi - lo
	r.releasedTensors++
	return nil
}

// releasedMapping reports what releaseMapped actually dropped. A sweep that
// compares a release arm against a matched control needs this: without it an
// unplumbed gate makes the two arms byte-identical, and the run reads as
// "releasing does not help" rather than "releasing did not happen".
func (r *cudaFoundation) releasedMapping() (tensors int, bytes uint64) {
	return r.releasedTensors, r.releasedBytes
}

// stagedBytes reports how many artifact source bytes were read through the
// O_DIRECT descriptor and how many through the buffered one. An arm that
// measures direct staging needs this for the same reason releasedMapping
// exists: a descriptor that never engages -- the gate unset, the open refused,
// or the first read falling back -- makes the two arms byte-identical, and the
// run then reads as "O_DIRECT does not help" when the truth is "O_DIRECT never
// ran".
func (r *cudaFoundation) stagedBytes() (direct, buffered uint64) {
	return r.directBytes, r.bufferedBytes
}

// warmModel makes the whole mapped model resident before any kernel reads it.
//
// Under host page tables the kernels address the file-backed mapping directly,
// so a kernel argument can name a page the process has never touched. Resolving
// that first reference from the device does not reliably deliver the file
// contents: the first evaluation then reads wrong bytes for whichever tensors
// it happens to reach first, and every later evaluation is correct because the
// first one left the pages present. Faulting the span in from the host removes
// the device-side first reference. It costs the same residency wait the pinned
// implementations pay at load time; a warm page cache makes it nearly free.
//
// It is gated because it is one of the three host-memory differences between
// this runtime and the C oracle, which warms nothing at all: on an integrated
// part the touch leaves all 80.76 GiB in a page cache that shares one physical
// pool with device memory. DS4_CUDA_WARM_MODEL=0 skips it, =1 forces it, and
// cudaWarmModelDefault decides when it says nothing.
func (r *cudaFoundation) warmModel() error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if !r.hmm {
		return nil
	}
	defer beginOp("model.warm").end()
	if err := syscall.Madvise(r.model, syscall.MADV_WILLNEED); err != nil && !errors.Is(err, syscall.EINVAL) {
		return fmt.Errorf("cuda: advise model mapping: %w", err)
	}
	page := syscall.Getpagesize()
	var sink byte
	for i := 0; i < len(r.model); i += page {
		sink ^= r.model[i]
	}
	runtime.KeepAlive(sink)
	return nil
}

// cudaSupportsHostPageTables reports whether ordinary system allocations use
// the same virtual addresses on the CPU and GPU. All four properties are
// required: accepting a host pointer based only on UVA is not sufficient.
func cudaSupportsHostPageTables(dev *cuda.Device) (bool, error) {
	attrs := [...]driver.DeviceAttribute{
		driver.DevAttrIntegrated,
		driver.DevAttrUnifiedAddressing,
		driver.DevAttrPageableMemoryAccess,
		driver.DevAttrPageableMemoryAccessUsesHostPageTables,
	}
	for _, attr := range attrs {
		value, err := dev.Attribute(attr)
		if err != nil {
			return false, fmt.Errorf("cuda: query HMM capability %v: %w", attr, err)
		}
		if value == 0 {
			return false, nil
		}
	}
	return true, nil
}

// modelRange returns a device view of bytes in the stable GGUF mapping. With a
// resident set it is the device copy; see residentRange. Otherwise, on an
// integrated device using the host page tables, it is the mmap address itself
// and requires neither registration nor a CUDA allocation. Other devices use
// the bounded registered fallback.
func (r *cudaFoundation) modelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if err := r.checkOpen(); err != nil {
		return 0, err
	}
	if length == 0 {
		return 0, fmt.Errorf("cuda: empty model range")
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.resident.Len() > 0 {
		return r.residentRange(fileOffset, length)
	}
	if r.hmm {
		return driver.DevicePtr(uintptr(unsafe.Pointer(&r.model[fileOffset]))), nil
	}
	return r.mapModelRange(fileOffset, length)
}

func (r *cudaFoundation) tensor(name string) (cudaModelTensor, error) {
	if r.gguf == nil {
		return cudaModelTensor{}, fmt.Errorf("cuda: model is closed")
	}
	for _, tensor := range r.gguf.Tensors {
		if tensor.Name != name {
			continue
		}
		bytes, err := tensor.Bytes()
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: size tensor %q: %w", name, err)
		}
		offset := r.gguf.DataOffset + tensor.Offset
		ptr, err := r.modelRange(offset, bytes)
		if err != nil {
			return cudaModelTensor{}, fmt.Errorf("cuda: bind tensor %q: %w", name, err)
		}
		return cudaModelTensor{name: name, typeID: tensor.TypeID, dimensions: append([]uint64(nil), tensor.Dimensions...), device: ptr, fileOffset: offset, bytes: bytes}, nil
	}
	return cudaModelTensor{}, fmt.Errorf("cuda: model has no tensor %q", name)
}

func (r *cudaFoundation) probe() error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	compiled, err := compileCUDAProbe(r.device, "ds4_probe.cu", cudaProbeCompiler())
	if err != nil {
		return fmt.Errorf("cuda: compile probe: %w", err)
	}
	return r.withContext(func() (err error) {
		cleanup := r.probeResources()
		defer func() { err = finishBackendCleanup(err, cleanup) }()
		mod, err := loadCUDAProbe(compiled, r.device, cudaProbeCompiler())
		cleanup.module = mod
		if err != nil {
			return fmt.Errorf("cuda: load probe: %w", err)
		}
		fn, err := mod.Function("ds4_probe")
		if err != nil {
			return fmt.Errorf("cuda: find probe: %w", err)
		}
		buf, err := cudamem.Alloc[float32](r.device, 1)
		cleanup.buffer = buf.Free
		if err != nil {
			return fmt.Errorf("cuda: allocate probe: %w", err)
		}
		if err := cudalaunch.Kernel(fn).Launch(buf); err != nil {
			return fmt.Errorf("cuda: launch probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize probe: %w", err)
		}
		var got [1]float32
		if err := buf.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read probe: %w", err)
		}
		if got[0] != 4 {
			return fmt.Errorf("cuda: probe result %g, want 4", got[0])
		}
		return nil
	})
}

// mapModelRange makes a small, page-aligned part of the GGUF directly
// addressable by CUDA. It uses a private writable mapping because GB10 does not
// advertise CUDA's read-only host-registration extension. Device writes cannot
// modify the model file. Large ranges are rejected: pinning the entire 86 GB
// model would deprive the host of pageable memory.
func (r *cudaFoundation) mapModelRange(fileOffset, length uint64) (driver.DevicePtr, error) {
	if err := r.checkOpen(); err != nil {
		return 0, err
	}
	if length == 0 || length > cudaMaxRegisteredModelRange {
		return 0, fmt.Errorf("cuda: model range length %d exceeds limit %d", length, cudaMaxRegisteredModelRange)
	}
	if fileOffset > uint64(len(r.model)) || length > uint64(len(r.model))-fileOffset {
		return 0, fmt.Errorf("cuda: model range [%d,%d) outside mapping", fileOffset, fileOffset+length)
	}
	if r.mapped.host != 0 {
		return 0, fmt.Errorf("cuda: model range already mapped")
	}
	page := uint64(syscall.Getpagesize())
	start := fileOffset &^ (page - 1)
	end := (fileOffset + length + page - 1) &^ (page - 1)
	if end > uint64(len(r.model)) {
		end = uint64(len(r.model))
	}
	bytes := uint(end - start)
	view, err := syscall.Mmap(int(r.file.Fd()), int64(start), int(bytes), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE)
	if err != nil {
		return 0, fmt.Errorf("cuda: map private model range: %w", err)
	}
	host := uintptr(unsafe.Pointer(&view[0]))
	err = r.withContext(func() error {
		if err := driver.MemHostRegister(host, bytes, cudaMemHostRegisterDeviceMap); err != nil {
			return fmt.Errorf("register model range: %w", err)
		}
		device, err := driver.MemHostGetDevicePointer(host, 0)
		if err != nil {
			_ = driver.MemHostUnregister(host)
			return fmt.Errorf("get model device pointer: %w", err)
		}
		r.mapped = cudaModelRange{host: host, device: device, bytes: bytes, fileOffset: start, view: view}
		return nil
	})
	if err != nil {
		_ = syscall.Munmap(view)
		return 0, fmt.Errorf("cuda: map model range: %w", err)
	}
	return r.mapped.device + driver.DevicePtr(fileOffset-start), nil
}

func (r *cudaFoundation) probeModel(g *gguf.File) error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if err := validateSwiGLUClamp(g, cudaLayers); err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	var tensor *gguf.Tensor
	for i := range g.Tensors {
		if g.Tensors[i].Name == "blk.0.attn_kv.weight" {
			tensor = &g.Tensors[i]
			break
		}
	}
	if tensor == nil {
		return fmt.Errorf("cuda: model has no tensor %q", "blk.0.attn_kv.weight")
	}
	if tensor.TypeID != 8 || len(tensor.Dimensions) != 2 || tensor.Dimensions[0]%32 != 0 {
		return fmt.Errorf("cuda: tensor %q is not a Q8_0 matrix", tensor.Name)
	}
	tensorBytes, err := tensor.Bytes()
	if err != nil {
		return fmt.Errorf("cuda: size tensor %q: %w", tensor.Name, err)
	}
	if tensorBytes > cudaMaxRegisteredModelRange {
		return fmt.Errorf("cuda: Q8_0 tensor %q is %d bytes, limit %d", tensor.Name, tensorBytes, cudaMaxRegisteredModelRange)
	}
	offset := g.DataOffset + tensor.Offset
	want := uint16(r.model[offset]) | uint16(r.model[offset+1])<<8
	ptr, err := r.modelRange(offset, tensorBytes)
	if err != nil {
		return fmt.Errorf("cuda: bind tensor %q: %w", tensor.Name, err)
	}
	compiled, err := compileCUDAProbe(r.device, "ds4_model_probe.cu", cudaProbeCompiler())
	if err != nil {
		return fmt.Errorf("cuda: compile model probe: %w", err)
	}
	return r.withContext(func() (err error) {
		cleanup := r.probeResources()
		defer func() { err = finishBackendCleanup(err, cleanup) }()
		mod, err := loadCUDAProbe(compiled, r.device, cudaProbeCompiler())
		cleanup.module = mod
		if err != nil {
			return fmt.Errorf("cuda: load model probe: %w", err)
		}
		fn, err := mod.Function("ds4_read_u16")
		if err != nil {
			return fmt.Errorf("cuda: find model probe: %w", err)
		}
		out, err := cudamem.Alloc[uint32](r.device, 1)
		cleanup.buffer = out.Free
		if err != nil {
			return fmt.Errorf("cuda: allocate model probe: %w", err)
		}
		if err := cudalaunch.Kernel(fn).Launch(ptr, out); err != nil {
			return fmt.Errorf("cuda: launch model probe: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize model probe: %w", err)
		}
		var got [1]uint32
		if err := out.CopyToHost(got[:]); err != nil {
			return fmt.Errorf("cuda: read model probe: %w", err)
		}
		if got[0] != uint32(want) {
			return fmt.Errorf("cuda: model probe result %#x, want %#x", got[0], want)
		}
		r.probeTensor = tensor.Name
		r.probeHost = want
		r.probeDevice = uint16(got[0])
		r.probeCols = uint32(tensor.Dimensions[0])
		r.probeRows = uint32(tensor.Dimensions[1])
		r.probeBytes = tensorBytes
		r.probeOffset = offset
		r.probePtr = ptr
		return nil
	})
}

// q80Matvec multiplies the registered Q8_0 matrix by x. The caller owns both
// x and out; the foundation owns only the bounded registered model tensor.
func (r *cudaFoundation) q80Matvec(x, out cudamem.Slice[float32]) error {
	if r.probePtr == 0 || r.probeCols == 0 || r.probeRows == 0 {
		return fmt.Errorf("cuda: Q8_0 matrix is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != int(r.probeRows) {
		return fmt.Errorf("cuda: Q8_0 matrix input/output lengths %d/%d, want %d/%d", x.Len(), out.Len(), r.probeCols, r.probeRows)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(r.probeRows, 1, 1).Block(256, 1, 1).Launch(r.probePtr, x, out, r.probeCols, r.probeRows); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 matrix: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 matrix: %w", err)
		}
		return nil
	})
}

// encodeQ80Matvec appends a matrix-vector product to the foundation's CUDA
// stream. It does not synchronize or copy results to the host.
func (r *cudaFoundation) encodeQ80Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 ||
		uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid Q8_0 tensor or input", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.matrixFn).Grid(uint32(t.dimensions[1]), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeRMS appends grouped RMS normalization to the foundation's CUDA
// stream. Weight must be an F32 vector when weighted is true.
func (r *cudaFoundation) encodeRMS(x cudamem.Slice[float32], weight cudaModelTensor, out cudamem.Slice[float32], groups int, weighted bool) error {
	if x.Len() == 0 || out.Len() != x.Len() || groups <= 0 || x.Len()%groups != 0 ||
		(weighted && (weight.typeID != 0 || len(weight.dimensions) != 1 || weight.dimensions[0] != uint64(x.Len()/groups))) {
		return fmt.Errorf("cuda: RMS: invalid input, weight, or groups")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var weightPtr driver.DevicePtr
		if weighted {
			weightPtr = weight.device
		}
		weightedArg := uint32(0)
		if weighted {
			weightedArg = 1
		}
		width := x.Len() / groups
		threads := 256
		if width >= 1024 {
			threads = 1024
		} else if width == 128 {
			threads = 128
		}
		if err := cudalaunch.Kernel(r.rmsFn).Grid(uint32(groups), 1, 1).Block(uint32(threads), 1, 1).On(r.stream()).Launch(
			x, weightPtr, out, uint32(width), uint32(groups), weightedArg); err != nil {
			return fmt.Errorf("cuda: launch RMS: %w", err)
		}
		return nil
	})
}

// encodeRMSPair appends concurrent RMS normalization of two independent tensors
// (e.g. qLora and kvRaw) to the foundation's CUDA stream using a Grid(2) launch.
func (r *cudaFoundation) encodeRMSPair(
	x0 cudamem.Slice[float32], weight0 cudaModelTensor, out0 cudamem.Slice[float32],
	x1 cudamem.Slice[float32], weight1 cudaModelTensor, out1 cudamem.Slice[float32],
) error {
	weighted0 := weight0.device != 0
	if x0.Len() == 0 || out0.Len() != x0.Len() ||
		(weighted0 && (weight0.typeID != 0 || len(weight0.dimensions) != 1 || weight0.dimensions[0] != uint64(x0.Len()))) {
		return fmt.Errorf("cuda: RMS pair: invalid input or weight 0")
	}
	weighted1 := weight1.device != 0
	if x1.Len() == 0 || out1.Len() != x1.Len() ||
		(weighted1 && (weight1.typeID != 0 || len(weight1.dimensions) != 1 || weight1.dimensions[0] != uint64(x1.Len()))) {
		return fmt.Errorf("cuda: RMS pair: invalid input or weight 1")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var weight0Ptr, weight1Ptr driver.DevicePtr
		if weighted0 {
			weight0Ptr = weight0.device
		}
		if weighted1 {
			weight1Ptr = weight1.device
		}
		if err := cudalaunch.Kernel(r.rmsPairFn).Grid(2, 1, 1).Block(1024, 1, 1).On(r.stream()).Launch(
			x0, weight0Ptr, out0, uint32(x0.Len()),
			x1, weight1Ptr, out1, uint32(x1.Len()),
		); err != nil {
			return fmt.Errorf("cuda: launch RMS pair: %w", err)
		}
		return nil
	})
}

// encodeF16Matvec appends an F16 matrix-vector product to the default stream.
func (r *cudaFoundation) encodeF16Matvec(t cudaModelTensor, x, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(x.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("cuda: matvec %q: invalid F16 tensor or input", t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.f16Fn).Grid(uint32(t.dimensions[1]), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, x, out, uint32(t.dimensions[0]), uint32(t.dimensions[1])); err != nil {
			return fmt.Errorf("cuda: launch matvec %q: %w", t.name, err)
		}
		return nil
	})
}

// encodeHCPre appends the DS4 hyper-connection input transform. Out contains
// the current stream followed by its attention-normalized form.
func (r *cudaFoundation) encodeHCPre(state cudamem.Slice[float32], hcFn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32]) error {
	return r.encodeHCPrePrecision(state, hcFn, scale, base, norm, out, false)
}

func (r *cudaFoundation) encodeHCPrePrecision(state cudamem.Slice[float32], hcFn, scale, base, norm cudaModelTensor, out cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || out.Len() != 2*4096 || hcFn.typeID != 1 || len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 4*4096 || hcFn.dimensions[1] != 24 ||
		scale.typeID != 0 || base.typeID != 0 || norm.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 ||
		len(base.dimensions) != 1 || base.dimensions[0] != 24 || len(norm.dimensions) != 1 || norm.dimensions[0] != 4096 {
		return fmt.Errorf("cuda: HC pre: invalid tensor profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPreFn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			state, hcFn.device, scale.device, base.device, norm.device, out, half); err != nil {
			return fmt.Errorf("cuda: launch HC pre: %w", err)
		}
		return nil
	})
}

// cudaKernels names every kernel the layer loop needs, in load order.
// encodeHCPost appends the DS4 hyper-connection residual update, folding one
// block output back into the four-stream state.
func (r *cudaFoundation) encodeHCPost(state, block cudamem.Slice[float32], hcFn, scale, base cudaModelTensor, out cudamem.Slice[float32]) error {
	return r.encodeHCPostPrecision(state, block, hcFn, scale, base, out, false)
}

func (r *cudaFoundation) encodeHCPostPrecision(state, block cudamem.Slice[float32], hcFn, scale, base cudaModelTensor, out cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || block.Len() != 4096 || out.Len() != state.Len() ||
		hcFn.typeID != 1 || len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 4*4096 || hcFn.dimensions[1] != 24 ||
		scale.typeID != 0 || len(scale.dimensions) != 1 || scale.dimensions[0] != 3 ||
		base.typeID != 0 || len(base.dimensions) != 1 || base.dimensions[0] != 24 {
		return fmt.Errorf("cuda: HC post: invalid tensor profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPostFn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			state, block, hcFn.device, scale.device, base.device, out, half); err != nil {
			return fmt.Errorf("cuda: launch HC post: %w", err)
		}
		return nil
	})
}

// encodeOutputHead appends the final hyper-connection transform and the logits
// projection. Mixed receives the flattened stream followed by its normalized
// form; logits receives one score per vocabulary entry.
//
// The output head reuses the layer input transform: both collapse the four
// streams through the first four projection rows and normalize the result. The
// head's own gate and base vectors are shorter than a layer's, so its tensor
// profile is checked here rather than in encodeHCPre.
func (r *cudaFoundation) encodeOutputHead(state cudamem.Slice[float32], out cudaOutput, mixed, logits cudamem.Slice[float32]) error {
	if err := r.encodeOutputTransform(state, out, mixed, logits, false); err != nil {
		return err
	}
	return r.encodeQ80Matvec(out.weight, mixed.Slice(4096, 8192), logits)
}

func (r *cudaFoundation) encodeOutputTransform(state cudamem.Slice[float32], out cudaOutput, mixed, logits cudamem.Slice[float32], halfInput bool) error {
	half := uint32(0)
	if halfInput {
		half = 1
	}
	if state.Len() != 4*4096 || mixed.Len() != 2*4096 ||
		out.hcFn.typeID != 1 || len(out.hcFn.dimensions) != 2 || out.hcFn.dimensions[0] != 4*4096 || out.hcFn.dimensions[1] < 4 ||
		out.hcScale.typeID != 0 || len(out.hcScale.dimensions) != 1 || out.hcScale.dimensions[0] < 1 ||
		out.hcBase.typeID != 0 || len(out.hcBase.dimensions) != 1 || out.hcBase.dimensions[0] < 4 ||
		out.norm.typeID != 0 || len(out.norm.dimensions) != 1 || out.norm.dimensions[0] != 4096 ||
		out.weight.typeID != 8 || len(out.weight.dimensions) != 2 || out.weight.dimensions[0] != 4096 ||
		uint64(logits.Len()) != out.weight.dimensions[1] {
		return fmt.Errorf("cuda: output head: invalid tensor profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.hcPreFn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			state, out.hcFn.device, out.hcScale.device, out.hcBase.device, out.norm.device, mixed, half); err != nil {
			return fmt.Errorf("cuda: launch output transform: %w", err)
		}
		return nil
	})
}

// encodeEmbeddingRow appends the widening of one F16 embedding row.
func (r *cudaFoundation) encodeEmbeddingRow(t cudaModelTensor, row uint32, out cudamem.Slice[float32]) error {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(out.Len()) != t.dimensions[0] || uint64(row) >= t.dimensions[1] {
		return fmt.Errorf("cuda: embedding row %d of %q: invalid tensor or row", row, t.name)
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		cols := uint32(t.dimensions[0])
		if err := cudalaunch.Kernel(r.rowFn).Grid((cols+255)/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, out, row, cols); err != nil {
			return fmt.Errorf("cuda: launch embedding row: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeEmbeddingRows(t cudaModelTensor, rows cudamem.Slice[uint32], n int, out cudamem.Slice[float32]) error {
	if n <= 0 || rows.Len() < n || t.typeID != 1 || len(t.dimensions) != 2 || out.Len() < n*4*int(t.dimensions[0]) {
		return fmt.Errorf("cuda: embedding rows: invalid tensor or rows")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		cols := uint32(t.dimensions[0])
		blocks := (cols + 255) / 256
		if err := cudalaunch.Kernel(r.rowBatchFn).Grid(uint32(n*4)*blocks, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			t.device, rows.Slice(0, n), out.Slice(0, n*4*int(cols)), uint32(n), cols); err != nil {
			return fmt.Errorf("cuda: launch embedding rows: %w", err)
		}
		return nil
	})
}

// encodeRawAttention appends the uncompressed rotary attention used by layers
// 0 and 1. Out holds the rotated queries, the stored key-value row, and the
// inverse-rotated head outputs, in that order.
func (r *cudaFoundation) encodeRawAttention(q, kv cudamem.Slice[float32], sinks cudaModelTensor, cache, scores, out cudamem.Slice[float32], position int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		cache.Len() == 0 || cache.Len()%512 != 0 || scores.Len() != 64*(cache.Len()/512) ||
		out.Len() != 32768+512+32768 || position < 0 {
		return fmt.Errorf("cuda: raw attention: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if r.rawAttentionPrepareFn != 0 && r.rawAttentionEvalFn != 0 && !r.disableRawAttentionFused {
			if err := cudalaunch.Kernel(r.rawAttentionPrepareFn).Grid(64, 1, 1).Block(512, 1, 1).On(r.stream()).Launch(
				q, kv, cache, out, uint32(position), uint32(cache.Len()/512)); err != nil {
				return fmt.Errorf("cuda: launch raw attention prepare: %w", err)
			}
			shmem := uint32((cache.Len() / 512) * 4)
			if err := cudalaunch.Kernel(r.rawAttentionEvalFn).Grid(64, 1, 1).Block(512, 1, 1).SharedMem(shmem).On(r.stream()).Launch(
				sinks.device, cache, scores, out, uint32(position), uint32(cache.Len()/512)); err != nil {
				return fmt.Errorf("cuda: launch raw attention eval: %w", err)
			}
			return nil
		}
		if err := cudalaunch.Kernel(r.rawAttentionFn).Grid(1, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(
			q, kv, sinks.device, cache, scores, out, uint32(position), uint32(cache.Len()/512)); err != nil {
			return fmt.Errorf("cuda: launch raw attention: %w", err)
		}
		return nil
	})
}

// encodeRouterHash appends the dense-layer expert selection, which reads the
// token's row of the identity map.
func (r *cudaFoundation) encodeRouterHash(logits cudamem.Slice[float32], table cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32], token uint32) error {
	if logits.Len() != 256 || scores.Len() != 256 || selected.Len() != 6 || weights.Len() != 6 ||
		table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 || uint64(token) >= table.dimensions[1] {
		return fmt.Errorf("cuda: router hash: invalid tensor profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerHashFn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			logits, table.device, scores, selected, weights, token); err != nil {
			return fmt.Errorf("cuda: launch router hash: %w", err)
		}
		return nil
	})
}

// encodeRouterBiased appends the routed-layer expert selection. Bias picks the
// six experts; their unbiased scores set the mixture weights.
func (r *cudaFoundation) encodeRouterBiased(logits cudamem.Slice[float32], bias cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32]) error {
	if logits.Len() != 256 || scores.Len() != 256 || selected.Len() != 6 || weights.Len() != 6 ||
		bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return fmt.Errorf("cuda: router biased: invalid tensor profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerBiasedFn).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			logits, bias.device, scores, selected, weights); err != nil {
			return fmt.Errorf("cuda: launch router biased: %w", err)
		}
		return nil
	})
}

// encodeRouterHashBatch appends the dense-layer expert selection across batch tokens.
func (r *cudaFoundation) encodeRouterHashBatch(logits cudamem.Slice[float32], table cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32], tokens cudamem.Slice[uint32], batch int) error {
	if batch <= 0 || logits.Len() < batch*256 || scores.Len() < batch*256 || selected.Len() < batch*6 || weights.Len() < batch*6 || tokens.Len() < batch ||
		table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 {
		return fmt.Errorf("cuda: router hash batch: invalid profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerHashBatchFn).Grid(uint32(batch), 1, 1).Block(1, 1, 1).On(r.stream()).Launch(
			logits, table.device, scores, selected, weights, tokens, uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch router hash batch: %w", err)
		}
		return nil
	})
}

// encodeRouterBiasedBatch appends the routed-layer expert selection across batch tokens.
func (r *cudaFoundation) encodeRouterBiasedBatch(logits cudamem.Slice[float32], bias cudaModelTensor, scores cudamem.Slice[float32], selected cudamem.Slice[int32], weights cudamem.Slice[float32], batch int) error {
	if batch <= 0 || logits.Len() < batch*256 || scores.Len() < batch*256 || selected.Len() < batch*6 || weights.Len() < batch*6 ||
		bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return fmt.Errorf("cuda: router biased batch: invalid profile")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.routerBiasedBatchFn).Grid(uint32(batch), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			logits, bias.device, scores, selected, weights, uint32(batch)); err != nil {
			return fmt.Errorf("cuda: launch router biased batch: %w", err)
		}
		return nil
	})
}

// encodeSwiGLU appends the clamped SwiGLU activation scaled by one weight.
func (r *cudaFoundation) encodeSwiGLU(gate, up, out cudamem.Slice[float32], weight float32) error {
	return r.encodeSwiGLUSlots(gate, up, cudamem.Slice[float32]{}, out, 1, weight)
}

// encodeSwiGLUSlots appends the clamped SwiGLU activation over slots adjacent
// expert lanes. A non-empty weights vector supplies one weight per slot and
// keeps the mixture weights on the device.
func (r *cudaFoundation) encodeSwiGLUSlots(gate, up, weights, out cudamem.Slice[float32], slots int, weight float32) error {
	if slots <= 0 || gate.Len() == 0 || gate.Len()%slots != 0 || up.Len() != gate.Len() || out.Len() != gate.Len() ||
		(weights.Len() != 0 && weights.Len() != slots) {
		return fmt.Errorf("cuda: SwiGLU: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		n := uint32(gate.Len())
		if err := cudalaunch.Kernel(r.swigluFn).Grid((n+255)/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			gate, up, out, n/uint32(slots), uint32(slots), weight, weights); err != nil {
			return fmt.Errorf("cuda: launch SwiGLU: %w", err)
		}
		return nil
	})
}

// encodeSumVectors appends a reduction across count adjacent vectors.
func (r *cudaFoundation) encodeSumVectors(in, out cudamem.Slice[float32], count int) error {
	if count <= 0 || in.Len() == 0 || in.Len()%count != 0 || out.Len() != in.Len()/count {
		return fmt.Errorf("cuda: sum vectors: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		width := uint32(out.Len())
		if err := cudalaunch.Kernel(r.sumVectorsFn).Grid((width+255)/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			in, out, width, uint32(count)); err != nil {
			return fmt.Errorf("cuda: launch sum vectors: %w", err)
		}
		return nil
	})
}

// encodeSwiGLUQuantQ81 fuses clamped SwiGLU activation and Q8_1 quantization.
func (r *cudaFoundation) encodeSwiGLUQuantQ81(gate, up, weights cudamem.Slice[float32], out cudamem.Slice[byte], width, slots int) error {
	if width <= 0 || width%32 != 0 || slots <= 0 || gate.Len() != width*slots || up.Len() != width*slots || weights.Len() != slots || out.Len() != (width*slots/32)*36 {
		return fmt.Errorf("cuda: SwiGLU quant Q8_1: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := uint32(width * slots / 32)
		if err := cudalaunch.Kernel(r.swigluQuantQ81Fn).Grid(blocks, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(
			gate, up, weights, out, uint32(width)); err != nil {
			return fmt.Errorf("cuda: launch SwiGLU quant Q8_1: %w", err)
		}
		return nil
	})
}

// encodeSwiGLUQuantQ80 fuses clamped SwiGLU activation and Q8_0 quantization.
func (r *cudaFoundation) encodeSwiGLUQuantQ80(gate, up cudamem.Slice[float32], out cudamem.Slice[byte], width int, weight float32) error {
	if width <= 0 || width%32 != 0 || gate.Len() != width || up.Len() != width || uint64(out.Len()) != uint64(width/32)*36 {
		return fmt.Errorf("cuda: SwiGLU quant Q8_0: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		blocks := uint32(width / 32)
		grid := (blocks + 7) / 8
		if err := cudalaunch.Kernel(r.swigluQuantQ80Fn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			gate, up, out, blocks, weight); err != nil {
			return fmt.Errorf("cuda: launch SwiGLU quant Q8_0: %w", err)
		}
		return nil
	})
}

// encodeSumRoutedShared sums 6 routed expert outputs and 1 shared output directly into out.
func (r *cudaFoundation) encodeSumRoutedShared(experts, shared, out cudamem.Slice[float32], width int) error {
	if width <= 0 || width%256 != 0 || experts.Len() != width*6 || shared.Len() != width || out.Len() != width {
		return fmt.Errorf("cuda: sum routed shared: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.sumRoutedSharedFn).Grid(uint32(width/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			experts, shared, out, uint32(width)); err != nil {
			return fmt.Errorf("cuda: launch sum routed shared: %w", err)
		}
		return nil
	})
}

// encodeSumRoutedSharedBatch sums 6 routed expert outputs and 1 shared output for each of n tokens.
func (r *cudaFoundation) encodeSumRoutedSharedBatch(experts, shared, out cudamem.Slice[float32], width, n int) error {
	if width <= 0 || width%256 != 0 || n <= 0 || experts.Len() != width*6*n || shared.Len() != width*n || out.Len() != width*n {
		return fmt.Errorf("cuda: sum routed shared batch: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		total := uint32(width * n)
		grid := (total + 255) / 256
		if err := cudalaunch.Kernel(r.sumRoutedSharedBatchFn).Grid(grid, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(
			experts, shared, out, uint32(width), uint32(n)); err != nil {
			return fmt.Errorf("cuda: launch sum routed shared batch: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) cudaKernels() []struct {
	name string
	dst  *driver.Function
} {
	return []struct {
		name string
		dst  *driver.Function
	}{
		{"ds4_q8_0_matvec", &r.matrixFn},
		{"ds4_quant_q8_1", &r.quantQ81Fn},
		{"ds4_quant_q8_1_batch", &r.quantQ81BatchFn},
		{"ds4_q8_0_matvec_q8_1", &r.q81MatvecFn},
		{"ds4_q8_0_matvec_q8_1_aligned", &r.q81MatvecAlignedFn},
		{"ds4_q8_0_matvec_q8_1_batch", &r.q81MatvecBatchFn},
		{"ds4_q8_0_matvec_quantized", &r.q80QuantizedFn},
		{"ds4_q8_0_matvec_quantized_aligned", &r.q80QuantizedAlignedFn},
		{"ds4_q8_0_matvec_quantized_blockdiag_aligned", &r.q80QuantizedBlockDiagAlignedFn},
		{"ds4_q8_0_matvec_quantized_batch", &r.q80QuantizedBatchFn},
		{"ds4_q8_0_matvec_quantized_small", &r.q80QuantizedSmallFn},
		{"ds4_q8_0_matvec_quantized_blockdiag", &r.q80QuantizedBlockDiagFn},
		{"ds4_q8_0_matvec_quantized_blockdiag_batch", &r.q80BlockDiagBatchFn},
		{"ds4_rms", &r.rmsFn},
		{"ds4_rms_pair", &r.rmsPairFn},
		{"ds4_f16_matvec", &r.f16Fn},
		{"ds4_f16_matvec_half_input", &r.f16DefaultFn},
		{"ds4_f16_matvec_half_input_batch", &r.f16DefaultBatchFn},
		{"ds4_f16_pair_ordered_chunks", &r.f16PairOrderedFn},
		{"ds4_f16_pair_ordered_chunks_batch", &r.f16PairOrderedBatchFn},
		{"ds4_f32_to_f16", &r.f32ToF16Fn},
		{"ds4_hc_pre", &r.hcPreFn},
		{"ds4_hc_half_input", &r.hcHalfFn},
		{"ds4_hc_cached_split", &r.hcSplitFn},
		{"ds4_hc_cached_collapse", &r.hcCollapseFn},
		{"ds4_hc_cached_expand", &r.hcExpandFn},
		{"ds4_hc_post", &r.hcPostFn},
		{"ds4_get_row_f16", &r.rowFn},
		{"ds4_get_rows_f16", &r.rowBatchFn},
		{"ds4_rope_attention_raw", &r.rawAttentionFn},
		{"ds4_raw_attention_prepare", &r.rawAttentionPrepareFn},
		{"ds4_raw_attention_eval", &r.rawAttentionEvalFn},
		{"ds4_router_hash", &r.routerHashFn},
		{"ds4_router_hash_batch", &r.routerHashBatchFn},
		{"ds4_router_biased", &r.routerBiasedFn},
		{"ds4_router_biased_batch", &r.routerBiasedBatchFn},
		{"ds4_swiglu", &r.swigluFn},
		{"ds4_swiglu_quant_q8_0", &r.swigluQuantQ80Fn},
		{"ds4_swiglu_quant_q8_1", &r.swigluQuantQ81Fn},
		{"ds4_sum_vectors", &r.sumVectorsFn},
		{"ds4_sum_routed_shared", &r.sumRoutedSharedFn},
		{"ds4_sum_routed_shared_batch", &r.sumRoutedSharedBatchFn},
		{"ds4_iq2_xxs_pair_selected_f32", &r.iq2SelectedFn},
		{"ds4_q2_k_selected_f32", &r.q2KSelectedFn},
		{"ds4_quant_q8_k", &r.quantQ8KFn},
		{"ds4_iq2_xxs_pair_selected_q8_k", &r.iq2SelectedQ8KFn},
		{"ds4_q2_k_selected_q8_k", &r.q2KSelectedQ8KFn},
		{"ds4_q4_k_selected_q8_k", &r.q4KSelectedQ8KFn},
		{"ds4_q4_k_pair_selected_q8_k_gather", &r.q4KPairGatherFn},
		{"ds4_q4_k_selected_q8_k_gather", &r.q4KDownGatherFn},
		{"ds4_iq2_xxs_pair_selected_q8_1", &r.iq2SelectedQ81Fn},
		{"ds4_iq2_xxs_pair_selected_q8_1_scalar", &r.iq2SelectedQ81ScalarFn},
		{"ds4_q2_k_selected_q8_1", &r.q2KSelectedQ81Fn},
		{"ds4_q2_k_selected_q8_1_scalar", &r.q2KSelectedQ81ScalarFn},
		{"ds4_iq2_xxs_pair_selected_q8_1_aligned", &r.iq2SelectedQ81AlignedFn},
		{"ds4_q2_k_selected_q8_1_aligned", &r.q2KSelectedQ81AlignedFn},
		{"ds4_moe_group_count", &r.moeGroupCountFn},
		{"ds4_moe_group_prefix", &r.moeGroupPrefixFn},
		{"ds4_moe_group_fill", &r.moeGroupFillFn},
		{"ds4_moe_ids_helper", &r.moeIDsHelperFn},
		{"ds4_iq2_xxs_pair_selected_q8_1_gather", &r.iq2GatherQ81Fn},
		{"ds4_q2_k_selected_q8_1_gather", &r.q2KGatherQ81Fn},
		{"ds4_quantize_mmq_q8_1", &r.quantMMQFn},
		{"ds4_quantize_mmq_q8_1_blocks", &r.quantMMQBlocksFn},
		{"ds4_d2r_make_ids_src", &r.d2rMakeIdsSrcFn},
		{"ds4_d2r_build_worklist_32", &r.d2rBuild32Fn},
		{"ds4_d2r_build_worklist_64", &r.d2rBuild64Fn},
		{"ds4_gateup_iq2_swiglu_q8_d2r", &r.gateupD2RFn},
		{"ds4_gateup_iq2_swiglu_q8_d2r_mid", &r.gateupD2RMidFn},
		{"ds4_down_q2k_d2r", &r.downD2RFn},
		{"ds4_repack_iq2_xxs_aligned", &r.repackIQ2Fn},
		{"ds4_repack_q2_k_aligned", &r.repackQ2KFn},
		{"ds4_compressor_pool", &r.compressorPoolFn},
		{"ds4_compressor_attention", &r.compressorAttentionFn},
		{"ds4_compressor_indexer", &r.compressorIndexerFn},
		{"ds4_compressor_pool_batch", &r.compressorPoolBatchFn},
		{"ds4_compressor_state_batch", &r.compressorStateBatchFn},
		{"ds4_compressor_attention_batch", &r.compressorAttentionBatchFn},
		{"ds4_compressor_indexer_batch", &r.compressorIndexerBatchFn},
		{"ds4_attention_mixed", &r.mixedAttentionFn},
		{"ds4_attention_mixed_prepare", &r.mixedAttentionPrepareFn},
		{"ds4_attention_mixed_scores", &r.mixedAttentionScoresFn},
		{"ds4_attention_mixed_normalize", &r.mixedAttentionNormalizeFn},
		{"ds4_attention_mixed_values", &r.mixedAttentionValuesFn},
		{"ds4_attention_mixed_fused", &r.mixedAttentionFusedFn},
		{"ds4_attention_mixed_scores_tile", &r.mixedAttentionTileScoresFn},
		{"ds4_attention_mixed_finalize", &r.mixedAttentionFinalizeFn},
		{"ds4_attention_mixed_prepare_batch", &r.mixedAttentionPrepareBatchFn},
		{"ds4_attention_mixed_prepare_kv_batch", &r.mixedAttentionPrepareKVBatchFn},
		{"ds4_head_rms_rope_batch", &r.headRMSRoPEBatchFn},
		{"ds4_attention_mixed_scores_tile_batch", &r.mixedAttentionTileScoresBatchFn},
		{"ds4_attention_mixed_finalize_batch", &r.mixedAttentionFinalizeBatchFn},
		{"ds4_attention_mixed_commit_raw", &r.mixedAttentionCommitRawFn},
		{"ds4_indexer_query", &r.indexerQueryFn},
		{"ds4_indexer_score", &r.indexerScoreFn},
		{"ds4_indexer_select", &r.indexerSelectFn},
		{"ds4_fill_u32", &r.fillU32Fn},
		{"ds4_attention_tokentile_dense_build", &r.tokentileDenseBuildFn},
		{"ds4_attention_tokentile_union_build", &r.tokentileUnionBuildFn},
		{"ds4_attention_tokentile_raw_mirror", &r.tokentileRawMirrorFn},
		{"ds4_attention_tokentile_comp_mirror", &r.tokentileCompMirrorFn},
		{"ds4_attention_tokentile_hmma", &r.tokentileHMMAFn},
		{"ds4_repack_q8_0_aligned", &r.repackQ80Fn},
		{"ds4_dense_q8_d2r", &r.denseQ8D2RFn},
		{"ds4_dense_q8_d2r_blockdiag", &r.denseQ8D2RBlockDiagFn},
	}
}

func (r *cudaFoundation) loadMathKernels() error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if r.matrixMod != 0 {
		return nil
	}
	compiled, err := nvrtcx.Compile(cudaMathCompilationSource(), cudaMathCompileOptions(r.major, r.minor)...)
	if err != nil {
		return fmt.Errorf("cuda: compile math kernels: %w", err)
	}
	r.mathCompiled = compiled
	mod, err := compiled.Load(r.device)
	if err != nil {
		return fmt.Errorf("cuda: load math kernels: %w", err)
	}
	for _, kernel := range r.cudaKernels() {
		fn, err := mod.Function(kernel.name)
		if err != nil {
			_ = mod.Close()
			r.clearMathKernels()
			return fmt.Errorf("cuda: find kernel %q: %w", kernel.name, err)
		}
		*kernel.dst = fn
	}
	optin, err := r.device.Raw().Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil || optin <= 0 {
		optin = 49152
	}
	if r.mixedAttentionTileScoresFn != 0 {
		if err := r.mixedAttentionTileScoresFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, cudaScoreTileShared); err != nil {
			return fmt.Errorf("cuda: set tile scores shmem: %w", err)
		}
	}
	if r.mixedAttentionFinalizeFn != 0 {
		// Finalize's dynamic shared is one float per score, not the tile
		// layout; this is only a ceiling, and it shares the tile's by
		// coincidence of size.
		_ = r.mixedAttentionFinalizeFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, cudaScoreTileShared)
	}
	if r.mixedAttentionTileScoresBatchFn != 0 {
		if err := r.mixedAttentionTileScoresBatchFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 99072); err != nil {
			return fmt.Errorf("cuda: set tile scores batch shmem: %w", err)
		}
	}
	if r.mixedAttentionFinalizeBatchFn != 0 {
		_ = r.mixedAttentionFinalizeBatchFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, cudaScoreTileShared)
	}
	if r.tokentileUnionBuildFn != 0 {
		_ = r.tokentileUnionBuildFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 65540)
	}
	if r.tokentileHMMAFn != 0 {
		if err := r.tokentileHMMAFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 88576); err != nil {
			return fmt.Errorf("cuda: set tokentile hmma shmem: %w", err)
		}
	}
	if r.denseQ8D2RFn != 0 {
		if err := r.denseQ8D2RFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 69120); err != nil {
			return fmt.Errorf("cuda: set dense q8 d2r shmem: %w", err)
		}
	}
	if r.denseQ8D2RBlockDiagFn != 0 {
		if err := r.denseQ8D2RBlockDiagFn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, 69120); err != nil {
			return fmt.Errorf("cuda: set dense q8 d2r blockdiag shmem: %w", err)
		}
	}
	r.matrixMod = mod
	return nil
}

func (r *cudaFoundation) MathCompiled() *nvrtcx.CompiledModule {
	if r == nil {
		return nil
	}
	return r.mathCompiled
}

func (r *cudaFoundation) clearMathKernels() {
	for _, kernel := range r.cudaKernels() {
		*kernel.dst = 0
	}
}

// q80Row computes the first registered Q8_0 tensor row. The caller owns x and
// out; this foundation owns only the bounded, registered model row.
func (r *cudaFoundation) q80Row(x, out cudamem.Slice[float32]) error {
	if err := r.checkOpen(); err != nil {
		return err
	}
	if r.probePtr == 0 || r.probeCols == 0 {
		return fmt.Errorf("cuda: Q8_0 row is not bound")
	}
	if x.Len() != int(r.probeCols) || out.Len() != 1 {
		return fmt.Errorf("cuda: Q8_0 row input/output lengths %d/%d, want %d/1", x.Len(), out.Len(), r.probeCols)
	}
	compiled, err := compileCUDAProbe(r.device, "ds4_q8_0_row.cu", cudaProbeCompiler())
	if err != nil {
		return fmt.Errorf("cuda: compile Q8_0 row: %w", err)
	}
	return r.withContext(func() (err error) {
		cleanup := r.probeResources()
		defer func() { err = finishBackendCleanup(err, cleanup) }()
		mod, err := loadCUDAProbe(compiled, r.device, cudaProbeCompiler())
		cleanup.module = mod
		if err != nil {
			return fmt.Errorf("cuda: load Q8_0 row: %w", err)
		}
		fn, err := mod.Function("ds4_q8_0_matvec")
		if err != nil {
			return fmt.Errorf("cuda: find Q8_0 row: %w", err)
		}
		if err := cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(256, 1, 1).Launch(r.probePtr, x, out, r.probeCols, uint32(1)); err != nil {
			return fmt.Errorf("cuda: launch Q8_0 row: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("cuda: synchronize Q8_0 row: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) Close() error {
	if r == nil {
		return nil
	}
	r.closing = true
	if r.streamingExperts != nil {
		if err := r.streamingExperts.Close(); err != nil {
			return fmt.Errorf("cuda: close streaming experts: %w", err)
		}
		r.streamingExperts = nil
	}
	ops := r.closeOperations()
	if !r.closeDrained {
		if r.workStream != nil {
			if err := ops.synchronize(r.workStream); err != nil {
				return fmt.Errorf("cuda: drain foundation: %w", err)
			}
		}
		if r.device != nil {
			if err := ops.run(r, ops.synchronizeContext); err != nil {
				return fmt.Errorf("cuda: drain foundation context: %w", err)
			}
		}
		r.closeDrained = true
	}
	if r.quantQ80Mod != 0 && r.device != nil {
		// A foundation may also be closed directly by a qualification harness.
		// Establish completion before unloading the separately owned quantizer.
		err := ops.run(r, func() error {
			if err := ops.unload(r.quantQ80Mod); err != nil {
				return err
			}
			r.quantQ80Mod, r.quantQ80Fn, r.quantQ80StridedFn = 0, 0, 0
			return nil
		})
		if err != nil {
			return fmt.Errorf("cuda: close Q8_0 quantizer: %w", err)
		}
	}
	if r.matrixMod != 0 && r.device != nil {
		err := ops.run(r, func() error {
			if err := ops.unload(r.matrixMod); err != nil {
				return err
			}
			r.matrixMod = 0
			r.clearMathKernels()
			return nil
		})
		if err != nil {
			return fmt.Errorf("cuda: close math module: %w", err)
		}
	}
	if r.residentStageClose != nil {
		if err := r.residentStageClose(); err != nil {
			return fmt.Errorf("cuda: free resident staging: %w", err)
		}
		r.residentStageClose = nil
	}
	if !r.residentReleased {
		free := func() error { return ops.freeResident(r) }
		if r.residentClose != nil {
			free = r.residentClose
		}
		if err := free(); err != nil {
			return fmt.Errorf("cuda: free resident weights: %w", err)
		}
		r.resident = cudamem.Slice[byte]{}
		r.residentClose = nil
		r.residentReleased = true
	}
	if r.mapped.host != 0 && r.device != nil {
		err := ops.run(r, func() error {
			if err := ops.unregister(r.mapped.host); err != nil {
				return err
			}
			r.mapped.host = 0
			return nil
		})
		if err != nil {
			return fmt.Errorf("cuda: unregister mapped weights: %w", err)
		}
	}
	if r.mapped.view != nil {
		if err := ops.unmap(r.mapped.view); err != nil {
			return fmt.Errorf("cuda: unmap registered weights: %w", err)
		}
		r.mapped = cudaModelRange{}
	}
	if r.workStream != nil {
		if err := ops.destroy(r.workStream); err != nil {
			return fmt.Errorf("cuda: destroy work stream: %w", err)
		}
		r.workStream = nil
	}
	if r.device != nil {
		if err := ops.closeDevice(r.device); err != nil {
			return fmt.Errorf("cuda: close device: %w", err)
		}
		r.device = nil
	}
	if r.model != nil {
		if err := ops.unmap(r.model); err != nil {
			return fmt.Errorf("cuda: unmap model: %w", err)
		}
		r.model = nil
	}
	r.gguf = nil
	if r.file != nil {
		f := r.file
		r.file = nil
		if err := ops.closeFile(f); err != nil {
			return fmt.Errorf("cuda: close model file: %w", err)
		}
	}
	if r.direct != nil {
		f := r.direct
		r.direct = nil
		if err := ops.closeFile(f); err != nil {
			return fmt.Errorf("cuda: close direct file: %w", err)
		}
	}
	return nil
}

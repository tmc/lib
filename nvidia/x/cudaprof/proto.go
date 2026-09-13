package cudaprof

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
)

type stringTable struct {
	strings []string
	lookup  map[string]int64
}

func newStringTable() *stringTable {
	st := &stringTable{
		strings: []string{""},
		lookup:  map[string]int64{"": 0},
	}
	return st
}

func (st *stringTable) Index(s string) int64 {
	if idx, ok := st.lookup[s]; ok {
		return idx
	}
	idx := int64(len(st.strings))
	st.strings = append(st.strings, s)
	st.lookup[s] = idx
	return idx
}

func encodeVarint(fieldNum int, val uint64) []byte {
	tag := uint64((fieldNum << 3) | 0)
	var buf [20]byte
	n1 := binary.PutUvarint(buf[:], tag)
	n2 := binary.PutUvarint(buf[n1:], val)
	return buf[:n1+n2]
}

func encodeLengthDelimited(fieldNum int, data []byte) []byte {
	tag := uint64((fieldNum << 3) | 2)
	var buf [10]byte
	n1 := binary.PutUvarint(buf[:], tag)
	var lenBuf [10]byte
	n2 := binary.PutUvarint(lenBuf[:], uint64(len(data)))

	out := make([]byte, 0, n1+n2+len(data))
	out = append(out, buf[:n1]...)
	out = append(out, lenBuf[:n2]...)
	out = append(out, data...)
	return out
}

func encodeString(fieldNum int, s string) []byte {
	return encodeLengthDelimited(fieldNum, []byte(s))
}

// SampleValues contains caller-supplied metrics for an instruction or source
// line. The caller is responsible for measurement and attribution; the builder
// does not collect or verify hardware counters. Leave unmeasured fields zero.
// All metrics are strictly additive to support valid pprof callstack aggregation;
// ratios (such as hit rates or occupancies) must be computed post-aggregation.
type SampleValues struct {
	Samples             int64 // Execution clock cycles / sample hits
	PCSamples           int64 // Hardware PC samples collected at an address
	VRAMAllocBytes      int64 // VRAM allocated / staged (bytes)
	DRAMReadBytes       int64 // Global memory / DRAM read traffic (bytes)
	DRAMWriteBytes      int64 // Global memory / DRAM write traffic (bytes)
	L2SectorsRead       int64 // L2 cache read sectors (32 bytes per sector)
	L2SectorsMiss       int64 // L2 cache read miss sectors (32 bytes per sector)
	SharedMemBytes      int64 // Shared memory read/write traffic (bytes)
	BankConflicts       int64 // Shared memory bank conflict replay cycles
	TensorWarps         int64 // Tensor Core (HMMA/IMMA/WGMMA) warp executions
	ActiveWarps         int64 // Active warp cycles across SMs (for occupancy derivation)
	EligibleWarps       int64 // Eligible warp cycles ready to issue
	IssuedWarps         int64 // Warp instructions issued
	InstExecuted        int64 // Instructions executed
	RegSpillBytes       int64 // Register spills to local memory (bytes)
	InterconnectTxBytes int64 // Multi-GPU interconnect / NVLink / PCIe transmit traffic (bytes)
	InterconnectRxBytes int64 // Multi-GPU interconnect / NVLink / PCIe receive traffic (bytes)
	StallScoreboard     int64 // Warps stalled on register/memory scoreboard
	StallWait           int64 // Warps stalled on synchronization barrier (__syncthreads)
	StallMath           int64 // Warps stalled on arithmetic pipe throttle
	StallMem            int64 // Warps stalled on memory throttle or load latency
	LaunchDelayNanos    int64 // Host enqueue to device launch delay (nanoseconds)
	QueueDelayNanos     int64 // Stream queue latency (nanoseconds)
	GPUNanos            int64 // Observed device timeline interval (nanoseconds)
}

// Builder constructs a pprof profile containing GPU kernel, PTX, and SASS samples.
type Builder struct {
	st          *stringTable
	mappings    []mapping
	locations   []location
	functions   []function
	funcMap     map[string]uint64
	binaryPath  string
	samples     []sample
	sampleTypes []int
	locMap      map[locationKey]uint64

	timeNanos     int64
	durationNanos int64
	periodType    struct{ name, unit string }
	period        int64
	defaultType   string
	comments      []string
	docURL        string
	numericLabels map[string]string
}

// locationKey identifies a location by everything the proto records about it,
// so that repeated samples at one address reuse a single location entry.
type locationKey struct {
	mappingID  uint64
	address    uint64
	functionID uint64
	line       int64
}

var profileSampleTypes = []struct{ name, unit string }{
	{"samples", "count"},
	{"pc_samples", "count"},
	{"vram_alloc", "bytes"},
	{"dram_read", "bytes"},
	{"dram_write", "bytes"},
	{"l2_sectors_read", "count"},
	{"l2_sectors_miss", "count"},
	{"shared_mem", "bytes"},
	{"bank_conflicts", "count"},
	{"tensor_warps", "count"},
	{"active_warps", "count"},
	{"eligible_warps", "count"},
	{"issued_warps", "count"},
	{"inst_executed", "count"},
	{"reg_spill", "bytes"},
	{"interconnect_tx", "bytes"},
	{"interconnect_rx", "bytes"},
	{"stall_scoreboard", "count"},
	{"stall_wait", "count"},
	{"stall_math", "count"},
	{"stall_mem", "count"},
	{"launch_delay", "nanoseconds"},
	{"queue_delay", "nanoseconds"},
	{"gpu_time", "nanoseconds"},
}

// SetSampleTypes selects the metrics to write, in order. The last metric is
// pprof's default view. With no names, it restores all metrics. Invalid or
// duplicate names leave the selection unchanged. Selection applies to all
// recorded samples.
func (b *Builder) SetSampleTypes(names ...string) error {
	var selected []int
	for _, name := range names {
		i := slices.IndexFunc(profileSampleTypes, func(t struct{ name, unit string }) bool { return t.name == name })
		if i < 0 {
			return fmt.Errorf("cudaprof: unknown sample type %q", name)
		}
		if slices.Contains(selected, i) {
			return fmt.Errorf("cudaprof: duplicate sample type %q", name)
		}
		selected = append(selected, i)
	}
	b.sampleTypes = selected
	return nil
}

type mapping struct {
	id              uint64
	memoryStart     uint64
	memoryLimit     uint64
	fileOffset      uint64
	filename        int64
	buildID         int64
	hasFunctions    bool
	hasFilenames    bool
	hasLineNumbers  bool
	hasInlineFrames bool
}

type function struct {
	id         uint64
	name       int64
	systemName int64
	file       int64
	startLine  int64
}

type location struct {
	id         uint64
	mappingID  uint64
	address    uint64
	functionID uint64
	line       int64
}

// Frame represents a single callstack level (in Go host code, CUDA kernel, PTX virtual ISA, or SASS disassembly).
type Frame struct {
	FunctionName string
	FileName     string
	Line         int64
	Address      uint64 // Zero denotes a source-only frame, without a binary mapping.
}

type sample struct {
	locationIDs []uint64
	values      []int64
	labels      map[string]string
}

// DefaultMappingStart is the profile's logical image base. Addresses are
// DefaultMappingStart plus CUBIN file offsets; they are not sampled device
// program counters.
const DefaultMappingStart = uint64(0x7f000000)

// PTXMappingStart is the logical image base for recorded PTX text. It sits
// below DefaultMappingStart so the PTX and CUBIN images never overlap.
const PTXMappingStart = uint64(0x7e000000)

// PTXLineOffsets returns the byte offset at which each one-based line of text
// begins. Element 0 is unused so that a line number indexes the slice directly,
// and the last element is the length of text, so the line past the final one
// has an offset too.
//
// A PTX address is PTXMappingStart plus one of these offsets. The capture that
// writes the profile and the viewer that annotates it must derive addresses the
// same way, or one statement acquires two addresses; sharing this function is
// what keeps them from drifting.
func PTXLineOffsets(text string) []uint64 {
	offsets := []uint64{0}
	var offset uint64
	for _, line := range strings.SplitAfter(text, "\n") {
		offsets = append(offsets, offset)
		offset += uint64(len(line))
	}
	return append(offsets, offset)
}

// NewBuilder initializes a profile builder.
func NewBuilder() *Builder {
	b := &Builder{
		st:      newStringTable(),
		funcMap: make(map[string]uint64),
		locMap:  make(map[locationKey]uint64),
	}
	b.mappings = append(b.mappings, mapping{
		id:             1,
		memoryStart:    DefaultMappingStart,
		memoryLimit:    0x7fffffff,
		fileOffset:     0,
		filename:       b.st.Index("cuda_kernels.so"),
		buildID:        0,
		hasFunctions:   true,
		hasFilenames:   true,
		hasLineNumbers: true,
		// hasInlineFrames stays false: every location this builder emits carries
		// exactly one Line. SASS, PTX and source are modeled as separate frames
		// in the sample's stack, not as inline frames on one location.
	})
	return b
}

// SetBinaryPath requests the legacy generated ELF sidecar. Nonempty paths now
// cause WriteGzip to fail: instruction bytes cannot be inferred from samples.
// Use SetMappingFile with a retained native code artifact instead.
//
// Deprecated: generated sidecars do not represent captured GPU instructions.
func (b *Builder) SetBinaryPath(path string) {
	b.binaryPath = path
}

// SetMappingFile configures the binary/PTX file path for instruction disassembly resolution.
func (b *Builder) SetMappingFile(filename, buildID string) {
	if len(b.mappings) > 0 {
		b.mappings[0].filename = b.st.Index(filename)
		if buildID != "" {
			b.mappings[0].buildID = b.st.Index(buildID)
		}
	}
}

// SetTimespan records when the profile was collected and how long it covers.
// pprof reports the sampled total as a fraction of the duration, so a profile
// without these prints no wall-clock context at all.
func (b *Builder) SetTimespan(start time.Time, d time.Duration) {
	if !start.IsZero() {
		b.timeNanos = start.UnixNano()
	}
	if d > 0 {
		b.durationNanos = int64(d)
	}
}

// SetPeriod records the sampling period and what it measures. For an event
// interval profile the period is the measurement granularity, not a rate.
func (b *Builder) SetPeriod(name, unit string, period int64) {
	b.periodType.name = name
	b.periodType.unit = unit
	b.period = period
}

// SetDefaultSampleType names the sample type a viewer should open. Without it
// a consumer that defaults to index 0 may render an empty profile, because the
// first type need not be the one carrying data. The name must be one of the
// configured sample types.
func (b *Builder) SetDefaultSampleType(name string) error {
	for _, i := range b.selectedTypes() {
		if profileSampleTypes[i].name == name {
			b.defaultType = name
			return nil
		}
	}
	return fmt.Errorf("cudaprof: default sample type %q is not among the configured types", name)
}

// AddComment records a line of free text on the profile. It is the home for a
// fact that is true of the whole run -- the device, what was measured, how the
// samples were attributed -- because a label carrying the same value on every
// sample can neither filter nor split a profile, and pprof shows it on every
// sample's tooltip regardless.
func (b *Builder) AddComment(text string) {
	if text != "" {
		b.comments = append(b.comments, text)
	}
}

// SetDocURL records a documentation link for the profile. pprof's web UI
// renders it, which makes it the place to send a reader who needs to know how
// these samples were produced and what tool reads them best.
func (b *Builder) SetDocURL(url string) {
	b.docURL = url
}

// SetNumericLabels declares label keys whose values are numbers, mapping each
// to its unit. pprof filters a bare number or "a:b" in -tagfocus as a numeric
// range against these, so a count left as a string label cannot be
// range-filtered. A declared value that does not parse as an integer is still
// written as a string label rather than dropped.
//
// A key mapped to an empty unit is recorded with the unit "index", because
// pprof discards a numeric label whose value and unit are both zero, and zero
// is a valid index for a pass or a layer.
func (b *Builder) SetNumericLabels(unitByKey map[string]string) {
	b.numericLabels = maps.Clone(unitByKey)
}

// selectedTypes returns the configured sample type indexes, defaulting to all.
func (b *Builder) selectedTypes() []int {
	if len(b.sampleTypes) > 0 {
		return b.sampleTypes
	}
	all := make([]int, len(profileSampleTypes))
	for i := range profileSampleTypes {
		all[i] = i
	}
	return all
}

// AddMapping registers an additional logical image and returns its mapping ID.
// A frame whose address falls inside [start, limit) is attributed to this
// mapping rather than the default one, which is how a profile can carry both
// the CUBIN and the PTX text as separately addressed images. Ranges must not
// overlap; the first containing range wins.
func (b *Builder) AddMapping(filename, buildID string, start, limit uint64) uint64 {
	id := uint64(len(b.mappings) + 1)
	m := mapping{
		id:             id,
		memoryStart:    start,
		memoryLimit:    limit,
		filename:       b.st.Index(filename),
		hasFunctions:   true,
		hasFilenames:   true,
		hasLineNumbers: true,
	}
	if buildID != "" {
		m.buildID = b.st.Index(buildID)
	}
	b.mappings = append(b.mappings, m)
	return id
}

// mappingFor returns the ID of the mapping containing addr, or zero when addr
// is zero or falls outside every registered range.
func (b *Builder) mappingFor(addr uint64) uint64 {
	if addr == 0 {
		return 0
	}
	for _, m := range b.mappings {
		if addr >= m.memoryStart && addr < m.memoryLimit {
			return m.id
		}
	}
	// The default mapping keeps addresses it does not bound, so that an
	// unregistered image still disassembles against the CUBIN as before.
	return 1
}

func (b *Builder) getOrCreateFunction(kernelName, filename string, startLine int64) uint64 {
	key := fmt.Sprintf("%s:%s", kernelName, filename)
	if id, ok := b.funcMap[key]; ok {
		return id
	}

	fnID := uint64(len(b.functions) + 1)
	b.functions = append(b.functions, function{
		id:         fnID,
		name:       b.st.Index(kernelName),
		systemName: b.st.Index(kernelName),
		file:       b.st.Index(filename),
		startLine:  startLine,
	})
	b.funcMap[key] = fnID
	return fnID
}

// AddSample records a GPU execution sample with source attribution and optional pprof labels.
func (b *Builder) AddSample(kernelName, filename string, line int64, sassPC uint64, count int64, labels map[string]string) {
	b.AddDetailedSample(kernelName, filename, line, sassPC, SampleValues{
		Samples: count,
	}, labels)
}

// AddDetailedSample records comprehensive multi-dimensional metrics across execution, VRAM, DRAM, and stalls.
func (b *Builder) AddDetailedSample(kernelName, filename string, line int64, sassPC uint64, vals SampleValues, labels map[string]string) {
	b.AddStackSample(vals, []Frame{{
		FunctionName: kernelName,
		FileName:     filename,
		Line:         line,
		Address:      sassPC,
	}}, labels)
}

// AddStackSample records a multi-frame callstack sample (ordered from leaf callee to root caller).
func (b *Builder) AddStackSample(vals SampleValues, frames []Frame, labels map[string]string) {
	if len(frames) == 0 {
		return
	}

	locIDs := make([]uint64, 0, len(frames))
	for _, frame := range frames {
		fnID := b.getOrCreateFunction(frame.FunctionName, frame.FileName, frame.Line)
		mappingID := b.mappingFor(frame.Address)
		key := locationKey{mappingID: mappingID, address: frame.Address, functionID: fnID, line: frame.Line}
		locID, ok := b.locMap[key]
		if !ok {
			locID = uint64(len(b.locations) + 1)
			b.locations = append(b.locations, location{
				id:         locID,
				mappingID:  mappingID,
				address:    frame.Address,
				functionID: fnID,
				line:       frame.Line,
			})
			if b.locMap == nil {
				b.locMap = make(map[locationKey]uint64)
			}
			b.locMap[key] = locID
		}
		locIDs = append(locIDs, locID)
	}

	metricVals := []int64{
		vals.Samples,
		vals.PCSamples,
		vals.VRAMAllocBytes,
		vals.DRAMReadBytes,
		vals.DRAMWriteBytes,
		vals.L2SectorsRead,
		vals.L2SectorsMiss,
		vals.SharedMemBytes,
		vals.BankConflicts,
		vals.TensorWarps,
		vals.ActiveWarps,
		vals.EligibleWarps,
		vals.IssuedWarps,
		vals.InstExecuted,
		vals.RegSpillBytes,
		vals.InterconnectTxBytes,
		vals.InterconnectRxBytes,
		vals.StallScoreboard,
		vals.StallWait,
		vals.StallMath,
		vals.StallMem,
		vals.LaunchDelayNanos,
		vals.QueueDelayNanos,
		vals.GPUNanos,
	}

	b.samples = append(b.samples, sample{
		locationIDs: locIDs,
		values:      metricVals,
		labels:      labels,
	})
}

// AddTracebackSample records a unified multi-frame sample connecting SASS/PTX execution -> CUDA C++ kernel -> Go host callers.
func (b *Builder) AddTracebackSample(
	vals SampleValues,
	kernelName string,
	cuFile string,
	cuLine int64,
	ptxFile string,
	ptxLine int64,
	sassPC uint64,
	labels map[string]string,
) {
	var frames []Frame
	if ptxFile != "" && ptxLine > 0 {
		frames = append(frames, Frame{
			FunctionName: kernelName + "_PTX",
			FileName:     ptxFile,
			Line:         ptxLine,
			Address:      sassPC,
		})
	}
	if cuFile != "" && cuLine > 0 {
		frames = append(frames, Frame{
			FunctionName: kernelName,
			FileName:     cuFile,
			Line:         cuLine,
			Address:      sassPC,
		})
	}

	// Capture Go host frames
	var pcs [16]uintptr
	n := runtime.Callers(2, pcs[:])
	goFrames := runtime.CallersFrames(pcs[:n])
	for {
		frame, more := goFrames.Next()
		if frame.Function != "" && !strings.Contains(frame.Function, "runtime.") && !strings.Contains(frame.Function, "cudaprof.") {
			frames = append(frames, Frame{
				FunctionName: frame.Function,
				FileName:     frame.File,
				Line:         int64(frame.Line),
			})
		}
		if !more {
			break
		}
	}

	b.AddStackSample(vals, frames, labels)
}

// WriteGzip encodes the profile as a standard gzipped pprof profile with multiple metric dimensions.
func (b *Builder) WriteGzip(w io.Writer) error {
	if b.binaryPath != "" {
		return fmt.Errorf("cudaprof: generated ELF sidecars are not captured code; use SetMappingFile with a retained native artifact")
	}

	var body []byte

	// 1. Sample types. Keep the same selection for each sample's values.
	selected := b.selectedTypes()
	for _, i := range selected {
		st := profileSampleTypes[i]
		var stBody []byte
		stBody = append(stBody, encodeVarint(1, uint64(b.st.Index(st.name)))...)
		stBody = append(stBody, encodeVarint(2, uint64(b.st.Index(st.unit)))...)
		body = append(body, encodeLengthDelimited(1, stBody)...)
	}

	// 2. Samples
	for _, s := range b.samples {
		var sBody []byte
		for _, locID := range s.locationIDs {
			sBody = append(sBody, encodeVarint(1, locID)...) // location_id (repeated uint64)
		}
		for _, i := range selected {
			sBody = append(sBody, encodeVarint(2, uint64(s.values[i]))...) // values (repeated int64)
		}

		for _, k := range slices.Sorted(maps.Keys(s.labels)) {
			v := s.labels[k]
			var lBody []byte
			lBody = append(lBody, encodeVarint(1, uint64(b.st.Index(k)))...)
			unit, numeric := b.numericLabels[k]
			n, convErr := strconv.ParseInt(v, 10, 64)
			if numeric && convErr == nil {
				if unit == "" {
					unit = "index"
				}
				lBody = append(lBody, encodeVarint(3, uint64(n))...)
				lBody = append(lBody, encodeVarint(4, uint64(b.st.Index(unit)))...)
			} else {
				lBody = append(lBody, encodeVarint(2, uint64(b.st.Index(v)))...)
			}
			sBody = append(sBody, encodeLengthDelimited(3, lBody)...)
		}
		body = append(body, encodeLengthDelimited(2, sBody)...)
	}

	// 3. Mappings
	for _, m := range b.mappings {
		var mBody []byte
		mBody = append(mBody, encodeVarint(1, m.id)...)
		mBody = append(mBody, encodeVarint(2, m.memoryStart)...)
		mBody = append(mBody, encodeVarint(3, m.memoryLimit)...)
		mBody = append(mBody, encodeVarint(4, m.fileOffset)...)
		mBody = append(mBody, encodeVarint(5, uint64(m.filename))...)
		mBody = append(mBody, encodeVarint(6, uint64(m.buildID))...)
		if m.hasFunctions {
			mBody = append(mBody, encodeVarint(7, 1)...)
		}
		if m.hasFilenames {
			mBody = append(mBody, encodeVarint(8, 1)...)
		}
		if m.hasLineNumbers {
			mBody = append(mBody, encodeVarint(9, 1)...)
		}
		if m.hasInlineFrames {
			mBody = append(mBody, encodeVarint(10, 1)...)
		}
		body = append(body, encodeLengthDelimited(3, mBody)...)
	}

	// 4. Locations
	for _, loc := range b.locations {
		var lBody []byte
		lBody = append(lBody, encodeVarint(1, loc.id)...)
		if loc.mappingID > 0 {
			lBody = append(lBody, encodeVarint(2, loc.mappingID)...)
		}
		lBody = append(lBody, encodeVarint(3, loc.address)...) // SASS PC address

		// Line
		var lineBody []byte
		lineBody = append(lineBody, encodeVarint(1, loc.functionID)...)
		lineBody = append(lineBody, encodeVarint(2, uint64(loc.line))...)
		lBody = append(lBody, encodeLengthDelimited(4, lineBody)...)

		body = append(body, encodeLengthDelimited(4, lBody)...)
	}

	// 5. Functions
	for _, fn := range b.functions {
		var fBody []byte
		fBody = append(fBody, encodeVarint(1, fn.id)...)
		fBody = append(fBody, encodeVarint(2, uint64(fn.name))...)
		fBody = append(fBody, encodeVarint(3, uint64(fn.systemName))...)
		fBody = append(fBody, encodeVarint(4, uint64(fn.file))...)
		fBody = append(fBody, encodeVarint(5, uint64(fn.startLine))...)
		body = append(body, encodeLengthDelimited(5, fBody)...)
	}

	// 6. Profile-level fields. period_type and default_sample_type intern
	// strings, so they are resolved before the table is serialized.
	var header []byte
	if b.timeNanos != 0 {
		header = append(header, encodeVarint(9, uint64(b.timeNanos))...)
	}
	if b.durationNanos != 0 {
		header = append(header, encodeVarint(10, uint64(b.durationNanos))...)
	}
	if b.periodType.name != "" || b.periodType.unit != "" {
		var ptBody []byte
		ptBody = append(ptBody, encodeVarint(1, uint64(b.st.Index(b.periodType.name)))...)
		ptBody = append(ptBody, encodeVarint(2, uint64(b.st.Index(b.periodType.unit)))...)
		header = append(header, encodeLengthDelimited(11, ptBody)...)
	}
	if b.period != 0 {
		header = append(header, encodeVarint(12, uint64(b.period))...)
	}
	for _, c := range b.comments {
		header = append(header, encodeVarint(13, uint64(b.st.Index(c)))...)
	}
	if b.defaultType != "" {
		header = append(header, encodeVarint(14, uint64(b.st.Index(b.defaultType)))...)
	}
	if b.docURL != "" {
		header = append(header, encodeVarint(15, uint64(b.st.Index(b.docURL)))...)
	}
	body = append(body, header...)

	// 7. String table
	for _, str := range b.st.strings {
		body = append(body, encodeString(6, str)...)
	}

	// Gzip compress
	gw := gzip.NewWriter(w)
	if _, err := gw.Write(body); err != nil {
		return err
	}
	return gw.Close()
}

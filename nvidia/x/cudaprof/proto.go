package cudaprof

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
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

// SampleValues contains multi-dimensional hardware metrics for an instruction or source line.
type SampleValues struct {
	Samples         int64 // Execution clock cycles / sample hits
	VRAMAllocBytes  int64 // VRAM allocated / staged (bytes)
	DRAMReadBytes   int64 // Global memory / DRAM read traffic (bytes)
	DRAMWriteBytes  int64 // Global memory / DRAM write traffic (bytes)
	SharedMemBytes  int64 // Shared memory read/write traffic (bytes)
	TensorWarps     int64 // Tensor Core (HMMA/IMMA) warp executions
	StallScoreboard int64 // Warps stalled on register/memory scoreboard
	StallWait       int64 // Warps stalled on synchronization barrier (__syncthreads)
}

// Builder constructs a pprof profile containing GPU kernel, PTX, and SASS samples.
type Builder struct {
	st        *stringTable
	mappings  []mapping
	locations []location
	functions []function
	funcMap   map[string]uint64
	samples   []sample
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

type sample struct {
	locationID uint64
	values     []int64
	labels     map[string]string
}

// NewBuilder initializes a profile builder.
func NewBuilder() *Builder {
	b := &Builder{
		st:      newStringTable(),
		funcMap: make(map[string]uint64),
	}
	b.mappings = append(b.mappings, mapping{
		id:              1,
		memoryStart:     0x7f000000,
		memoryLimit:     0x7fffffff,
		fileOffset:      0,
		filename:        b.st.Index("cuda_kernels"),
		buildID:         b.st.Index("nvrtc_jit"),
		hasFunctions:    true,
		hasFilenames:    true,
		hasLineNumbers:  true,
		hasInlineFrames: true,
	})
	return b
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
	fnID := b.getOrCreateFunction(kernelName, filename, line)

	locID := uint64(len(b.locations) + 1)
	b.locations = append(b.locations, location{
		id:         locID,
		mappingID:  1,
		address:    sassPC,
		functionID: fnID,
		line:       line,
	})

	metricVals := []int64{
		vals.Samples,
		vals.VRAMAllocBytes,
		vals.DRAMReadBytes,
		vals.DRAMWriteBytes,
		vals.SharedMemBytes,
		vals.TensorWarps,
		vals.StallScoreboard,
		vals.StallWait,
	}

	b.samples = append(b.samples, sample{
		locationID: locID,
		values:     metricVals,
		labels:     labels,
	})
}

// WriteGzip encodes the profile as a standard gzipped pprof profile with multiple metric dimensions.
func (b *Builder) WriteGzip(w io.Writer) error {
	var body []byte

	// 1. SampleTypes: [samples/count, vram_alloc/bytes, dram_read/bytes, dram_write/bytes, shared_mem/bytes, tensor_warps/count, stall_scoreboard/count, stall_wait/count]
	sampleTypes := []struct {
		typ  string
		unit string
	}{
		{"samples", "count"},
		{"vram_alloc", "bytes"},
		{"dram_read", "bytes"},
		{"dram_write", "bytes"},
		{"shared_mem", "bytes"},
		{"tensor_warps", "count"},
		{"stall_scoreboard", "count"},
		{"stall_wait", "count"},
	}

	for _, st := range sampleTypes {
		var stBody []byte
		stBody = append(stBody, encodeVarint(1, uint64(b.st.Index(st.typ)))...)
		stBody = append(stBody, encodeVarint(2, uint64(b.st.Index(st.unit)))...)
		body = append(body, encodeLengthDelimited(1, stBody)...)
	}

	// 2. Samples
	for _, s := range b.samples {
		var sBody []byte
		sBody = append(sBody, encodeVarint(1, s.locationID)...) // location_id
		for _, val := range s.values {
			sBody = append(sBody, encodeVarint(2, uint64(val))...) // values (repeated int64)
		}

		for k, v := range s.labels {
			var lBody []byte
			lBody = append(lBody, encodeVarint(1, uint64(b.st.Index(k)))...)
			lBody = append(lBody, encodeVarint(2, uint64(b.st.Index(v)))...)
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

	// 6. String table
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

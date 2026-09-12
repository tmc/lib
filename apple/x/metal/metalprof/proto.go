package metalprof

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
	return &stringTable{
		strings: []string{""},
		lookup:  map[string]int64{"": 0},
	}
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

// SampleValues represents metric counts for a profiling sample.
type SampleValues struct {
	Samples                   int64 // Number of occurrences / dispatches
	GPUTimeNs                 int64 // Cumulative GPU execution duration in nanoseconds
	CommandBufferCount        int64 // Successful command-buffer creations
	CommandBufferCommitCount  int64 // Command-buffer submissions
	CommandBufferCreateTimeNs int64 // Host time spent creating command buffers
}

// Builder constructs a standard pprof profile from recorded Metal samples.
type Builder struct {
	st             *stringTable
	mappings       []mapping
	locations      []location
	functions      []function
	funcMap        map[string]uint64
	commandBuffers bool
	samples        []sample
}

type mapping struct {
	id          uint64
	memoryStart uint64
	memoryLimit uint64
	fileOffset  uint64
	filename    int64
	buildID     int64
}

type function struct {
	id         uint64
	name       int64
	systemName int64
	filename   int64
	startLine  int64
}

type line struct {
	functionID uint64
	line       int64
}

type location struct {
	id        uint64
	mappingID uint64
	address   uint64
	lines     []line
}

type label struct {
	key int64
	str int64
	num int64
}

type sample struct {
	locationIDs []uint64
	values      []int64
	labels      []label
}

// NewBuilder creates a new profile.proto builder.
func NewBuilder() *Builder {
	b := &Builder{
		st:      newStringTable(),
		funcMap: make(map[string]uint64),
	}
	b.mappings = append(b.mappings, mapping{
		id:          1,
		memoryStart: 0x1000,
		memoryLimit: 0x1000000,
		filename:    b.st.Index("MetalPipeline"),
	})
	return b
}

// AddFunction registers a named function in the profile.
func (b *Builder) AddFunction(name, filename string, startLine int64) uint64 {
	if id, ok := b.funcMap[name]; ok {
		return id
	}
	id := uint64(len(b.functions) + 1)
	b.functions = append(b.functions, function{
		id:         id,
		name:       b.st.Index(name),
		systemName: b.st.Index(name),
		filename:   b.st.Index(filename),
		startLine:  startLine,
	})
	b.funcMap[name] = id
	return id
}

// AddLocation registers a source location referencing a function.
// For source-only profiles, address is 0 rather than a fabricated memory offset.
func (b *Builder) AddLocation(funcID uint64, lineNum int64) uint64 {
	id := uint64(len(b.locations) + 1)
	b.locations = append(b.locations, location{
		id:        id,
		mappingID: 1,
		address:   0,
		lines: []line{
			{functionID: funcID, line: lineNum},
		},
	})
	return id
}

// AddSample records a sample with the given callstack location IDs, values, and string labels.
func (b *Builder) AddSample(locIDs []uint64, vals SampleValues, strLabels map[string]string) {
	if vals.CommandBufferCount != 0 || vals.CommandBufferCommitCount != 0 || vals.CommandBufferCreateTimeNs != 0 {
		b.commandBuffers = true
	}
	var pbLabels []label
	for k, v := range strLabels {
		pbLabels = append(pbLabels, label{
			key: b.st.Index(k),
			str: b.st.Index(v),
		})
	}

	b.samples = append(b.samples, sample{
		locationIDs: locIDs,
		values:      []int64{vals.Samples, vals.GPUTimeNs, vals.CommandBufferCount, vals.CommandBufferCommitCount, vals.CommandBufferCreateTimeNs},
		labels:      pbLabels,
	})
}

// Encode serializes the profile into gzip-compressed protobuf bytes.
func (b *Builder) Encode(w io.Writer) error {
	gw := gzip.NewWriter(w)
	defer gw.Close()

	var msg []byte

	// 1. SampleType
	// [0] samples / count
	var st1 []byte
	st1 = append(st1, encodeVarint(1, uint64(b.st.Index("samples")))...)
	st1 = append(st1, encodeVarint(2, uint64(b.st.Index("count")))...)
	msg = append(msg, encodeLengthDelimited(1, st1)...)

	// [1] gpu_time / nanoseconds
	var st2 []byte
	st2 = append(st2, encodeVarint(1, uint64(b.st.Index("gpu_time")))...)
	st2 = append(st2, encodeVarint(2, uint64(b.st.Index("nanoseconds")))...)
	msg = append(msg, encodeLengthDelimited(1, st2)...)

	if b.commandBuffers {
		for _, typ := range []struct{ name, unit string }{
			{"command_buffer_count", "count"},
			{"command_buffer_commit_count", "count"},
			{"command_buffer_create_time", "nanoseconds"},
		} {
			var st []byte
			st = append(st, encodeVarint(1, uint64(b.st.Index(typ.name)))...)
			st = append(st, encodeVarint(2, uint64(b.st.Index(typ.unit)))...)
			msg = append(msg, encodeLengthDelimited(1, st)...)
		}
	}
	msg = append(msg, encodeVarint(14, uint64(b.st.Index("gpu_time")))...)

	// 2. Samples
	for _, s := range b.samples {
		var sMsg []byte
		for _, locID := range s.locationIDs {
			sMsg = append(sMsg, encodeVarint(1, locID)...)
		}
		values := s.values
		if !b.commandBuffers {
			values = values[:2]
		}
		for _, val := range values {
			sMsg = append(sMsg, encodeVarint(2, uint64(val))...)
		}
		for _, lbl := range s.labels {
			var lMsg []byte
			lMsg = append(lMsg, encodeVarint(1, uint64(lbl.key))...)
			if lbl.str != 0 {
				lMsg = append(lMsg, encodeVarint(2, uint64(lbl.str))...)
			}
			sMsg = append(sMsg, encodeLengthDelimited(3, lMsg)...)
		}
		msg = append(msg, encodeLengthDelimited(2, sMsg)...)
	}

	// 3. Mappings
	for _, m := range b.mappings {
		var mMsg []byte
		mMsg = append(mMsg, encodeVarint(1, m.id)...)
		mMsg = append(mMsg, encodeVarint(2, m.memoryStart)...)
		mMsg = append(mMsg, encodeVarint(3, m.memoryLimit)...)
		mMsg = append(mMsg, encodeVarint(4, m.fileOffset)...)
		mMsg = append(mMsg, encodeVarint(5, uint64(m.filename))...)
		msg = append(msg, encodeLengthDelimited(3, mMsg)...)
	}

	// 4. Locations
	for _, loc := range b.locations {
		var locMsg []byte
		locMsg = append(locMsg, encodeVarint(1, loc.id)...)
		locMsg = append(locMsg, encodeVarint(2, loc.mappingID)...)
		locMsg = append(locMsg, encodeVarint(3, loc.address)...)
		for _, l := range loc.lines {
			var lineMsg []byte
			lineMsg = append(lineMsg, encodeVarint(1, l.functionID)...)
			lineMsg = append(lineMsg, encodeVarint(2, uint64(l.line))...)
			locMsg = append(locMsg, encodeLengthDelimited(4, lineMsg)...)
		}
		msg = append(msg, encodeLengthDelimited(4, locMsg)...)
	}

	// 5. Functions
	for _, fn := range b.functions {
		var fnMsg []byte
		fnMsg = append(fnMsg, encodeVarint(1, fn.id)...)
		fnMsg = append(fnMsg, encodeVarint(2, uint64(fn.name))...)
		fnMsg = append(fnMsg, encodeVarint(3, uint64(fn.systemName))...)
		fnMsg = append(fnMsg, encodeVarint(4, uint64(fn.filename))...)
		fnMsg = append(fnMsg, encodeVarint(5, uint64(fn.startLine))...)
		msg = append(msg, encodeLengthDelimited(5, fnMsg)...)
	}

	// 6. String table
	for _, s := range b.st.strings {
		msg = append(msg, encodeString(6, s)...)
	}

	// 7. Period type: gpu_time / nanoseconds
	var pt []byte
	pt = append(pt, encodeVarint(1, uint64(b.st.Index("gpu_time")))...)
	pt = append(pt, encodeVarint(2, uint64(b.st.Index("nanoseconds")))...)
	msg = append(msg, encodeLengthDelimited(11, pt)...)
	msg = append(msg, encodeVarint(12, 1000)...) // period = 1000ns

	if _, err := gw.Write(msg); err != nil {
		return fmt.Errorf("metalprof: write compressed profile: %w", err)
	}
	return nil
}

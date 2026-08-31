package cudaprof

import (
	"compress/gzip"
	"encoding/binary"
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

// Builder constructs a pprof profile containing GPU kernel, PTX, and SASS samples.
type Builder struct {
	st        *stringTable
	locations []location
	functions []function
	samples   []sample
}

type function struct {
	id       uint64
	name     int64
	file     int64
	line     int64
}

type location struct {
	id         uint64
	address    uint64
	functionID uint64
	line       int64
}

type sample struct {
	locationID uint64
	value      int64
	labels     map[string]string
}

// NewBuilder initializes a profile builder.
func NewBuilder() *Builder {
	return &Builder{
		st: newStringTable(),
	}
}

// AddSample records a GPU execution sample with source attribution and optional pprof labels.
func (b *Builder) AddSample(kernelName, filename string, line int64, sassPC uint64, count int64, labels map[string]string) {
	fnID := uint64(len(b.functions) + 1)
	b.functions = append(b.functions, function{
		id:   fnID,
		name: b.st.Index(kernelName),
		file: b.st.Index(filename),
		line: line,
	})

	locID := uint64(len(b.locations) + 1)
	b.locations = append(b.locations, location{
		id:         locID,
		address:    sassPC,
		functionID: fnID,
		line:       line,
	})

	b.samples = append(b.samples, sample{
		locationID: locID,
		value:      count,
		labels:     labels,
	})
}

// WriteGzip encodes the profile as a standard gzipped pprof profile.
func (b *Builder) WriteGzip(w io.Writer) error {
	var body []byte

	// SampleType: samples / count
	var sampleType []byte
	sampleType = append(sampleType, encodeVarint(1, uint64(b.st.Index("samples")))...)
	sampleType = append(sampleType, encodeVarint(2, uint64(b.st.Index("count")))...)
	body = append(body, encodeLengthDelimited(1, sampleType)...)

	// Samples
	for _, s := range b.samples {
		var sBody []byte
		sBody = append(sBody, encodeVarint(1, s.locationID)...) // location_id
		sBody = append(sBody, encodeVarint(2, uint64(s.value))...) // value

		for k, v := range s.labels {
			var lBody []byte
			lBody = append(lBody, encodeVarint(1, uint64(b.st.Index(k)))...)
			lBody = append(lBody, encodeVarint(2, uint64(b.st.Index(v)))...)
			sBody = append(sBody, encodeLengthDelimited(3, lBody)...)
		}
		body = append(body, encodeLengthDelimited(2, sBody)...)
	}

	// Locations
	for _, loc := range b.locations {
		var lBody []byte
		lBody = append(lBody, encodeVarint(1, loc.id)...)
		lBody = append(lBody, encodeVarint(3, loc.address)...) // address (SASS PC)

		// Line
		var lineBody []byte
		lineBody = append(lineBody, encodeVarint(1, loc.functionID)...)
		lineBody = append(lineBody, encodeVarint(2, uint64(loc.line))...)
		lBody = append(lBody, encodeLengthDelimited(4, lineBody)...)

		body = append(body, encodeLengthDelimited(4, lBody)...)
	}

	// Functions
	for _, fn := range b.functions {
		var fBody []byte
		fBody = append(fBody, encodeVarint(1, fn.id)...)
		fBody = append(fBody, encodeVarint(2, uint64(fn.name))...)
		fBody = append(fBody, encodeVarint(4, uint64(fn.file))...)
		fBody = append(fBody, encodeVarint(5, uint64(fn.line))...)
		body = append(body, encodeLengthDelimited(5, fBody)...)
	}

	// String table
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

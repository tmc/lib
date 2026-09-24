package safetensors

import (
	"bufio"
	"bytes"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
)

// A Dtype names a tensor element type as it appears in a safetensors header.
type Dtype string

// The element types this package accepts. They are the types
// safetensors.torch.load_file maps to a torch dtype.
const (
	BOOL   Dtype = "BOOL"
	U8     Dtype = "U8"
	I8     Dtype = "I8"
	F8E4M3 Dtype = "F8_E4M3"
	F8E5M2 Dtype = "F8_E5M2"
	I16    Dtype = "I16"
	U16    Dtype = "U16"
	F16    Dtype = "F16"
	BF16   Dtype = "BF16"
	I32    Dtype = "I32"
	U32    Dtype = "U32"
	F32    Dtype = "F32"
	I64    Dtype = "I64"
	U64    Dtype = "U64"
	F64    Dtype = "F64"
)

// maxHead bounds the header length, as the reference implementation does.
const maxHead = 100 << 20

// Size returns the size of one element in bytes, or 0 if d is not a known dtype.
func (d Dtype) Size() int {
	switch d {
	case BOOL, U8, I8, F8E4M3, F8E5M2:
		return 1
	case I16, U16, F16, BF16:
		return 2
	case I32, U32, F32:
		return 4
	case I64, U64, F64:
		return 8
	}
	return 0
}

// A Tensor is a named tensor with its raw little-endian bytes.
type Tensor struct {
	Name  string
	Dtype Dtype
	Shape []int
	Data  []byte
}

// metadataKey is the reserved header key holding the string metadata map.
const metadataKey = "__metadata__"

// byteSize returns the number of bytes a tensor of dtype and shape occupies.
func byteSize(name string, dtype Dtype, shape []int) (int64, error) {
	size := int64(dtype.Size())
	if size == 0 {
		return 0, fmt.Errorf("tensor %q: unknown dtype %q", name, dtype)
	}
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("tensor %q: negative dimension in shape %v", name, shape)
		}
		if d != 0 && size > math.MaxInt64/int64(d) {
			return 0, fmt.Errorf("tensor %q: shape %v overflows", name, shape)
		}
		size *= int64(d)
	}
	return size, nil
}

// Write writes tensors and metadata to w in the safetensors format. Tensor
// data is laid out in the order given, contiguously after the header. The
// header is padded with spaces so that the data section starts at a
// multiple of 8 bytes from the beginning of the file.
//
// Write reports an error, before writing anything, if a name is empty,
// repeated, or "__metadata__", if a dtype is unknown, or if a tensor's Data
// length does not match its shape. A nil or empty metadata map writes no
// "__metadata__" entry.
func Write(w io.Writer, tensors []Tensor, metadata map[string]string) error {
	header, err := encodeHeader(tensors, metadata)
	if err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(header)))
	if _, err := w.Write(n[:]); err != nil {
		return fmt.Errorf("safetensors: write header length: %w", err)
	}
	if _, err := w.Write(header); err != nil {
		return fmt.Errorf("safetensors: write header: %w", err)
	}
	for _, t := range tensors {
		if _, err := w.Write(t.Data); err != nil {
			return fmt.Errorf("safetensors: write tensor %q: %w", t.Name, err)
		}
	}
	return nil
}

func encodeHeader(tensors []Tensor, metadata map[string]string) ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	if len(metadata) > 0 {
		m, err := json.Marshal(metadata)
		if err != nil {
			return nil, fmt.Errorf("encode metadata: %w", err)
		}
		b.WriteString(`"` + metadataKey + `":`)
		b.Write(m)
	}
	seen := make(map[string]bool, len(tensors))
	var offset int64
	for i, t := range tensors {
		switch {
		case t.Name == "":
			return nil, fmt.Errorf("tensor %d has an empty name", i)
		case t.Name == metadataKey:
			return nil, fmt.Errorf("tensor name %q is reserved", t.Name)
		case seen[t.Name]:
			return nil, fmt.Errorf("duplicate tensor name %q", t.Name)
		}
		seen[t.Name] = true
		size, err := byteSize(t.Name, t.Dtype, t.Shape)
		if err != nil {
			return nil, err
		}
		if int64(len(t.Data)) != size {
			return nil, fmt.Errorf("tensor %q: %d data bytes, want %d for %s%v", t.Name, len(t.Data), size, t.Dtype, t.Shape)
		}
		if b.Len() > 1 {
			b.WriteByte(',')
		}
		name, err := json.Marshal(t.Name)
		if err != nil {
			return nil, fmt.Errorf("encode tensor name: %w", err)
		}
		b.Write(name)
		b.WriteString(`:{"dtype":"` + string(t.Dtype) + `","shape":[`)
		for j, d := range t.Shape {
			if j > 0 {
				b.WriteByte(',')
			}
			b.WriteString(strconv.Itoa(d))
		}
		b.WriteString(`],"data_offsets":[` + strconv.FormatInt(offset, 10) + "," + strconv.FormatInt(offset+size, 10) + "]}")
		offset += size
	}
	b.WriteByte('}')
	for (8+b.Len())%8 != 0 {
		b.WriteByte(' ')
	}
	return b.Bytes(), nil
}

// WriteFile writes tensors and metadata to the named file, as [Write] does.
// It writes to a temporary file in the same directory, syncs it, and renames
// it over name, so a reader sees either the old file or the complete new one.
func WriteFile(name string, tensors []Tensor, metadata map[string]string) (err error) {
	dir, base := filepath.Split(name)
	if dir == "" {
		dir = "."
	}
	f, err := os.CreateTemp(dir, "."+base+".tmp*")
	if err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := Write(bw, tensors, metadata); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	if err := f.Chmod(0o644); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	if err := os.Rename(f.Name(), name); err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	if d, err := os.Open(dir); err == nil {
		d.Sync()
		d.Close()
	}
	return nil
}

// TensorInfo describes one tensor in a [File]. Offset is relative to the start
// of the data section, which follows the header.
type TensorInfo struct {
	Name   string
	Dtype  Dtype
	Shape  []int
	Offset int64
	Size   int64
}

// A File is an opened safetensors file. Its header has been read and
// validated; tensor bytes are read on demand by [File.Tensor].
type File struct {
	// Metadata is the header's "__metadata__" map, or nil if it has none.
	Metadata map[string]string

	// Tensors lists the tensors in the order their data appears in the file.
	Tensors []TensorInfo

	r     io.ReaderAt
	base  int64
	index map[string]int
	c     io.Closer
}

// Open opens the named safetensors file and reads its header.
func Open(name string) (*File, error) {
	osf, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	st, err := osf.Stat()
	if err != nil {
		osf.Close()
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	f, err := NewFile(osf, st.Size())
	if err != nil {
		osf.Close()
		return nil, fmt.Errorf("%w (%s)", err, name)
	}
	f.c = osf
	return f, nil
}

// NewFile reads and validates the header of the size-byte safetensors file
// held by r. The header must describe tensors that exactly tile the data
// section, as safetensors.torch.load_file requires: no gaps, no overlap, and
// no trailing bytes. Duplicate names are an error.
func NewFile(r io.ReaderAt, size int64) (*File, error) {
	var n [8]byte
	if _, err := r.ReadAt(n[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	hlen := binary.LittleEndian.Uint64(n[:])
	if hlen > maxHead || int64(hlen) > size-8 {
		return nil, fmt.Errorf("safetensors: header length %d invalid for %d-byte file", hlen, size)
	}
	header := make([]byte, hlen)
	if _, err := r.ReadAt(header, 8); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	f := &File{r: r, base: 8 + int64(hlen), index: make(map[string]int)}
	if err := f.decodeHeader(header); err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	if err := f.checkLayout(size - f.base); err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	return f, nil
}

func (f *File) decodeHeader(header []byte) error {
	dec := json.NewDecoder(bytes.NewReader(header))
	dec.UseNumber()
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return errors.New("header is not a JSON object")
	}
	seen := make(map[string]bool)
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return fmt.Errorf("decode header: %w", err)
		}
		name := tok.(string)
		if seen[name] {
			return fmt.Errorf("duplicate header key %q", name)
		}
		seen[name] = true
		if name == metadataKey {
			if err := dec.Decode(&f.Metadata); err != nil {
				return fmt.Errorf("decode metadata: %w", err)
			}
			continue
		}
		var h struct {
			Dtype       Dtype   `json:"dtype"`
			Shape       []int   `json:"shape"`
			DataOffsets []int64 `json:"data_offsets"`
		}
		if err := dec.Decode(&h); err != nil {
			return fmt.Errorf("decode tensor %q: %w", name, err)
		}
		if h.Shape == nil || len(h.DataOffsets) != 2 {
			return fmt.Errorf("tensor %q: missing shape or data_offsets", name)
		}
		size, err := byteSize(name, h.Dtype, h.Shape)
		if err != nil {
			return err
		}
		start, end := h.DataOffsets[0], h.DataOffsets[1]
		if start < 0 || end < start || end-start != size {
			return fmt.Errorf("tensor %q: data_offsets [%d,%d] do not hold %d bytes", name, start, end, size)
		}
		f.Tensors = append(f.Tensors, TensorInfo{Name: name, Dtype: h.Dtype, Shape: h.Shape, Offset: start, Size: size})
	}
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("decode header: %w", err)
	}
	if rest := header[dec.InputOffset():]; len(bytes.TrimRight(rest, " \t\r\n")) != 0 {
		return errors.New("trailing data after header object")
	}
	return nil
}

func (f *File) checkLayout(dataSize int64) error {
	slices.SortStableFunc(f.Tensors, func(a, b TensorInfo) int {
		return cmp.Or(cmp.Compare(a.Offset, b.Offset), cmp.Compare(a.Size, b.Size))
	})
	var next int64
	for i, t := range f.Tensors {
		if t.Offset != next {
			return fmt.Errorf("tensor %q starts at %d, want %d: data is not contiguous", t.Name, t.Offset, next)
		}
		next = t.Offset + t.Size
		f.index[t.Name] = i
	}
	if next != dataSize {
		return fmt.Errorf("tensors cover %d of %d data bytes", next, dataSize)
	}
	return nil
}

// Info describes the named tensor without reading its data.
func (f *File) Info(name string) (TensorInfo, bool) {
	i, ok := f.index[name]
	if !ok {
		return TensorInfo{}, false
	}
	return f.Tensors[i], true
}

// Tensor reads the named tensor. Its Data holds the file's bytes unchanged.
func (f *File) Tensor(name string) (Tensor, error) {
	i, ok := f.index[name]
	if !ok {
		return Tensor{}, fmt.Errorf("safetensors: no tensor %q", name)
	}
	info := f.Tensors[i]
	data := make([]byte, info.Size)
	if _, err := f.r.ReadAt(data, f.base+info.Offset); err != nil && !(err == io.EOF && len(data) == 0) {
		return Tensor{}, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}
	return Tensor{Name: info.Name, Dtype: info.Dtype, Shape: slices.Clone(info.Shape), Data: data}, nil
}

// Close closes the underlying file if the File came from [Open].
func (f *File) Close() error {
	if f.c == nil {
		return nil
	}
	return f.c.Close()
}

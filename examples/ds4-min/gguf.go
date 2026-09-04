package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	maxMetadata = 1 << 20
	maxTensors  = 1 << 20
	maxString   = 1 << 24
	maxArray    = 1 << 20
)

var (
	errInvalidGGUF = errors.New("invalid GGUF")
	errUnsupported = errors.New("unsupported GGUF encoding")
)

const (
	pinnedModelBytes    = 86720111488
	pinnedModelSHA256   = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	pinnedMetadataHash  = "857d48d201da0faaaa6f2af0cdf0aa5fef523323217e29a902579130c24a32b5"
	pinnedTensorHash    = "0bcce58e7234ba89f247bfab49387121f375e8322d843fc166fb95ed12db99ca"
	pinnedQuantHash     = "b07c8ece42d1ff0273a8b7db85514e66ae6aca48dc55ea5b33d50f8f02380788"
	pinnedTokenizerHash = "d58f8cbcd40bfa9319e0954e01ad619c179567cf001258423713d3d604998eec"
)

type profileDigests struct {
	metadata  string
	tensors   string
	quant     string
	tokenizer string
}

type metadataEntry struct {
	Key    string `json:"key"`
	TypeID uint32 `json:"type_id"`
	Value  any    `json:"value"`
}

type tensorEntry struct {
	Name       string   `json:"name"`
	Dimensions []uint64 `json:"dimensions"`
	TypeID     uint32   `json:"type_id"`
	Offset     uint64   `json:"offset"`
	Bytes      uint64   `json:"bytes"`
}

type quantEntry struct {
	Bytes   uint64 `json:"bytes"`
	Tensors uint64 `json:"tensors"`
	TypeID  uint32 `json:"type_id"`
}

type ggufFile struct {
	metadata   map[string]ggufValue
	tensors    []ggufTensor
	dataOffset uint64
	size       uint64
}

type ggufValue struct {
	typeID uint32
	value  any
}

type ggufTensor struct {
	name       string
	dimensions []uint64
	typeID     uint32
	offset     uint64
}

func readGGUF(r io.ReaderAt, size int64) (*ggufFile, error) {
	if size < 0 {
		return nil, fmt.Errorf("%w: negative file size", errInvalidGGUF)
	}
	c := ggufCursor{r: r, size: uint64(size)}
	magic, err := c.bytes(4)
	if err != nil {
		return nil, err
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("%w: bad magic", errInvalidGGUF)
	}
	version, err := c.u32()
	if err != nil {
		return nil, err
	}
	if version != 3 {
		return nil, fmt.Errorf("%w: version %d", errUnsupported, version)
	}
	tensorCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	metadataCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	if tensorCount > maxTensors || metadataCount > maxMetadata {
		return nil, fmt.Errorf("%w: unreasonable counts", errInvalidGGUF)
	}
	metadataBytes, ok := multiply(metadataCount, 13) // empty key, type, and smallest value
	if !ok || metadataBytes > c.size-c.off {
		return nil, fmt.Errorf("%w: metadata count exceeds file size", errInvalidGGUF)
	}
	g := &ggufFile{metadata: make(map[string]ggufValue, metadataCount), size: uint64(size)}
	for range metadataCount {
		key, err := c.string()
		if err != nil {
			return nil, fmt.Errorf("metadata key: %w", err)
		}
		if _, ok := g.metadata[key]; ok {
			return nil, fmt.Errorf("%w: duplicate metadata key %q", errInvalidGGUF, key)
		}
		if len(key) > 1<<16-1 {
			return nil, fmt.Errorf("%w: metadata key too long", errInvalidGGUF)
		}
		value, err := c.value(0)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		g.metadata[key] = value
	}
	minimumTensorBytes, ok := multiply(tensorCount, 32) // empty name and one dimension
	if !ok || minimumTensorBytes > c.size-c.off {
		return nil, fmt.Errorf("%w: tensor count exceeds file size", errInvalidGGUF)
	}
	g.tensors = make([]ggufTensor, 0, tensorCount)
	names := make(map[string]bool, tensorCount)
	for range tensorCount {
		name, err := c.string()
		if err != nil {
			return nil, fmt.Errorf("tensor name: %w", err)
		}
		if names[name] {
			return nil, fmt.Errorf("%w: duplicate tensor %q", errInvalidGGUF, name)
		}
		if len(name) > 64 {
			return nil, fmt.Errorf("%w: tensor name too long", errInvalidGGUF)
		}
		names[name] = true
		n, err := c.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q dimensions: %w", name, err)
		}
		if n == 0 || n > 4 {
			return nil, fmt.Errorf("%w: tensor %q has %d dimensions", errInvalidGGUF, name, n)
		}
		t := ggufTensor{name: name, dimensions: make([]uint64, n)}
		for i := range t.dimensions {
			t.dimensions[i], err = c.u64()
			if err != nil || t.dimensions[i] == 0 {
				return nil, fmt.Errorf("%w: tensor %q dimension %d", errInvalidGGUF, name, i)
			}
		}
		t.typeID, err = c.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q type: %w", name, err)
		}
		t.offset, err = c.u64()
		if err != nil {
			return nil, fmt.Errorf("tensor %q offset: %w", name, err)
		}
		g.tensors = append(g.tensors, t)
	}
	alignment := uint64(32)
	if v, ok := g.metadata["general.alignment"]; ok {
		if v.typeID != 4 {
			return nil, fmt.Errorf("%w: alignment is not uint32", errInvalidGGUF)
		}
		n, ok := valueUint(v)
		if !ok || n == 0 || n > 1<<20 || n%8 != 0 {
			return nil, fmt.Errorf("%w: invalid alignment", errInvalidGGUF)
		}
		alignment = n
	}
	g.dataOffset = c.off
	if tensorCount != 0 {
		g.dataOffset, err = align(c.off, alignment)
	}
	if err != nil || g.dataOffset > g.size {
		return nil, fmt.Errorf("%w: tensor data offset", errInvalidGGUF)
	}
	expectedOffset := uint64(0)
	for _, t := range g.tensors {
		if t.offset%alignment != 0 {
			return nil, fmt.Errorf("%w: tensor %q offset is not aligned", errInvalidGGUF, t.name)
		}
		if t.offset != expectedOffset {
			return nil, fmt.Errorf("%w: tensor %q offset %d, want %d", errInvalidGGUF, t.name, t.offset, expectedOffset)
		}
		length, err := tensorBytes(t)
		if err != nil {
			return nil, err
		}
		end, ok := add(t.offset, length)
		if !ok || end > g.size-g.dataOffset {
			return nil, fmt.Errorf("%w: tensor %q offset outside file", errInvalidGGUF, t.name)
		}
		expectedOffset, err = align(end, alignment)
		if err != nil {
			return nil, fmt.Errorf("%w: tensor data size overflow", errInvalidGGUF)
		}
	}
	return g, nil
}

func validatePinnedProfile(g *ggufFile) error {
	if g.size != pinnedModelBytes {
		return fmt.Errorf("model size %d, want %d", g.size, uint64(pinnedModelBytes))
	}
	d, err := profileHashes(g)
	if err != nil {
		return err
	}
	checks := []struct {
		name, got, want string
	}{
		{"metadata", d.metadata, pinnedMetadataHash},
		{"tensor", d.tensors, pinnedTensorHash},
		{"quantization", d.quant, pinnedQuantHash},
		{"tokenizer", d.tokenizer, pinnedTokenizerHash},
	}
	for _, check := range checks {
		if check.got != check.want {
			return fmt.Errorf("%s manifest sha256 %s, want %s", check.name, check.got, check.want)
		}
	}
	return nil
}

func validatePinnedModel(r io.ReaderAt, size int64) (*ggufFile, error) {
	g, err := readGGUF(r, size)
	if err != nil {
		return nil, err
	}
	if err := validatePinnedProfile(g); err != nil {
		return nil, err
	}
	if err := checkSHA256(r, size, pinnedModelSHA256); err != nil {
		return nil, err
	}
	return g, nil
}

func checkSHA256(r io.ReaderAt, size int64, want string) error {
	if size < 0 {
		return fmt.Errorf("%w: negative file size", errInvalidGGUF)
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(r, 0, size)); err != nil {
		return fmt.Errorf("hash model: %w", err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("model sha256 %s, want %s", got, want)
	}
	return nil
}

func profileHashes(g *ggufFile) (profileDigests, error) {
	const schema = "ds4-min-gguf-manifest-v1"
	keys := make([]string, 0, len(g.metadata))
	for key := range g.metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	metadata := make([]metadataEntry, 0, len(keys))
	tokenizer := make([]metadataEntry, 0)
	for _, key := range keys {
		v := g.metadata[key]
		e := metadataEntry{key, v.typeID, v.value}
		metadata = append(metadata, e)
		if strings.HasPrefix(key, "tokenizer.") {
			tokenizer = append(tokenizer, e)
		}
	}
	tensors := make([]tensorEntry, 0, len(g.tensors))
	quantByType := make(map[uint32]quantEntry)
	for _, t := range g.tensors {
		n, err := tensorBytes(t)
		if err != nil {
			return profileDigests{}, err
		}
		tensors = append(tensors, tensorEntry{t.name, t.dimensions, t.typeID, t.offset, n})
		q := quantByType[t.typeID]
		q.TypeID = t.typeID
		q.Tensors++
		var ok bool
		q.Bytes, ok = add(q.Bytes, n)
		if !ok {
			return profileDigests{}, fmt.Errorf("quantization type %d byte count overflow", t.typeID)
		}
		quantByType[t.typeID] = q
	}
	types := make([]int, 0, len(quantByType))
	for typeID := range quantByType {
		types = append(types, int(typeID))
	}
	sort.Ints(types)
	quant := make([]quantEntry, 0, len(types))
	for _, typeID := range types {
		quant = append(quant, quantByType[uint32(typeID)])
	}
	values := []any{
		struct {
			Schema     string          `json:"schema"`
			FileBytes  uint64          `json:"file_bytes"`
			DataOffset uint64          `json:"data_offset"`
			Metadata   []metadataEntry `json:"metadata"`
		}{schema, g.size, g.dataOffset, metadata},
		struct {
			Schema     string        `json:"schema"`
			FileBytes  uint64        `json:"file_bytes"`
			DataOffset uint64        `json:"data_offset"`
			Tensors    []tensorEntry `json:"tensors"`
		}{schema, g.size, g.dataOffset, tensors},
		struct {
			Schema string       `json:"schema"`
			Types  []quantEntry `json:"types"`
		}{schema, quant},
		struct {
			Metadata []metadataEntry `json:"metadata"`
			Schema   string          `json:"schema"`
		}{tokenizer, schema},
	}
	var d profileDigests
	out := []*string{&d.metadata, &d.tensors, &d.quant, &d.tokenizer}
	for i, value := range values {
		b, err := json.Marshal(value)
		if err != nil {
			return profileDigests{}, fmt.Errorf("encode profile manifest: %w", err)
		}
		sum := sha256.Sum256(b)
		*out[i] = hex.EncodeToString(sum[:])
	}
	return d, nil
}

type ggufType struct{ block, bytes uint64 }

var ggufTypes = map[uint32]ggufType{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20},
	6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 40},
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176},
	14: {256, 210}, 15: {256, 292}, 16: {256, 66}, 17: {256, 74},
	18: {256, 98}, 19: {256, 110}, 20: {256, 50}, 21: {256, 110},
	22: {256, 82}, 23: {256, 136}, 24: {1, 1}, 25: {1, 2},
	26: {1, 4}, 27: {1, 8}, 28: {1, 8}, 29: {256, 56},
	30: {1, 2}, 39: {32, 17},
}

func tensorBytes(t ggufTensor) (uint64, error) {
	typ, ok := ggufTypes[t.typeID]
	if !ok {
		return 0, fmt.Errorf("%w: tensor %q type %d", errUnsupported, t.name, t.typeID)
	}
	if t.dimensions[0]%typ.block != 0 {
		return 0, fmt.Errorf("%w: tensor %q row is not block aligned", errInvalidGGUF, t.name)
	}
	n, ok := multiply(t.dimensions[0]/typ.block, typ.bytes)
	if !ok {
		return 0, fmt.Errorf("%w: tensor %q byte count overflow", errInvalidGGUF, t.name)
	}
	for _, d := range t.dimensions[1:] {
		var ok bool
		n, ok = multiply(n, d)
		if !ok {
			return 0, fmt.Errorf("%w: tensor %q byte count overflow", errInvalidGGUF, t.name)
		}
	}
	return n, nil
}

func valueUint(v ggufValue) (uint64, bool) {
	switch n := v.value.(type) {
	case uint8:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint32:
		return uint64(n), true
	case uint64:
		return n, true
	}
	return 0, false
}

type ggufCursor struct {
	r    io.ReaderAt
	size uint64
	off  uint64
}

func (c *ggufCursor) bytes(n uint64) ([]byte, error) {
	end, ok := add(c.off, n)
	if !ok || end > c.size || n > uint64(math.MaxInt) {
		return nil, io.ErrUnexpectedEOF
	}
	b := make([]byte, int(n))
	if _, err := c.r.ReadAt(b, int64(c.off)); err != nil {
		return nil, err
	}
	c.off = end
	return b, nil
}

func (c *ggufCursor) u32() (uint32, error) {
	b, err := c.bytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (c *ggufCursor) u64() (uint64, error) {
	b, err := c.bytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (c *ggufCursor) string() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > maxString {
		return "", fmt.Errorf("%w: string too long", errInvalidGGUF)
	}
	b, err := c.bytes(n)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", fmt.Errorf("%w: string is not UTF-8", errInvalidGGUF)
	}
	return string(b), nil
}

func (c *ggufCursor) value(depth int) (ggufValue, error) {
	if depth > 8 {
		return ggufValue{}, fmt.Errorf("%w: nested array", errUnsupported)
	}
	t, err := c.u32()
	if err != nil {
		return ggufValue{}, err
	}
	return c.valueOfType(t, depth)
}

func (c *ggufCursor) valueOfType(t uint32, depth int) (ggufValue, error) {
	if depth > 8 {
		return ggufValue{}, fmt.Errorf("%w: metadata array nesting is too deep", errInvalidGGUF)
	}
	v := ggufValue{typeID: t}
	var err error
	switch t {
	case 0:
		b, err := c.bytes(1)
		if err != nil {
			return v, err
		}
		v.value = b[0]
	case 1:
		b, err := c.bytes(1)
		if err != nil {
			return v, err
		}
		v.value = int8(b[0])
	case 2:
		b, err := c.bytes(2)
		if err != nil {
			return v, err
		}
		v.value = binary.LittleEndian.Uint16(b)
	case 3:
		b, err := c.bytes(2)
		if err != nil {
			return v, err
		}
		v.value = int16(binary.LittleEndian.Uint16(b))
	case 4:
		v.value, err = c.u32()
	case 5:
		n, e := c.u32()
		err = e
		v.value = int32(n)
	case 6:
		n, e := c.u32()
		err = e
		v.value = math.Float32frombits(n)
	case 7:
		b, e := c.bytes(1)
		err = e
		if e == nil {
			if b[0] > 1 {
				return v, fmt.Errorf("%w: invalid boolean", errInvalidGGUF)
			}
			v.value = b[0] == 1
		}
	case 8:
		v.value, err = c.string()
	case 9:
		v.value, err = c.array(depth + 1)
	case 10:
		v.value, err = c.u64()
	case 11:
		n, e := c.u64()
		err = e
		v.value = int64(n)
	case 12:
		n, e := c.u64()
		err = e
		v.value = math.Float64frombits(n)
	default:
		return v, fmt.Errorf("%w: value type %d", errUnsupported, t)
	}
	return v, err
}

func (c *ggufCursor) array(depth int) (any, error) {
	t, err := c.u32()
	if err != nil {
		return nil, err
	}
	n, err := c.u64()
	if err != nil {
		return nil, err
	}
	if n > maxArray {
		return nil, fmt.Errorf("%w: array too long", errInvalidGGUF)
	}
	minimum, ok := minimumValueBytes(t)
	if !ok {
		return nil, fmt.Errorf("%w: value type %d", errUnsupported, t)
	}
	need, ok := multiply(n, minimum)
	if !ok || need > c.size-c.off {
		return nil, fmt.Errorf("%w: array exceeds file size", errInvalidGGUF)
	}
	values := make([]any, 0, n)
	for range n {
		v, err := c.valueOfType(t, depth)
		if err != nil {
			return nil, err
		}
		values = append(values, v.value)
	}
	return values, nil
}

func minimumValueBytes(t uint32) (uint64, bool) {
	switch t {
	case 0, 1, 7:
		return 1, true
	case 2, 3:
		return 2, true
	case 4, 5, 6:
		return 4, true
	case 8, 10, 11, 12:
		return 8, true
	case 9:
		return 12, true
	default:
		return 0, false
	}
}

func add(a, b uint64) (uint64, bool) {
	if a > math.MaxUint64-b {
		return 0, false
	}
	return a + b, true
}

func multiply(a, b uint64) (uint64, bool) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, false
	}
	return a * b, true
}

func align(n, alignment uint64) (uint64, error) {
	rem := n % alignment
	if rem == 0 {
		return n, nil
	}
	v, ok := add(n, alignment-rem)
	if !ok {
		return 0, errors.New("offset overflow")
	}
	return v, nil
}

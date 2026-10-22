package dsv41

// TensorDescriptor describes a validated serialized tensor, including Engram
// tables. Dimensions are in GGUF order (the contiguous dimension first).
// Encoding is the GGUF type ID. Offset is absolute in the bound file, and Bytes
// is the serialized extent. A descriptor does not expose or own tensor data.
type TensorDescriptor struct {
	Name       string
	Encoding   uint32
	Dimensions []uint64
	Offset     int64
	Bytes      int64
}

// Tensor looks up a required tensor validated by Bind. The returned dimensions
// belong to the caller; changing them does not change the model or later lookups.
// A nil or unbound model, or an unknown name, returns a zero descriptor and false.
func (m *Model) Tensor(name string) (TensorDescriptor, bool) {
	if m == nil {
		return TensorDescriptor{}, false
	}
	t, ok := m.tensors[name]
	if !ok {
		return TensorDescriptor{}, false
	}
	t.Dimensions = append([]uint64(nil), t.Dimensions...)
	return t, true
}

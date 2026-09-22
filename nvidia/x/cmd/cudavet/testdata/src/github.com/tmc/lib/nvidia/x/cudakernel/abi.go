// Package cudakernel is a declaration-only stand-in for the real package, so
// that analysistest fixtures type-check in GOPATH mode. cargalign reads the
// offsets out of the real cudakernel linked into the analyzer; the types here
// only have to carry the same names.
package cudakernel

type Param struct {
	Name      string
	Type      string
	ByteWidth int
	Alignment int
	IsPointer bool
	Offset    int
}

type Schema struct {
	Params          []Param
	TotalParamBytes int
}

func NewSchema(params ...Param) (Schema, error) { return Schema{Params: params}, nil }

package metalx

import (
	"fmt"
	"reflect"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

// pipelineType is the reflect type of the fields Bind fills in.
var pipelineType = reflect.TypeOf((*apple.MTLComputePipelineState)(nil)).Elem()

// Bind builds a compute pipeline state for every kernel named by the struct
// pointed to by kernels and stores it in that struct. Each exported
// [apple.MTLComputePipelineState] field is looked up in lib under the name
// given by its "metal" struct tag, or under the field name when the tag is
// absent. A field tagged `metal:"-"` is skipped.
//
// Bind reports the first kernel it cannot resolve, so a renamed or
// dead-stripped kernel surfaces as an error naming the field instead of a nil
// pipeline state that fails at dispatch:
//
//	type kernels struct {
//		Embed apple.MTLComputePipelineState `metal:"EmbeddingLookup"`
//		Norm  apple.MTLComputePipelineState `metal:"ComputeRMSNorm"`
//	}
//	var k kernels
//	if err := metalx.Bind(dev, lib, &k); err != nil {
//		return err
//	}
//
// Fields of other types are ignored unless they carry a "metal" tag, which is
// reported as an error.
func Bind(dev *metal.Device, lib apple.MTLLibrary, kernels any) error {
	rv := reflect.ValueOf(kernels)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("metalx: bind: want non-nil pointer to struct, got %T", kernels)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf("metalx: bind: want pointer to struct, got %T", kernels)
	}
	fields, err := kernelFields(rv.Type())
	if err != nil {
		return err
	}
	if dev == nil {
		return fmt.Errorf("metalx: bind: nil device")
	}
	if lib == nil {
		return fmt.Errorf("metalx: bind: nil library")
	}
	for _, f := range fields {
		pso, err := dev.NewComputePipeline(lib, f.kernel)
		if err != nil {
			return fmt.Errorf("metalx: bind kernel %q (field %s): %w", f.kernel, f.field, err)
		}
		rv.Field(f.index).Set(reflect.ValueOf(pso))
	}
	return nil
}

// kernelField records one struct field Bind fills in.
type kernelField struct {
	index  int
	field  string
	kernel string
}

// kernelFields reports the fields of t that Bind resolves, in declaration
// order. It rejects a struct whose tagged fields cannot be filled in.
func kernelFields(t reflect.Type) ([]kernelField, error) {
	var fields []kernelField
	for i := range t.NumField() {
		sf := t.Field(i)
		tag, tagged := sf.Tag.Lookup("metal")
		if tag == "-" {
			continue
		}
		if sf.Type != pipelineType {
			if tagged {
				return nil, fmt.Errorf("metalx: bind: field %s has type %s, want apple.MTLComputePipelineState", sf.Name, sf.Type)
			}
			continue
		}
		if !sf.IsExported() {
			return nil, fmt.Errorf("metalx: bind: field %s is unexported", sf.Name)
		}
		name := tag
		if name == "" {
			name = sf.Name
		}
		fields = append(fields, kernelField{index: i, field: sf.Name, kernel: name})
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("metalx: bind: %s has no apple.MTLComputePipelineState fields", t)
	}
	return fields, nil
}

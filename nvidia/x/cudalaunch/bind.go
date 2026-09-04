package cudalaunch

import (
	"fmt"
	"reflect"

	"github.com/tmc/lib/nvidia/driver"
)

// functionType is the reflect type of the fields Bind fills in.
var functionType = reflect.TypeOf(driver.Function(0))

// Bind resolves every kernel named by the struct pointed to by kernels and stores
// the resulting functions in it. Each exported [driver.Function] field is looked up
// in mod under the name given by its "cuda" struct tag, or under the field name when
// the tag is absent. A field tagged `cuda:"-"` is skipped.
//
// Bind reports the first kernel it cannot resolve, so a renamed or dead-stripped
// kernel surfaces as an error instead of a zero function that fails at launch:
//
//	type kernels struct {
//		Embed driver.Function `cuda:"EmbeddingLookup"`
//		Norm  driver.Function `cuda:"RMSNorm"`
//	}
//	var k kernels
//	if err := cudalaunch.Bind(mod, &k); err != nil {
//		return err
//	}
//
// Fields of other types are ignored unless they carry a "cuda" tag, which is
// reported as an error.
func Bind(mod driver.Module, kernels any) error {
	rv := reflect.ValueOf(kernels)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("cudalaunch: bind: want non-nil pointer to struct, got %T", kernels)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf("cudalaunch: bind: want pointer to struct, got %T", kernels)
	}
	fields, err := kernelFields(rv.Type())
	if err != nil {
		return err
	}
	for _, f := range fields {
		fn, err := mod.GetFunction(f.kernel)
		if err != nil {
			return fmt.Errorf("cudalaunch: bind kernel %q (field %s): %w", f.kernel, f.field, err)
		}
		rv.Field(f.index).SetUint(uint64(fn))
	}
	return nil
}

// kernelField records one struct field Bind fills in.
type kernelField struct {
	index  int
	field  string
	kernel string
}

// kernelFields reports the fields of t that Bind resolves, in declaration order.
// It rejects a struct whose tagged fields cannot be filled in.
func kernelFields(t reflect.Type) ([]kernelField, error) {
	var fields []kernelField
	for i := range t.NumField() {
		sf := t.Field(i)
		tag, tagged := sf.Tag.Lookup("cuda")
		if tag == "-" {
			continue
		}
		if sf.Type != functionType {
			if tagged {
				return nil, fmt.Errorf("cudalaunch: bind: field %s has type %s, want driver.Function", sf.Name, sf.Type)
			}
			continue
		}
		if !sf.IsExported() {
			return nil, fmt.Errorf("cudalaunch: bind: field %s is unexported", sf.Name)
		}
		name := tag
		if name == "" {
			name = sf.Name
		}
		fields = append(fields, kernelField{index: i, field: sf.Name, kernel: name})
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("cudalaunch: bind: %s has no driver.Function fields", t)
	}
	return fields, nil
}

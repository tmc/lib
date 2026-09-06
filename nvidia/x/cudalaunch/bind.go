package cudalaunch

import (
	"fmt"
	"reflect"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
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
	return bind(kernels, functionType, func(name string) (reflect.Value, error) {
		fn, err := mod.GetFunction(name)
		return reflect.ValueOf(fn), err
	})
}

// BindOwned resolves exported *cuda.Kernel fields using their cuda tags, as in Bind.
// Each kernel retains mod and rejects launches after mod is closed.
func BindOwned(mod *cuda.Module, kernels any) error {
	return bind(kernels, reflect.TypeOf((*cuda.Kernel)(nil)), func(name string) (reflect.Value, error) {
		fn, err := mod.Function(name)
		return reflect.ValueOf(fn), err
	})
}

func bind(kernels any, typ reflect.Type, resolve func(string) (reflect.Value, error)) error {
	rv := reflect.ValueOf(kernels)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return fmt.Errorf("cudalaunch: bind: want non-nil pointer to struct, got %T", kernels)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf("cudalaunch: bind: want pointer to struct, got %T", kernels)
	}
	fields, err := typedKernelFields(rv.Type(), typ)
	if err != nil {
		return err
	}
	for _, f := range fields {
		fn, err := resolve(f.kernel)
		if err != nil {
			return fmt.Errorf("cudalaunch: bind kernel %q (field %s): %w", f.kernel, f.field, err)
		}
		rv.Field(f.index).Set(fn)
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
	return typedKernelFields(t, functionType)
}

func typedKernelFields(t, typ reflect.Type) ([]kernelField, error) {
	var fields []kernelField
	for i := range t.NumField() {
		sf := t.Field(i)
		tag, tagged := sf.Tag.Lookup("cuda")
		if tag == "-" {
			continue
		}
		if sf.Type != typ {
			if tagged {
				return nil, fmt.Errorf("cudalaunch: bind: field %s has type %s, want %s", sf.Name, sf.Type, typ)
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
		return nil, fmt.Errorf("cudalaunch: bind: %s has no %s fields", t, typ)
	}
	return fields, nil
}

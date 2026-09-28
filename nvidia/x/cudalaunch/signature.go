package cudalaunch

import (
	"fmt"
	"reflect"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// Param describes an expected kernel parameter declaration.
type Param struct {
	Name      string // Parameter name as declared in PTX (e.g. "vec4_param_0")
	Type      string // PTX data type (e.g. "u32", "u64", "f32", "b8")
	Size      int    // Parameter size in bytes
	IsPointer bool   // True if the parameter represents a memory pointer
}

func (p Param) String() string {
	if p.IsPointer {
		return fmt.Sprintf("*%s %s", p.Type, p.Name)
	}
	return fmt.Sprintf("%s %s", p.Type, p.Name)
}

// Signature describes a kernel's parameter signature.
type Signature struct {
	KernelName string
	Params     []Param
}

func (s Signature) String() string {
	var b strings.Builder
	b.WriteString(s.KernelName)
	b.WriteByte('(')
	for i, p := range s.Params {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(p.String())
	}
	b.WriteByte(')')
	return b.String()
}

// SignatureFromPTX extracts parameter counts and widths for the named kernel.
// PTX parameter names do not establish whether an integer holds a pointer.
func SignatureFromPTX(ptx string, kernelName string) (Signature, error) {
	mod, err := cudaptx.Parse(strings.NewReader(ptx))
	if err != nil {
		return Signature{}, fmt.Errorf("cudalaunch: parse ptx: %w", err)
	}
	k := mod.Kernel(kernelName)
	if k == nil {
		return Signature{}, fmt.Errorf("cudalaunch: kernel %q not found in PTX", kernelName)
	}
	return kernelToSignature(k), nil
}

// ExtractSignaturesFromPTX parses PTX source and returns signatures for all kernels found.
func ExtractSignaturesFromPTX(ptx string) (map[string]Signature, error) {
	mod, err := cudaptx.Parse(strings.NewReader(ptx))
	if err != nil {
		return nil, fmt.Errorf("cudalaunch: parse ptx: %w", err)
	}
	sigs := make(map[string]Signature)
	for _, k := range mod.Kernels {
		if !k.Entry {
			continue
		}
		if _, ok := sigs[k.Name]; ok && !k.Body {
			continue
		}
		sigs[k.Name] = kernelToSignature(k)
	}
	return sigs, nil
}

func kernelToSignature(k *cudaptx.Kernel) Signature {
	sig := Signature{
		KernelName: k.Name,
		Params:     make([]Param, len(k.Params)),
	}
	for i, p := range k.Params {
		sig.Params[i] = Param{
			Name: p.Name,
			Type: p.Type,
			Size: ptxParamByteSize(p),
		}
	}
	return sig
}

func ptxTypeByteSize(t string) int {
	switch t {
	case "b8", "u8", "s8", "pred":
		return 1
	case "b16", "u16", "s16", "f16", "bf16":
		return 2
	case "b32", "u32", "s32", "f32", "tf32", "f16x2", "bf16x2":
		return 4
	case "b64", "u64", "s64", "f64":
		return 8
	case "b128":
		return 16
	default:
		return 0
	}
}

func ptxParamByteSize(p cudaptx.Param) int {
	elemSize := ptxTypeByteSize(p.Type)
	if p.Size > 0 {
		if elemSize == 0 {
			elemSize = 1
		}
		return p.Size * elemSize
	}
	if elemSize > 0 {
		return elemSize
	}
	if p.Align > 0 {
		return p.Align
	}
	return 0
}

// Validate checks that the provided launch arguments match the kernel signature.
// It verifies argument count, sizes, and scalar types. Pointer intent must be supplied explicitly.
func Validate(sig Signature, args ...any) error {
	if len(args) != len(sig.Params) {
		if sig.KernelName != "" {
			return fmt.Errorf("cudalaunch: argument count mismatch: kernel %q expects %d arguments, got %d", sig.KernelName, len(sig.Params), len(args))
		}
		return fmt.Errorf("cudalaunch: argument count mismatch: kernel expects %d arguments, got %d", len(sig.Params), len(args))
	}

	for i, arg := range args {
		p := sig.Params[i]
		if arg == nil {
			return fmt.Errorf("cudalaunch: argument %d (%s) is nil", i, p.Name)
		}

		info, err := inspectArgument(arg)
		if err != nil {
			return fmt.Errorf("cudalaunch: argument %d (%s): %w", i, p.Name, err)
		}

		// Validate size.
		if p.Size > 0 && info.size != p.Size {
			return fmt.Errorf("cudalaunch: argument %d (%s) size mismatch: kernel expects %d bytes (%s), got %d bytes (%T)",
				i, p.Name, p.Size, p.Type, info.size, arg)
		}

		// PTX bit types carry storage width, not a source-language scalar type.
		if !p.IsPointer {
			switch p.Type {
			case "b8", "b16", "b32", "b64", "b128":
				continue
			}
		}

		// Validate pointer vs scalar.
		if p.IsPointer {
			if !info.isPointer {
				return fmt.Errorf("cudalaunch: argument %d (%s) type mismatch: kernel expects pointer (%s), got %T",
					i, p.Name, p.Type, arg)
			}
		} else {
			// Scalar parameter.
			if info.isPointer && !info.isInteger && !isIntegerPTXType(p.Type) {
				return fmt.Errorf("cudalaunch: argument %d (%s) type mismatch: kernel expects scalar (%s), got pointer (%T)",
					i, p.Name, p.Type, arg)
			}
			if isFloatPTXType(p.Type) && !info.isFloat {
				return fmt.Errorf("cudalaunch: argument %d (%s) type mismatch: kernel expects %s, got %T",
					i, p.Name, p.Type, arg)
			}
			if isIntegerPTXType(p.Type) && info.isFloat {
				return fmt.Errorf("cudalaunch: argument %d (%s) type mismatch: kernel expects %s, got %T",
					i, p.Name, p.Type, arg)
			}
			if isIntegerPTXType(p.Type) && info.isStruct && p.Type != "b8" {
				return fmt.Errorf("cudalaunch: argument %d (%s) type mismatch: kernel expects %s, got struct %T",
					i, p.Name, p.Type, arg)
			}
		}
	}

	return nil
}

type argInfo struct {
	size      int
	isPointer bool
	isInteger bool
	isFloat   bool
	isBool    bool
	isStruct  bool
}

func inspectArgument(arg any) (argInfo, error) {
	// Every device pointer packs as an 8-byte slot regardless of its value;
	// the address is not inspected. See the packing loop in launch.go for why
	// no alignment rule belongs here.
	if _, ok := arg.(DevicePtrProvider); ok {
		return argInfo{size: 8, isPointer: true}, nil
	}

	switch arg.(type) {
	case driver.DevicePtr:
		return argInfo{size: 8, isPointer: true}, nil
	case int8, uint8:
		return argInfo{size: 1, isInteger: true}, nil
	case bool:
		return argInfo{size: 1, isBool: true, isInteger: true}, nil
	case int16, uint16:
		return argInfo{size: 2, isInteger: true}, nil
	case int32, uint32:
		return argInfo{size: 4, isInteger: true}, nil
	case int64, uint64:
		return argInfo{size: 8, isInteger: true}, nil
	case int, uint:
		return argInfo{size: int(unsafe.Sizeof(int(0))), isInteger: true}, nil
	case uintptr:
		return argInfo{size: int(unsafe.Sizeof(uintptr(0))), isPointer: true, isInteger: true}, nil
	case float32:
		return argInfo{size: 4, isFloat: true}, nil
	case float64:
		return argInfo{size: 8, isFloat: true}, nil
	default:
		rv := reflect.ValueOf(arg)
		if rv.Kind() == reflect.Pointer {
			if rv.IsNil() {
				return argInfo{}, fmt.Errorf("is a nil pointer")
			}
			return argInfo{size: int(unsafe.Sizeof(uintptr(0))), isPointer: true}, nil
		}
		if rv.Kind() == reflect.UnsafePointer {
			if rv.IsNil() {
				return argInfo{}, fmt.Errorf("is a nil pointer")
			}
			return argInfo{size: int(unsafe.Sizeof(uintptr(0))), isPointer: true}, nil
		}
		if rv.Kind() == reflect.Struct {
			return argInfo{size: int(rv.Type().Size()), isStruct: true}, nil
		}
		return argInfo{}, fmt.Errorf("has unsupported type %T", arg)
	}
}

func isFloatPTXType(t string) bool {
	switch t {
	case "f16", "f16x2", "f32", "f64", "bf16", "bf16x2", "tf32":
		return true
	default:
		return false
	}
}

func isIntegerPTXType(t string) bool {
	switch t {
	case "u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64", "b8", "b16", "b32", "b64", "b128":
		return true
	default:
		return false
	}
}

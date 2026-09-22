package cudalineinfo

import "testing"

// TestUnitKernelMangled covers the spellings a disassembly function answers
// to. A
// __global__ function without extern "C" reaches the disassembly mangled, so
// a caller naming it as the source writes it must still match; the mangled
// names below are the ones nvcc emits for the two kernels of
// handWrittenSource.
func TestUnitKernelMangled(t *testing.T) {
	tests := []struct {
		name string
		unit Unit
		fn   string
		want bool
	}{
		{"extern C unmangled", Unit{Kernels: []string{"geglu", "rmsnorm"}}, "geglu", true},
		{"mangled by plain name", Unit{Kernels: []string{"geglu", "rmsnorm"}}, "_Z5gegluPKfPfi", true},
		{"mangled by plain name, second", Unit{Kernels: []string{"geglu", "rmsnorm"}}, "_Z7rmsnormPKfPfi", true},
		{"mangled, other kernel", Unit{Kernels: []string{"rmsnorm"}}, "_Z5gegluPKfPfi", false},
		{"mangled by symbol", Unit{Kernels: []string{"_Z5gegluPKfPfi"}}, "_Z5gegluPKfPfi", true},
		{"mangled, no signature match", Unit{Kernels: []string{"geglu(float const*, float*, int)"}}, "_Z5gegluPKfPfi", false},
		{"namespaced, qualified", Unit{Kernels: []string{"ns::geglu"}}, "_ZN2ns5gegluEPKfPfi", true},
		{"namespaced, unqualified", Unit{Kernels: []string{"geglu"}}, "_ZN2ns5gegluEPKfPfi", true},
		{"template argument dropped", Unit{Kernels: []string{"foo"}}, "_Z3fooIiEvT_", true},
		{"not a symbol", Unit{Kernels: []string{"geglu"}}, "not a symbol", false},
		{"empty kernel name matches nothing", Unit{Kernels: []string{""}}, "not a symbol", false},
		{"empty function name", Unit{Kernels: []string{"geglu"}}, "", false},
		{"unmangled, other kernel", Unit{Kernels: []string{"geglu"}}, "rmsnorm", false},

		// The zero-Kernels fallback: Name alone, and neither.
		{"no kernels, name matches", Unit{Name: "forward"}, "forward", true},
		{"no kernels, name differs", Unit{Name: "forward"}, "geglu", false},
		{"no kernels, no name", Unit{}, "anything", true},
		// The fallback is a literal match: it does not demangle.
		{"no kernels, mangled", Unit{Name: "geglu"}, "_Z5gegluPKfPfi", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.unit.Kernel(tt.fn); got != tt.want {
				t.Errorf("Unit%+v.Kernel(%q) = %v, want %v", tt.unit, tt.fn, got, tt.want)
			}
		})
	}
}

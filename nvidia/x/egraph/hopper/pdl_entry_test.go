package hopper

import (
	"slices"
	"strings"
	"testing"
)

func TestAddPDLEntry(t *testing.T) {
	head := Prelude + PDLHelpers()
	tests := []struct {
		name  string
		src   string
		want  []string // rewritten kernels
		wrong bool
	}{
		{"plain", "__global__ void __launch_bounds__(256) k(\n    int n, float* x) {\n    x[0] = n;\n}\n", []string{"k"}, false},
		{"template", "template <int M>\n__global__ void __launch_bounds__(128) g(\n    const float* a) {\n}\n", []string{"g"}, false},
		{"cluster", "__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) c(\n    float* x) {\n}\n", []string{"c"}, false},
		{"extern", "extern \"C\" __global__ void e(const uint16_t* t, int h) {\n}\n", []string{"e"}, false},
		{"bare", "__global__ void b(float* x) { x[0] = 1; }\n", []string{"b"}, false},
		{"declaration", "__global__ void d(float* x);\n__global__ void k(float* x) {\n}\n", []string{"k"}, false},
		{"comment", "// a __global__ void fake(int) { mention\n__global__ void k(int n) {\n}\n", []string{"k"}, false},
		{"two", "__global__ void a(int n) {\n}\n\n__global__ void b(int n) {\n}\n", []string{"a", "b"}, false},
		{"none", "EG_DEV void f() {}\n", nil, true},
		{"no body", "__global__ void k(int n)\n", nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, names, err := AddPDLEntry(head + tt.src)
			if tt.wrong {
				if err == nil {
					t.Fatalf("AddPDLEntry succeeded, want error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(names, tt.want) {
				t.Errorf("names = %v, want %v", names, tt.want)
			}
			if n := strings.Count(got, pdlEntry); n != len(tt.want) {
				t.Errorf("source has %d entries, want %d:\n%s", n, len(tt.want), got[len(head):])
			}
			if !strings.HasPrefix(got, head) {
				t.Errorf("prelude changed")
			}
		})
	}
}

func TestAddPDLEntryNeedsHelpers(t *testing.T) {
	if _, _, err := AddPDLEntry(Prelude + "__global__ void k(int n) {\n}\n"); err == nil {
		t.Fatal("AddPDLEntry without PDLHelpers succeeded")
	}
}

// TestAddPDLEntryPlan checks that every kernel of a generated plan is
// rewritten and that each wait comes before the kernel's first statement.
func TestAddPDLEntryPlan(t *testing.T) {
	const h, n = 256, 512
	b := NewBuilder()
	x := b.Input("x", F32, 1, h)
	norm := b.Cast(BF16, b.RMSNorm(x, b.Input("norm", F32, h), h, 1e-5))
	y := b.MatMul(norm, b.Input("w", BF16, n, h))
	e, err := Optimize(b.G, y)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(e, "pdl_entry", F32)
	if err != nil {
		t.Fatal(err)
	}
	var src strings.Builder
	src.WriteString(Prelude + PDLHelpers())
	var want []string
	for _, k := range p.Kernels {
		var cu string
		switch {
		case k.GEMM != nil:
			cu, err = k.GEMM.CUDA()
			want = append(want, k.GEMM.Name)
		case k.Tile != nil:
			cu, err = k.Tile.CUDA()
			want = append(want, k.Tile.Name)
		case k.Row != nil:
			cu, err = k.Row.CUDA()
			want = append(want, k.Row.Name)
		default:
			t.Fatalf("unexpected kernel %+v", k)
		}
		if err != nil {
			t.Fatal(err)
		}
		src.WriteString("\n" + cu + "\n")
	}
	got, names, err := AddPDLEntry(src.String())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(names, want) {
		t.Fatalf("names = %v, want %v", names, want)
	}
	for _, name := range names {
		i := strings.Index(got, " "+name+"(")
		if i < 0 {
			t.Fatalf("kernel %s missing", name)
		}
		body := got[i+strings.Index(got[i:], "{")+1:]
		if !strings.HasPrefix(body, pdlEntry) {
			t.Errorf("kernel %s body starts %.60q, want the PDL entry", name, body)
		}
	}
}

package dsv41

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestRotaryPinnedC executes the pinned kernel body and its frequency-table
// constructor on the host. It does not qualify device transcendental functions.
func TestRotaryPinnedC(t *testing.T) {
	source := pinned41(t)
	start := strings.Index(source, "struct dsv41_rope_args {")
	if start < 0 {
		t.Fatal("rotary argument declaration missing")
	}
	end := strings.Index(source[start:], "};")
	if start < 0 || end < 0 {
		t.Fatal("rotary argument declaration missing")
	}
	args := source[start : start+end+2]
	start = strings.Index(source, "    struct frequencies {")
	if start < 0 {
		t.Fatal("rotary table constructor missing")
	}
	end = strings.Index(source[start:], "    static const frequencies table;")
	if start < 0 || end < 0 {
		t.Fatal("rotary table constructor missing")
	}
	table := source[start : start+end]
	f := literalFunction(t, source, "dsv41_bf16") + literalFunction(t, source, "dsv41_rope_kernel")
	prefix := operationPrefix + "\nstatic float __fsub_rn(float x,float y){volatile float z=x-y;return z;}\n" + args + table
	for _, compressed := range []bool{false, true} {
		for _, inverse := range []bool{false, true} {
			for _, position := range []int{0, 1, 64, 65536, 1048575} {
				t.Run(fmt.Sprintf("compressed=%t/inverse=%t/position=%d", compressed, inverse, position), func(t *testing.T) {
					x := make([]float32, 2*128)
					for i := range x {
						x[i] = BF16(float32(math.Sin(float64(i)*.13)) * 2)
					}
					want := append([]float32(nil), x...)
					if err := rotary(want, 128, position, compressed, inverse); err != nil {
						t.Fatal(err)
					}
					kind, inv := 0, 0
					if compressed {
						kind = 1
					}
					if inverse {
						inv = 1
					}
					main := fmt.Sprintf(`int main(){float x[256];if(fread(x,4,256,stdin)!=256)return 2;frequencies table;dsv41_rope_args a={128,2,1,%d,1,%d,{0}};memcpy(a.frequencies,table.value[%d],sizeof(a.frequencies));blockIdx.x=0;for(unsigned i=0;i<64;i++){threadIdx.x=i;dsv41_rope_kernel(x,a);}return fwrite(x,4,256,stdout)==256?0:3;}`, position, inv, kind)
					got := runLiteral(t, prefix+f+main, x)
					if !equalBits(got, want) {
						t.Fatal("rotary literal mismatch")
					}
					// Change pairing without accessing beyond the allocation.
					mutant := strings.Replace(f, "const float re = x[i], im = x[i + 1u];", "const float re = x[i], im = x[i];", 1)
					if mutant == f {
						t.Fatal("rotary pairing mutant missing")
					}
					if equalBits(runLiteral(t, prefix+mutant+main, x), want) {
						t.Fatal("rotary pairing mutant not detected")
					}
				})
			}
		}
	}
}

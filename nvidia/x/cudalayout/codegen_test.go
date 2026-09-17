package cudalayout

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This executes emitted indexing as ordinary C++, not CUDA. It detects unit
// and indexing errors without making a CUDA compilation or GPU claim.
func TestHostGeneratedIndexing(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudalayout-host-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	for _, elem := range []int{1, 4} {
		t.Run(fmt.Sprint(elem), func(t *testing.T) {
			tile, err := NewTile(7, 13, elem, WithStride(128/elem), WithSwizzle(Swizzle64B))
			if err != nil {
				t.Fatal(err)
			}
			source, err := FixtureSource(tile)
			if err != nil {
				t.Fatal(err)
			}
			typeName := "unsigned char"
			if elem == 4 {
				typeName = "float"
			}
			program := `#define __global__
struct dim { unsigned int x; } blockIdx, blockDim, threadIdx;
` + source + fmt.Sprintf(`
int main() {
  %s in[91], out[%d];
  for (int i=0;i<91;i++) in[i]=i+1;
  for (int i=0;i<%d;i++) out[i]=0;
  blockDim.x=32;
  for (blockIdx.x=0;blockIdx.x<3;blockIdx.x++)
    for(threadIdx.x=0;threadIdx.x<32;threadIdx.x++) layout_copy(in,out);
  for (int p=0;p<%d;p++) {
    int row=p/%d, col=(p%%%d)^(row%%%d);
    int want=col<13 ? row*13+col+1 : 0;
    if(out[p]!=want) return 1;
  }
  return 0;
}
`, typeName, 7*128/elem, 7*128/elem, 7*128/elem, 128/elem, 128/elem, 64/elem)
			path := filepath.Join(dir, fmt.Sprintf("index%d.cc", elem))
			bin := path + ".out"
			if err := os.WriteFile(path, []byte(program), 0600); err != nil {
				t.Fatal(err)
			}
			if out, err := exec.Command(compiler, "-std=c++17", path, "-o", bin).CombinedOutput(); err != nil {
				t.Fatalf("host C++ compile: %v\n%s", err, out)
			}
			if out, err := exec.Command(bin).CombinedOutput(); err != nil {
				t.Fatalf("host indexing oracle: %v\n%s", err, out)
			}
		})
	}
}

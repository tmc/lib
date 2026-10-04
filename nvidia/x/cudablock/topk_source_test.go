package cudablock

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type topkMutation struct{ name, old, replacement, fixture string }

var topkMutations = []topkMutation{
	{"padding_winner", "for (int i = 0; i < n; ++i) {\n            bool used", "for (int i = 0; i < width; ++i) {\n            bool used", "stride_padding"},
	{"high_index_tie", "v == best_value && i < best", "v == best_value && i > best", "ties_zero_infinity"},
	{"wrong_stride", "(unsigned long long)row * row_stride", "(unsigned long long)row * width", "stride_padding"},
	{"omitted_nan_status", "status[row] = 1; return;", "status[row] = 0; return;", "nan_short"},
	{"offset_sentinel", "indices[out + j] = -1;", "indices[out + j] = -1 + offsets[row];", "indices_only"},
}

func topkMutated(m topkMutation) (string, error) {
	if strings.Count(TopKSource, m.old) != 1 {
		return "", fmt.Errorf("mutation %s does not match exactly once", m.name)
	}
	return strings.Replace(TopKSource, m.old, m.replacement, 1), nil
}

const topkHostShim = `
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#define __global__
struct { unsigned int x; } blockIdx, threadIdx;
float __int_as_float(unsigned int bits) { float v; std::memcpy(&v,&bits,4); return v; }
`

const topkHostMain = `
int main() {
    int h[6];
    if (std::fread(h,4,6,stdin)!=6) return 2;
    int rows=h[0], width=h[1], stride=h[2], k=h[3], values=h[4], size=h[5];
    std::vector<float> in(size), out_values(rows*k);
    std::vector<int> lengths(rows),offsets(rows),indices(rows*k),counts(rows),status(rows);
    if (std::fread(in.data(),4,size,stdin)!=(size_t)size ||
        std::fread(lengths.data(),4,rows,stdin)!=(size_t)rows ||
        std::fread(offsets.data(),4,rows,stdin)!=(size_t)rows) return 3;
    for (int row=0;row<rows;++row) { blockIdx.x=row; threadIdx.x=0;
        topk_fp32(in.data(),lengths.data(),offsets.data(),indices.data(),out_values.data(),counts.data(),status.data(),stride,width,k,rows,values);
    }
    std::fwrite(indices.data(),4,rows*k,stdout);
    std::fwrite(out_values.data(),4,rows*k,stdout);
    std::fwrite(counts.data(),4,rows,stdout);
    std::fwrite(status.data(),4,rows,stdout);
    return std::ferror(stdout) ? 4 : 0;
}
`

func topkBuildHost(t *testing.T, source string) string {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("C++ source emulation unavailable: clang++ missing")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "topk.cc")
	bin := filepath.Join(dir, "topk-host")
	if err := os.WriteFile(path, []byte(topkHostShim+source+topkHostMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", path, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("host emulation build: %v\n%s", err, out)
	}
	return bin
}

func topkRunHost(t *testing.T, bin string, f topkCase) topkAnswer {
	t.Helper()
	var input bytes.Buffer
	write := func(v any) {
		if err := binary.Write(&input, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	write([]int32{int32(f.c.Rows), int32(f.c.Width), int32(f.c.RowStride), int32(f.c.K), 0, int32(len(f.input))})
	if f.c.Values {
		data := input.Bytes()
		binary.LittleEndian.PutUint32(data[16:20], 1)
	}
	write(f.input)
	write(f.lengths)
	write(f.offsets)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin)
	cmd.Stdin = &input
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("host emulation run: %v (context: %v)", err, ctx.Err())
	}
	n := f.c.Rows * f.c.K
	a := topkAnswer{indices: make([]int32, n), values: make([]float32, n), counts: make([]int32, f.c.Rows), status: make([]int32, f.c.Rows)}
	r := bytes.NewReader(out)
	for _, v := range []any{a.indices, a.values, a.counts, a.status} {
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if r.Len() != 0 {
		t.Fatal("extra emulation output")
	}
	return a
}

func TestTopKSourceEmulation(t *testing.T) {
	bin := topkBuildHost(t, TopKSource)
	for _, f := range topkFixtures() {
		t.Run(f.name, func(t *testing.T) {
			got := topkRunHost(t, bin, f)
			want := topkSortReference(f)
			if err := topkCompare(f.c, got, want); err != nil {
				t.Fatalf("%v: got %+v want %+v", err, got, want)
			}
		})
	}
}

func TestTopKBehavioralSourceMutants(t *testing.T) {
	for _, m := range topkMutations {
		t.Run(m.name, func(t *testing.T) {
			source, err := topkMutated(m)
			if err != nil {
				t.Fatal(err)
			}
			bin := topkBuildHost(t, source)
			found := false
			for _, f := range topkFixtures() {
				if f.name != m.fixture {
					continue
				}
				found = true
				got, want := topkRunHost(t, bin, f), topkSortReference(f)
				if err := topkCompare(f.c, got, want); err == nil {
					t.Fatal("behavioral mutant survived")
				} else {
					t.Logf("rejected source mutant %s: %v; got indices=%v counts=%v status=%v; want indices=%v counts=%v status=%v", m.name, err, got.indices, got.counts, got.status, want.indices, want.counts, want.status)
				}
			}
			if !found {
				t.Fatal("missing control fixture")
			}
		})
	}
}

func TestTopKValueOracleControl(t *testing.T) {
	f := topkFixtures()[14]
	a := topkSortReference(f)
	a.values[4] = 0
	if topkCompare(f.c, a, topkSortReference(f)) == nil {
		t.Fatal("signed-zero corruption survived")
	}
	a = topkSortReference(f)
	a.values[0] = float32(math.Inf(-1))
	if topkCompare(f.c, a, topkSortReference(f)) == nil {
		t.Fatal("value corruption survived")
	}
}

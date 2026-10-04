package cudablock

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type topkBF16Case struct {
	topkCase
	bits []uint16
}

func topkBF16Mutations() []topkMutation {
	return append(append([]topkMutation(nil), topkMutations...), topkMutation{"wrong_storage_shift", "input[base + i] << 16", "input[base + i] << 15", "storage_neighbors"})
}

func topkBF16Mutated(m topkMutation) string {
	return strings.ReplaceAll(TopKBF16Source, m.old, m.replacement)
}

func topkBF16Fixtures() []topkBF16Case {
	var cases []topkBF16Case
	for _, f := range topkFixtures() {
		bits := make([]uint16, len(f.input))
		decoded := make([]float32, len(bits))
		for i, v := range f.input {
			bits[i] = uint16(math.Float32bits(v) >> 16)
			decoded[i] = math.Float32frombits(uint32(bits[i]) << 16)
		}
		f.input = decoded
		cases = append(cases, topkBF16Case{f, bits})
	}
	for _, x := range []struct {
		name string
		bits []uint16
	}{
		{"storage_neighbors", []uint16{0x0001, 0x0002, 0x8001, 0x8002, 0x0000, 0x8000, 0x3f80, 0x3f81}},
		{"signaling_nan", []uint16{0x3f80, 0x7f81}},
		{"negative_nan", []uint16{0x3f80, 0xffc1}},
	} {
		f := topkCase{name: x.name, c: TopKConfig{1, len(x.bits), len(x.bits), 32, true}, lengths: []int32{int32(len(x.bits))}, offsets: []int32{9}}
		for _, b := range x.bits {
			f.input = append(f.input, math.Float32frombits(uint32(b)<<16))
		}
		cases = append(cases, topkBF16Case{f, x.bits})
	}
	return cases
}

func topkBF16BuildHost(t *testing.T, source string) string {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("BF16 source emulation unavailable: clang++ missing")
	}
	dir := t.TempDir()
	path, bin := filepath.Join(dir, "bf16.cc"), filepath.Join(dir, "bf16-host")
	main := strings.ReplaceAll(topkHostMain, "std::vector<float> in(size), out_values(rows*k);", "std::vector<unsigned short> in(size); std::vector<float> out_values(rows*k);")
	main = strings.Replace(main, "std::fread(in.data(),4,size,stdin)", "std::fread(in.data(),2,size,stdin)", 1)
	main = strings.ReplaceAll(main, "topk_fp32(", "topk_bf16(")
	shim := topkHostShim + "float __uint_as_float(unsigned int bits) { return __int_as_float(bits); }\n"
	if err := os.WriteFile(path, []byte(shim+source+main), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", path, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("BF16 source emulation build: %v\n%s", err, out)
	}
	return bin
}

func topkBF16RunHost(t *testing.T, bin string, f topkBF16Case) topkAnswer {
	t.Helper()
	var input bytes.Buffer
	wantValues := int32(0)
	if f.c.Values {
		wantValues = 1
	}
	for _, v := range []any{[]int32{int32(f.c.Rows), int32(f.c.Width), int32(f.c.RowStride), int32(f.c.K), wantValues, int32(len(f.bits))}, f.bits, f.lengths, f.offsets} {
		if err := binary.Write(&input, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin)
	cmd.Stdin = &input
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	a := topkAnswer{indices: make([]int32, f.c.Rows*f.c.K), values: make([]float32, f.c.Rows*f.c.K), counts: make([]int32, f.c.Rows), status: make([]int32, f.c.Rows)}
	r := bytes.NewReader(out)
	for _, v := range []any{a.indices, a.values, a.counts, a.status} {
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if r.Len() != 0 {
		t.Fatal("extra BF16 source output")
	}
	return a
}

func TestTopKBF16SourceEmulation(t *testing.T) {
	bin := topkBF16BuildHost(t, TopKBF16Source)
	for _, f := range topkBF16Fixtures() {
		t.Run(f.name, func(t *testing.T) {
			if err := topkCompare(f.c, topkBF16RunHost(t, bin, f), topkSortReference(f.topkCase)); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTopKBF16ConversionMutant(t *testing.T) {
	source := strings.ReplaceAll(TopKBF16Source, "input[base + i] << 16", "input[base + i] << 15")
	bin := topkBF16BuildHost(t, source)
	for _, f := range topkBF16Fixtures() {
		if f.name == "storage_neighbors" {
			if topkCompare(f.c, topkBF16RunHost(t, bin, f), topkSortReference(f.topkCase)) == nil {
				t.Fatal("wrong BF16 bit shift survived")
			}
			return
		}
	}
	t.Fatal("missing storage fixture")
}

func TestTopKBF16BehavioralSourceMutants(t *testing.T) {
	for _, m := range topkMutations {
		t.Run(m.name, func(t *testing.T) {
			if strings.Count(TopKBF16Source, m.old) != 1 {
				t.Fatal("BF16 mutation must match exactly once")
			}
			source := strings.Replace(TopKBF16Source, m.old, m.replacement, 1)
			bin := topkBF16BuildHost(t, source)
			for _, f := range topkBF16Fixtures() {
				if f.name == m.fixture {
					if topkCompare(f.c, topkBF16RunHost(t, bin, f), topkSortReference(f.topkCase)) == nil {
						t.Fatal("BF16 behavioral mutant survived")
					}
					return
				}
			}
			t.Fatal("missing BF16 control fixture")
		})
	}
}

func TestTopKBF16BufferAdmission(t *testing.T) {
	dev := new(cuda.Device)
	input := topkBuffer{name: "input", ptr: 0x1002, length: 3, need: 3, owner: dev, bytes: 2, alignment: 2}
	output := topkBuffer{name: "output", ptr: 0x1008, length: 1, need: 1, owner: dev}
	if err := topkCheckBuffers([]topkBuffer{input, output}, dev); err != nil {
		t.Fatal(err)
	}
	for _, x := range []struct {
		name   string
		change func(*topkBuffer)
	}{
		{"odd", func(b *topkBuffer) { b.ptr++ }},
		{"short", func(b *topkBuffer) { b.length-- }},
		{"overflow", func(b *topkBuffer) { b.ptr = ^uint64(0) - 1 }},
		{"owner", func(b *topkBuffer) { b.owner = nil }},
		{"overlap", func(b *topkBuffer) { b.ptr = 0x1004 }},
	} {
		t.Run(x.name, func(t *testing.T) {
			b := input
			x.change(&b)
			if topkCheckBuffers([]topkBuffer{b, output}, dev) == nil {
				t.Fatal("invalid BF16 buffer admitted")
			}
		})
	}
}

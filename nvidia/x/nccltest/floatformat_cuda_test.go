package nccltest

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestCUDAFloatCodec(t *testing.T) {
	headers := os.Getenv("NCCLTEST_CODEC_HEADERS")
	if headers == "" {
		t.Skip("set NCCLTEST_CODEC_HEADERS to the CUDA header root to require native conversion checks")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(home, "tmp", "nccltest-codec")
	if err := os.MkdirAll(root, 0755); err != nil {
		t.Fatal(err)
	}
	work, err := os.MkdirTemp(root, "cuda-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained native fixture: %s", work)
	src, err := os.ReadFile("testdata/floatcodec.cc")
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(work, "floatcodec.cc")
	if err := os.WriteFile(source, src, 0644); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(work, "floatcodec")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	args := []string{"-std=c++17", "-I" + filepath.Join(headers, "cuda_cudart"), "-I" + filepath.Join(headers, "cuda_crt"), "-I" + filepath.Join(headers, "cuda_cccl"), source, "-o", binary}
	if out, err := exec.CommandContext(ctx, "clang++", args...).CombinedOutput(); err != nil {
		t.Fatalf("compile native conversions: %v\n%s", err, out)
	}
	formats := []floatFormat{float16Format, bfloat16Format, float8e4m3Format, float8e5m2Format}
	var values []float64
	seen := make(map[uint64]bool)
	add := func(v float64) {
		bits := math.Float64bits(v)
		if !seen[bits] {
			seen[bits] = true
			values = append(values, v)
		}
	}
	for _, v := range []float64{0, math.Copysign(0, -1), math.Inf(1), math.Inf(-1), math.NaN(), math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64} {
		add(v)
	}
	boundary := func(v float64) {
		for _, x := range []float64{v, math.Nextafter(v, math.Inf(-1)), math.Nextafter(v, math.Inf(1))} {
			add(x)
			add(-x)
		}
	}
	for _, f := range formats {
		for bits := uint16(0); bits <= f.maxFiniteBits(); bits++ {
			v := f.decode(bits)
			add(v)
			add(-v)
			if bits < f.maxFiniteBits() {
				boundary((v + f.decode(bits+1)) / 2)
			}
		}
		maximum := f.decode(f.maxFiniteBits())
		ulp := maximum - f.decode(f.maxFiniteBits()-1)
		boundary(maximum + ulp/2)
	}
	var input bytes.Buffer
	for _, value := range values {
		fmt.Fprintf(&input, "%016x\n", math.Float64bits(value))
	}
	if err := os.WriteFile(filepath.Join(work, "input.txt"), input.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, binary)
	cmd.Stdin = &input
	out, err := cmd.CombinedOutput()
	if e := os.WriteFile(filepath.Join(work, "output.txt"), out, 0644); e != nil {
		t.Fatal(e)
	}
	if err != nil {
		t.Fatalf("native conversion process: %v", err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(out))
	count := 0
	for scanner.Scan() {
		if count >= len(values) {
			t.Fatal("extra native output")
		}
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2*len(formats) {
			t.Fatalf("malformed native row %q", scanner.Text())
		}
		for i, f := range formats {
			want, err := strconv.ParseUint(fields[i], 16, 16)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := strconv.ParseUint(fields[i+len(formats)], 16, 64)
			if err != nil {
				t.Fatal(err)
			}
			gotDecoded := f.decode(uint16(want))
			if !(math.IsNaN(gotDecoded) && math.IsNaN(math.Float64frombits(decoded))) && math.Float64bits(gotDecoded) != decoded {
				t.Fatalf("format=%d decode bits=%04x got=%016x native=%016x", i, want, math.Float64bits(gotDecoded), decoded)
			}
			got := f.encode(values[count])
			if math.IsNaN(f.decode(got)) && math.IsNaN(f.decode(uint16(want))) {
				continue
			}
			if got != uint16(want) {
				t.Fatalf("format=%d input=%016x got=%04x native=%04x", i, math.Float64bits(values[count]), got, want)
			}
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if count != len(values) {
		t.Fatalf("native rows=%d, want %d", count, len(values))
	}
	t.Logf("checked %d distinct input bit patterns across %d CUDA formats (%d encoding and %d decoding checks)", count, len(formats), count*len(formats), count*len(formats))
}

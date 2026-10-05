//go:build native

package cudaoutputcoverage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const outputSource = `extern "C" __global__ void output_contract(float* out, int n, int corrupt) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned int)n) out[i] = corrupt && i == 29 ? -9.0f : (float)(i + 1);
}`

type preparedOutput struct {
	SourceDigest, PTXDigest, NVRTCPath, Arch string
	NVRTCMajor, NVRTCMinor                   int32
}

func hashBytes(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// Compile-only preparation makes no CUDA driver call. Its products must be
// frozen with the exact source/toolchain packet before native dispatch.
func TestOutputContract_Prepare(t *testing.T) {
	dir := os.Getenv("OUTPUT_CONTRACT_PREPARE_DIR")
	if dir == "" {
		t.Skip("compile-only output packet directory not selected")
	}
	compiled, err := nvrtcx.Compile(outputSource, nvrtcx.WithArch(12, 1))
	if err != nil {
		var owner io.Closer
		if errors.As(err, &owner) {
			if closeErr := owner.Close(); closeErr != nil {
				retainedOutputCompileOwners.Lock()
				retainedOutputCompileOwners.owners = append(retainedOutputCompileOwners.owners, owner)
				retainedOutputCompileOwners.Unlock()
				t.Logf("retain failed compiler cleanup owner: %v", closeErr)
			}
		}
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	ptx := []byte(compiled.PTX())
	if len(ptx) == 0 {
		t.Fatal("compiler returned empty PTX")
	}
	meta := preparedOutput{SourceDigest: hashBytes([]byte(outputSource)), PTXDigest: hashBytes(ptx), NVRTCPath: nvrtc.Path(), Arch: "sm_121", NVRTCMajor: major, NVRTCMinor: minor}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	for name, b := range map[string][]byte{"output.ptx": ptx, "output.cu": []byte(outputSource), "compile.log": []byte(compiled.Log())} {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "prepared.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("compile-only packet %+v", meta)
}

type outputFixture struct {
	dev       *cuda.Device
	module    driver.Module
	buffer    cudamem.Slice[uint32]
	allocated bool
}

var retainedOutputFixtures struct {
	sync.Mutex
	owners []*outputFixture
}

var retainedOutputCompileOwners struct {
	sync.Mutex
	owners []io.Closer
}

func (f *outputFixture) close() error {
	if f.dev != nil {
		if err := f.dev.WithContext(driver.CtxSynchronize); err != nil {
			return err
		}
	}
	if f.allocated {
		if err := f.buffer.Free(); err != nil {
			return err
		}
		f.allocated = false
	}
	if f.module != 0 {
		if err := f.dev.WithContext(f.module.Close); err != nil {
			return err
		}
		f.module = 0
	}
	if f.dev != nil {
		if err := f.dev.Close(); err != nil {
			return err
		}
		f.dev = nil
	}
	return nil
}

func newOutputFixture(t *testing.T) (*outputFixture, driver.Function, preparedOutput) {
	t.Helper()
	if os.Getenv("OUTPUT_CONTRACT_NATIVE") != "1" {
		t.Skip("native output contract dispatch not selected")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dir := os.Getenv("OUTPUT_CONTRACT_PACKET_DIR")
	if dir == "" {
		t.Fatal("missing prepared output packet")
	}
	b, err := os.ReadFile(filepath.Join(dir, "prepared.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta preparedOutput
	if err := json.Unmarshal(b, &meta); err != nil {
		t.Fatal(err)
	}
	ptx, err := os.ReadFile(filepath.Join(dir, "output.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	if meta.SourceDigest != hashBytes([]byte(outputSource)) || meta.PTXDigest != hashBytes(ptx) || meta.Arch != "sm_121" || meta.NVRTCPath == "" || meta.NVRTCMajor <= 0 {
		t.Fatal("prepared source or PTX identity mismatch")
	}
	f := &outputFixture{}
	t.Cleanup(func() {
		if err := f.close(); err != nil {
			retainedOutputFixtures.Lock()
			retainedOutputFixtures.owners = append(retainedOutputFixtures.owners, f)
			retainedOutputFixtures.Unlock()
			t.Errorf("retain failed fixture cleanup owners: %v", err)
		}
	})
	f.dev = cudatest.RequireDevice(t)
	major, minor, err := f.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Fatalf("prepared target sm_121 differs from sm_%d%d", major, minor)
	}
	err = f.dev.WithContext(func() error { var err error; f.module, err = driver.ModuleLoadData(append(ptx, 0)); return err })
	if err != nil {
		t.Fatal(err)
	}
	var fn driver.Function
	err = f.dev.WithContext(func() error { var err error; fn, err = f.module.Function("output_contract"); return err })
	if err != nil {
		t.Fatal(err)
	}
	f.buffer, err = cudamem.Alloc[uint32](f.dev, 1008)
	f.allocated = f.buffer.Ptr() != 0
	if err != nil {
		t.Fatal(err)
	}
	return f, fn, meta
}

func TestOutputContract_PairedNativeConsumer(t *testing.T) {
	f, fn, meta := newOutputFixture(t)
	const n = 1003
	for _, name := range []string{"floor", "corrupt", "healthy"} {
		t.Run(name, func(t *testing.T) {
			spec := recorded(n)
			spec.SourceDigest = meta.SourceDigest
			spec.Target = meta.Arch
			spec.MathMode = "NVRTC default finite FP32 integer conversion"
			if name == "floor" {
				spec.Geometry.Grid[0] = 3
			}
			corrupt := int32(0)
			if name == "corrupt" {
				corrupt = 1
			}
			input := make([]byte, 8)
			binary.LittleEndian.PutUint32(input, uint32(n))
			binary.LittleEndian.PutUint32(input[4:], uint32(corrupt))
			spec.InputDigest = hashBytes(input)
			launch := func(_ context.Context, fill uint32) ([]uint32, error) {
				base := make([]uint32, n+5)
				for i := range base {
					base[i] = fill
				}
				base[0] = math.Float32bits(77)
				base[len(base)-1] = math.Float32bits(-77)
				if err := f.buffer.CopyFromHost(base); err != nil {
					return nil, err
				}
				out := f.buffer.Slice(1, n+4)
				ptr := out.Ptr()
				nn := int32(n)
				if err := f.dev.WithContext(func() error {
					if err := fn.LaunchKernel(driver.Dims(spec.Geometry.Grid[0], spec.Geometry.Grid[1], spec.Geometry.Grid[2]), driver.Dims(spec.Geometry.Block[0], spec.Geometry.Block[1], spec.Geometry.Block[2]), 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr), unsafe.Pointer(&nn), unsafe.Pointer(&corrupt)}); err != nil {
						return err
					}
					return driver.CtxSynchronize()
				}); err != nil {
					return nil, err
				}
				if err := f.buffer.CopyToHost(base); err != nil {
					return nil, err
				}
				if base[0] != math.Float32bits(77) || base[len(base)-1] != math.Float32bits(-77) {
					return nil, fmt.Errorf("output guard overwritten")
				}
				for i := n + 1; i < n+4; i++ {
					if base[i] != fill {
						return nil, fmt.Errorf("excluded padding overwritten at %d", i)
					}
				}
				return append([]uint32(nil), base[1:n+4]...), nil
			}
			completed, written, err := Observe(context.Background(), spec, launch)
			if err != nil {
				t.Fatal(err)
			}
			// Ordinary value comparison runs the same kernel into reference residue.
			residue := make([]uint32, n+5)
			residue[0] = math.Float32bits(77)
			residue[len(residue)-1] = math.Float32bits(-77)
			copy(residue[1:], spec.Reference)
			if err := f.buffer.CopyFromHost(residue); err != nil {
				t.Fatal(err)
			}
			ptr := f.buffer.Slice(1, n+4).Ptr()
			nn := int32(n)
			if err := f.dev.WithContext(func() error {
				if err := fn.LaunchKernel(driver.Dims(spec.Geometry.Grid[0], spec.Geometry.Grid[1], spec.Geometry.Grid[2]), driver.Dims(spec.Geometry.Block[0], spec.Geometry.Block[1], spec.Geometry.Block[2]), 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr), unsafe.Pointer(&nn), unsafe.Pointer(&corrupt)}); err != nil {
					return err
				}
				return driver.CtxSynchronize()
			}); err != nil {
				t.Fatal(err)
			}
			if err := f.buffer.CopyToHost(residue); err != nil {
				t.Fatal(err)
			}
			if residue[0] != math.Float32bits(77) || residue[len(residue)-1] != math.Float32bits(-77) {
				t.Fatal("ordinary output guard overwritten")
			}
			for i := n + 1; i < n+4; i++ {
				if residue[i] != 0 {
					t.Fatal("ordinary excluded padding overwritten")
				}
			}
			paired, err := CheckOutput(completed, residue[1:n+4], policy(completed))
			if err != nil {
				t.Fatal(err)
			}
			if name == "floor" {
				if len(written.Unwritten) != 1 || written.Unwritten[0] != (Range{768, 1003}) || paired.NumericalMismatch != nil {
					t.Fatal(paired)
				}
			} else {
				if len(written.Unwritten) != 0 {
					t.Fatal(written)
				}
				if (paired.NumericalMismatch != nil) != (name == "corrupt") {
					t.Fatal(paired)
				}
			}
			b, err := json.Marshal(struct {
				Prepared  preparedOutput
				Readbacks TwoFill
				Ordinary  []uint32
				Report    PairedReport
			}{meta, completed, append([]uint32(nil), residue[1:n+4]...), paired})
			if err != nil {
				t.Fatal(err)
			}
			dir := os.Getenv("OUTPUT_CONTRACT_RECEIPT_DIR")
			if dir == "" {
				t.Fatal("missing native receipt directory")
			}
			if err := os.MkdirAll(dir, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name+".json"), b, 0600); err != nil {
				t.Fatal(err)
			}
			t.Logf("output elements written arm=%s target=%s grid=%v block=%v ptx=%s source=%s input=%s unwritten=%v numerical=%v", name, spec.Target, spec.Geometry.Grid, spec.Geometry.Block, meta.PTXDigest, spec.SourceDigest, spec.InputDigest, written.Unwritten, paired.NumericalMismatch)
		})
	}
}

//go:build native

package cudablock

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func topkExtensionResources(t *testing.T, owner *TopK) map[string]int {
	t.Helper()
	resources := make(map[string]int)
	for _, fact := range []struct {
		name      string
		attribute driver.FunctionAttribute
	}{
		{"allocated_registers_per_thread", driver.FuncAttributeNumRegs},
		{"local_bytes_per_thread", driver.FuncAttributeLocalSizeBytes},
		{"static_shared_bytes", driver.FuncAttributeSharedSizeBytes},
		{"max_threads_per_block", driver.FuncAttributeMaxThreadsPerBlock},
	} {
		value, err := owner.kernel.Attribute(fact.attribute)
		if err != nil {
			t.Fatal(err)
		}
		resources[fact.name] = value
	}
	return resources
}

// TestBuildTopKExtensionPacket compiles fixed BF16 and parallel FP32 images
// without launching. Use TOPK_BUILD_ONLY=1 and a fresh TOPK_EXTENSION_PACKET_DIR.
func TestBuildTopKExtensionPacket(t *testing.T) {
	if os.Getenv("TOPK_BUILD_ONLY") != "1" {
		t.Skip("offline extension build not requested")
	}
	dir, target := os.Getenv("TOPK_EXTENSION_PACKET_DIR"), os.Getenv("TOPK_TARGET")
	if !filepath.IsAbs(dir) || target == "" {
		t.Fatal("absolute extension packet directory and target required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(""); err != nil {
		t.Skipf("NVRTC unavailable: %v", err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	images := []struct{ name, source, entry string }{
		{"bf16", TopKBF16Source, "topk_bf16"},
		{"parallel", topkParallelSource, "topk_fp32"},
		{"ordinary", TopKSource, "topk_fp32"},
	}
	for _, m := range topkBF16Mutations() {
		images = append(images, struct{ name, source, entry string }{"bf16_mutant_" + m.name, topkBF16Mutated(m), "topk_bf16"})
	}
	for _, image := range images {
		a, err := topkCompileNamedArtifact(image.source, target, image.entry)
		if err != nil {
			t.Fatal(err)
		}
		r := topkPacketEntry{image.name, topkDigest([]byte(image.source)), a.Digest, target, nvrtc.Path(), major, minor, a.CompilerOpts}
		topkWriteJSON(t, filepath.Join(dir, image.name+".artifact.json"), a)
		topkWriteJSON(t, filepath.Join(dir, image.name+".receipt.json"), r)
		if err := os.WriteFile(filepath.Join(dir, image.name+".cu"), []byte(image.source), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func topkReadExtension(t *testing.T, name, source, entry, target string) (*cudakernel.Artifact, topkPacketEntry) {
	t.Helper()
	a, r, err := topkReadExtensionPacket(os.Getenv("TOPK_EXTENSION_PACKET_DIR"), name, source, entry, target)
	if err != nil {
		t.Fatal(err)
	}
	return a, r
}

func topkExtensionDevice(t *testing.T) (*cuda.Device, string, string) {
	t.Helper()
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("extension requires inherited gpu-lock: %v", err)
	}
	out := os.Getenv("TOPK_EXTENSION_OUTPUT_DIR")
	if !filepath.IsAbs(out) || !filepath.IsAbs(os.Getenv("TOPK_EXTENSION_PACKET_DIR")) {
		t.Skip("retained extension packet and output directories required")
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.Require(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	return dev, fmt.Sprintf("sm_%d%d", major, minor), out
}

func topkExtensionBuffers(t *testing.T, dev *cuda.Device, c TopKConfig) (TopKBuffers, []func()) {
	t.Helper()
	e, err := c.Extents()
	if err != nil {
		t.Fatal(err)
	}
	var checks []func()
	i := func(n int) cudamem.Slice[int32] {
		s, check := topkNativeInt(t, dev, n)
		checks = append(checks, check)
		return s
	}
	f := func(n int) cudamem.Slice[float32] {
		s, check := topkNativeFloat(t, dev, n)
		checks = append(checks, check)
		return s
	}
	b := TopKBuffers{Lengths: i(e.RowMetadata), Offsets: i(e.RowMetadata), ScratchIndices: i(e.Results), ScratchCounts: i(e.RowMetadata), Status: i(e.RowMetadata), Indices: i(e.Results), Counts: i(e.RowMetadata)}
	if c.Values {
		b.ScratchValues = f(e.Results)
		b.Values = f(e.Results)
	}
	return b, checks
}

func topkExtensionRead(t *testing.T, c TopKConfig, b TopKBuffers, executeErr error) topkAnswer {
	t.Helper()
	if executeErr != nil && !strings.Contains(executeErr.Error(), "device status") {
		t.Fatalf("extension device failure; readback refused: %v", executeErr)
	}
	n := c.Rows * c.K
	a := topkAnswer{indices: make([]int32, n), values: make([]float32, n), counts: make([]int32, c.Rows), status: make([]int32, c.Rows)}
	readInt := func(s cudamem.Slice[int32], v []int32) {
		if err := s.CopyToHost(v); err != nil {
			t.Fatal(err)
		}
	}
	readInt(b.Status, a.status)
	readInt(b.Indices, a.indices)
	readInt(b.Counts, a.counts)
	if c.Values {
		if err := b.Values.CopyToHost(a.values); err != nil {
			t.Fatal(err)
		}
	}
	if executeErr != nil {
		for _, v := range append(a.indices, a.counts...) {
			if v != 0x5a5a5a5a {
				t.Fatal("refused extension published output")
			}
		}
		if c.Values {
			for _, v := range a.values {
				if math.Float32bits(v) != 0x5a5a5a5a {
					t.Fatal("refused extension published values")
				}
			}
		}
		readInt(b.ScratchIndices, a.indices)
		readInt(b.ScratchCounts, a.counts)
		if c.Values {
			if err := b.ScratchValues.CopyToHost(a.values); err != nil {
				t.Fatal(err)
			}
		}
	}
	return a
}

func TestNativeTopKBF16Reference(t *testing.T) {
	dev, target, out := topkExtensionDevice(t)
	a, image := topkReadExtension(t, "bf16", TopKBF16Source, "topk_bf16", target)
	fixtures := topkBF16Fixtures()
	if len(fixtures)+len(topkBF16Mutations()) > 48 {
		t.Fatal("BF16 fixed fixture budget exceeded")
	}
	run := func(name string, f topkBF16Case, a *cudakernel.Artifact, image topkPacketEntry, mutant bool) {
		t.Run(name, func(t *testing.T) {
			owner, err := topkLoadArtifact(dev, f.c, a)
			if owner != nil {
				defer func() {
					if err := owner.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}
			selection := &TopKBF16{owner: owner}
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			b, checks := topkExtensionBuffers(t, dev, f.c)
			input, err := cudamem.Alloc[uint16](dev, len(f.bits)+2)
			if err != nil {
				if input.Ptr() != 0 {
					_ = input.Free()
				}
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := input.Free(); err != nil {
					t.Error(err)
				}
			})
			guard := append([]uint16{0x5a5a}, f.bits...)
			guard = append(guard, 0x5a5a)
			if err := input.CopyFromHost(guard); err != nil {
				t.Fatal(err)
			}
			if err := b.Lengths.CopyFromHost(f.lengths); err != nil {
				t.Fatal(err)
			}
			if err := b.Offsets.CopyFromHost(f.offsets); err != nil {
				t.Fatal(err)
			}
			bf := TopKBF16Buffers{Input: input.Slice(1, len(f.bits)+1), Lengths: b.Lengths, Offsets: b.Offsets, ScratchIndices: b.ScratchIndices, ScratchValues: b.ScratchValues, ScratchCounts: b.ScratchCounts, Status: b.Status, Indices: b.Indices, Values: b.Values, Counts: b.Counts}
			err = selection.Execute(stream, bf)
			got, want := topkExtensionRead(t, f.c, b, err), topkSortReference(f.topkCase)
			for _, check := range checks {
				check()
			}
			if e := input.CopyToHost(guard); e != nil {
				t.Fatal(e)
			}
			if guard[0] != 0x5a5a || guard[len(guard)-1] != 0x5a5a {
				t.Fatal("BF16 input redzone damaged")
			}
			r := topkNativeReceipt{Fixture: f.name, Config: f.c, Image: image, InputBits: topkFloatBits(f.input), Lengths: f.lengths, Offsets: f.offsets, Indices: got.indices, Counts: got.counts, Status: got.status, ValueBits: topkFloatBits(got.values), ExpectedIndices: want.indices, ExpectedCounts: want.counts, ExpectedStatus: want.status, ExpectedValueBits: topkFloatBits(want.values), Published: err == nil, RedzonesIntact: true}
			r.LoadedResources = topkExtensionResources(t, owner)
			if f.c.Values {
				policy := topkValuePolicy(f.topkCase, image)
				policy.MathMode = "bf16-bits-shift16-fp32-ordered-comparisons-copy-no-fast-math"
				r.ValuePolicy = &policy
				if err := cudatest.CompareCUDAOutputFP32(got.values, want.values, policy); err != nil {
					r.ValueComparisonError = err.Error()
				}
			}
			if err != nil {
				r.ExecuteError = err.Error()
			}
			topkWriteJSON(t, filepath.Join(out, name+".json"), struct {
				StorageBits []uint16
				Result      topkNativeReceipt
			}{f.bits, r})
			comparison := topkCompare(f.c, got, want)
			if mutant {
				if comparison == nil {
					t.Fatal("native BF16 mutant survived")
				}
				t.Logf("BF16 mutant rejected: %v", comparison)
				return
			}
			if comparison != nil {
				t.Fatal(comparison)
			}
			if (err == nil) != (CheckTopKStatus(want.status) == nil) {
				t.Fatal("BF16 publication admission mismatch")
			}
		})
	}
	for _, f := range fixtures {
		run("bf16-"+f.name, f, a, image, false)
	}
	if t.Failed() {
		t.Fatal("BF16 baseline failed; mutant dispatch refused")
	}
	for _, m := range topkBF16Mutations() {
		name := "bf16_mutant_" + m.name
		a, image := topkReadExtension(t, name, topkBF16Mutated(m), "topk_bf16", target)
		found := false
		for _, f := range fixtures {
			if f.name == m.fixture {
				run(name, f, a, image, true)
				found = true
				break
			}
		}
		if !found {
			t.Fatal("missing BF16 native mutant fixture")
		}
	}
}

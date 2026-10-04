//go:build native

package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestMHCOfflineExtensionPreparation(t *testing.T) {
	dir := os.Getenv("MHC_EXTENSION_PREP_DIR")
	if dir == "" {
		t.Skip("set MHC_EXTENSION_PREP_DIR for offline extension compilation")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute extension prep directory required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	for _, f := range mhcFixtures() {
		if f.config.Repeat != 10 {
			continue
		}
		for _, variant := range []string{"bf16", "fused", "two-stage"} {
			if variant == "two-stage" && f.config.Width != 33 && f.config.Width != 129 {
				continue
			}
			t.Run(f.name+"-"+variant, func(t *testing.T) {
				var sources []FusedKernel
				if variant == "bf16" {
					s, err := MHCBF16Source(f.config)
					if err != nil {
						t.Fatal(err)
					}
					sources = s[:]
				} else if variant == "fused" {
					s, err := MHCFusedSource(f.config)
					if err != nil {
						t.Fatal(err)
					}
					sources = []FusedKernel{s}
				} else {
					s, err := MHCSource(f.config)
					if err != nil {
						t.Fatal(err)
					}
					sources = s[:]
				}
				mhcPrepareExtensionCase(t, dir, f.name+"-"+variant, f, variant, sources)
			})
		}
	}
	f := mhcBF16CastFixture()
	s, _ := MHCBF16Source(f.config)
	mhcPrepareExtensionCase(t, dir, "bf16-final-rne-exact", f, "bf16", s[:])
	s[1].Source = strings.Replace(s[1].Source, "(v.u+0x7fffu+((v.u>>16)&1u))>>16", "v.u>>16", 1)
	mhcPrepareExtensionCase(t, dir, "bf16-final-cast-mutant", f, "bf16", s[:])
	for _, f := range mhcFixtures() {
		if f.name == "nonsymmetric-w33-r10" {
			s, _ := MHCFusedSource(f.config)
			s.Source = strings.Replace(s.Source, "a[m*4+n]", "a[n*4+m]", 1)
			mhcPrepareExtensionCase(t, dir, "fused-orientation-mutant", f, "fused", []FusedKernel{s})
			break
		}
	}
}

func mhcBF16CastFixture() mhcFixture {
	f := mhcFixtures()[0]
	f.config.Batch = 1
	f.logits = make([]float32, 16)
	f.x = []float32{1}
	f.residual = make([]float32, 4)
	f.post = []float32{math.Float32frombits(0x3f808000), math.Float32frombits(0x3f818000), math.Float32frombits(0xbf818000), math.Float32frombits(0x00008000)}
	return f
}

func mhcPrepareExtensionCase(t *testing.T, dir, name string, f mhcFixture, variant string, sources []FusedKernel) {
	t.Helper()
	var artifacts []*cudakernel.Artifact
	var keys []string
	for i, s := range sources {
		cm, err := nvrtcx.Compile(s.Source, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		schema, err := mhcSchema(append(append([]string(nil), s.Inputs...), s.Outputs...))
		if err != nil {
			t.Fatal(err)
		}
		a, err := cm.ToArtifact(s.Name, schema, "compute_121")
		if err != nil {
			t.Fatal(err)
		}
		artifacts = append(artifacts, a)
		keys = append(keys, cm.Key())
		for suffix, raw := range map[string][]byte{"cu": []byte(s.Source), "ptx": a.Bytes()} {
			if err := os.WriteFile(filepath.Join(dir, name+fmt.Sprintf("-stage%d.%s", i, suffix)), raw, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	q := f
	var xb, rb []uint16
	if variant == "bf16" {
		q, xb, rb = mhcBF16Fixture(f)
	}
	want := mhcOracle(q)
	mhcWriteJSON(t, dir, name+".json", struct {
		Config                                  MHCConfig
		Variant                                 string
		Keys                                    []string
		Artifacts                               []*cudakernel.Artifact
		Logits, X, Post, Residual               []float32
		XBF16Bits, ResidualBF16Bits             []uint16
		ExpectedMixBits, ExpectedFP32OutputBits []uint32
		Conversion                              string
	}{q.config, variant, keys, artifacts, q.logits, q.x, q.post, q.residual, xb, rb, mhcBits(want.mix), mhcBits(want.out), map[string]string{"bf16": "exact input bit expansion; FP32 accumulation; final BF16 RNE; separate half-ULP conversion bound", "fused": "FP32 no conversion", "two-stage": "FP32 no conversion"}[variant]})
}

type mhcBF16NativeInputs struct {
	inputs MHCBF16Inputs
	owners []cudamem.Slice[uint16]
	host   [][]uint16
	fp     mhcNativeInputs
}

func mhcAllocBF16Inputs(t *testing.T, dev *cuda.Device, f mhcFixture) (mhcFixture, mhcBF16NativeInputs) {
	t.Helper()
	q, x, residual := mhcBF16Fixture(f)
	fp := mhcAllocInputs(t, dev, q)
	out := mhcBF16NativeInputs{inputs: MHCBF16Inputs{Logits: fp.inputs.Logits, Post: fp.inputs.Post}, fp: fp}
	for i, xs := range [][]uint16{x, residual} {
		owner, err := cudamem.Alloc[uint16](dev, len(xs)+2)
		if owner.Ptr() != 0 {
			t.Cleanup(func() {
				if err := owner.Free(); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal(err)
		}
		host := make([]uint16, len(xs)+2)
		host[0], host[len(host)-1] = 0x7fc3, 0x7fc3
		copy(host[1:], xs)
		if err := owner.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		view := owner.Slice(1, len(xs)+1)
		if i == 0 {
			out.inputs.X = view
		} else {
			out.inputs.Residual = view
		}
		out.owners = append(out.owners, owner)
		out.host = append(out.host, host)
	}
	return q, out
}

func (in mhcBF16NativeInputs) check(t *testing.T) {
	t.Helper()
	in.fp.check(t)
	for i, b := range in.owners {
		xs := make([]uint16, b.Len())
		if err := b.CopyToHost(xs); err != nil {
			t.Fatal(err)
		}
		for j, x := range xs {
			if x != in.host[i][j] {
				t.Fatalf("BF16 input/guard changed at %d,%d", i, j)
			}
		}
	}
}

func TestNativeMHCExtensions(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native mhc extensions require gpu-lock")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, f := range mhcFixtures() {
		if f.config.Repeat != 10 {
			continue
		}
		for _, variant := range []string{"bf16", "fused"} {
			t.Run(f.name+"-"+variant, func(t *testing.T) {
				var e *Execution
				var q mhcFixture
				var source string
				var verify func()
				if variant == "bf16" {
					var in mhcBF16NativeInputs
					q, in = mhcAllocBF16Inputs(t, dev, f)
					s, _ := MHCBF16Source(q.config)
					source = s[0].Source + s[1].Source
					e, err = ExecuteMHCBF16(context.Background(), stream, q.config, in.inputs)
					verify = func() { in.check(t) }
				} else {
					q = f
					in := mhcAllocInputs(t, dev, q)
					s, _ := MHCFusedSource(q.config)
					source = s.Source
					e, err = ExecuteMHCFused(context.Background(), stream, q.config, in.inputs)
					verify = func() { in.check(t) }
				}
				if e != nil {
					t.Cleanup(func() {
						if err := e.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				out, err := e.Readback(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				want := mhcOracle(q)
				receipts := e.Receipts()
				if variant == "bf16" {
					mix := make([]float32, len(q.logits))
					if err := e.resources[0].read(mix); err != nil {
						t.Fatal(err)
					}
					if err := mhcCompareBF16(q, mhcAnswer{mix: mix, out: out}, source, receipts[0].Target); err != nil {
						t.Fatal(err)
					}
					if e.Stats().Launches != 2 || e.Stats().Allocations != 2 {
						t.Fatal(e.Stats())
					}
				} else {
					if err := cudatest.CompareCUDAOutputFP32(out, want.out, mhcPolicy(q, source, receipts[0].Target, len(out))); err != nil {
						t.Fatal(err)
					}
					if e.Stats().Launches != 1 || e.Stats().Allocations != 1 {
						t.Fatal(e.Stats())
					}
				}
				verify()
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				mhcWriteJSON(t, os.Getenv("MHC_EXTENSION_PACKET_DIR"), f.name+"-"+variant+".json", struct {
					Config                       MHCConfig
					Variant                      string
					Receipts                     []KernelReceipt
					Stats                        ExecutionStats
					ActualBits, ExpectedFP32Bits []uint32
					ConversionRelativeAllowance  float64
				}{q.config, variant, receipts, e.Stats(), mhcBits(out), mhcBits(want.out), map[string]float64{"bf16": 1.0 / 256, "fused": 0}[variant]})
			})
		}
	}
	t.Run("fused-orientation-mutant", func(t *testing.T) {
		var f mhcFixture
		for _, candidate := range mhcFixtures() {
			if candidate.name == "nonsymmetric-w33-r10" {
				f = candidate
				break
			}
		}
		if f.name == "" {
			t.Fatal("missing orientation fixture")
		}
		in := mhcAllocInputs(t, dev, f)
		s, err := MHCFusedSource(f.config)
		if err != nil {
			t.Fatal(err)
		}
		before := s.Source
		s.Source = strings.Replace(s.Source, "a[m*4+n]", "a[n*4+m]", 1)
		if before == s.Source {
			t.Fatal("orientation mutant did not match")
		}
		ptrs, releases, err := mhcBorrowFP32(dev, f.config, in.inputs)
		if err != nil {
			t.Fatal(err)
		}
		ops := nativeExecutionOps(dev, stream)
		compile := ops.compile
		ops.compile = func(FusedKernel) (*executionKernel, error) { return compile(s) }
		e, err := mhcFusedExecute(context.Background(), f.config, ptrs, ops, releases)
		if e != nil {
			t.Cleanup(func() {
				if err := e.Close(); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal("orientation mutant failed before comparison", err)
		}
		out, err := e.Readback(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		want := mhcOracle(f)
		comparison := cudatest.CompareCUDAOutputFP32(out, want.out, mhcPolicy(f, s.Source, e.Receipts()[0].Target, len(out)))
		var mismatch *cudatest.CUDAOutputMismatch
		if !errors.As(comparison, &mismatch) {
			t.Fatal("orientation mutant not semantically rejected", comparison)
		}
		in.check(t)
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		mhcWriteJSON(t, os.Getenv("MHC_EXTENSION_PACKET_DIR"), "fused-orientation-mutant.json", struct {
			ActualBits, ExpectedBits []uint32
			Receipts                 []KernelReceipt
			Comparison               string
			Rejected                 bool
		}{mhcBits(out), mhcBits(want.out), e.Receipts(), comparison.Error(), true})
	})
	t.Run("bf16-admission-controls", func(t *testing.T) {
		f := mhcFixtures()[0]
		q, in := mhcAllocBF16Inputs(t, dev, f)
		alias := in.inputs
		view, err := cudamem.Reinterpret[uint16](alias.Logits)
		if err != nil {
			t.Fatal(err)
		}
		alias.X = view.Slice(0, q.config.Batch*q.config.Width)
		e, aliasErr := ExecuteMHCBF16(context.Background(), stream, q.config, alias)
		if e != nil {
			_ = e.Close()
		}
		if e != nil || aliasErr == nil || !strings.Contains(aliasErr.Error(), "aliased") {
			t.Fatal("BF16 alias not correctly refused", aliasErr)
		}
		xs := append([]uint16(nil), in.host[0]...)
		xs[1] = 0x7fc0
		if err := in.owners[0].CopyFromHost(xs); err != nil {
			t.Fatal(err)
		}
		e, nonfiniteErr := ExecuteMHCBF16(context.Background(), stream, q.config, in.inputs)
		if e != nil {
			_ = e.Close()
		}
		if e != nil || nonfiniteErr == nil || !strings.Contains(nonfiniteErr.Error(), "nonfinite") {
			t.Fatal("BF16 NaN not correctly refused", nonfiniteErr)
		}
		mhcWriteJSON(t, os.Getenv("MHC_EXTENSION_PACKET_DIR"), "bf16-admission-controls.json", struct {
			AliasError, NonfiniteError string
			AdmissionOnly              bool
		}{aliasErr.Error(), nonfiniteErr.Error(), true})
	})
	t.Run("bf16-final-rne-exact", func(t *testing.T) {
		f := mhcBF16CastFixture()
		q, in := mhcAllocBF16Inputs(t, dev, f)
		e, err := ExecuteMHCBF16(context.Background(), stream, q.config, in.inputs)
		if e != nil {
			t.Cleanup(func() {
				if err := e.Close(); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal(err)
		}
		out, err := e.Readback(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			want := mhcBF16Reference(f.post[i])
			if math.Float32bits(v) != uint32(want)<<16 {
				t.Fatalf("device final RNE not exact at%d got=%08x want=%04x", i, math.Float32bits(v), want)
			}
		}
		in.check(t)
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		mhcWriteJSON(t, os.Getenv("MHC_EXTENSION_PACKET_DIR"), "bf16-final-rne-exact.json", struct {
			ActualBits []uint32
			Post       []float32
			Receipts   []KernelReceipt
		}{mhcBits(out), f.post, e.Receipts()})
		sources, _ := MHCBF16Source(q.config)
		before := sources[1].Source
		sources[1].Source = strings.Replace(before, "(v.u+0x7fffu+((v.u>>16)&1u))>>16", "v.u>>16", 1)
		if sources[1].Source == before {
			t.Fatal("BF16 cast mutant did not match")
		}
		var ptrs []driver.DevicePtr
		var releases []func()
		for _, b := range []cudamem.Slice[float32]{in.inputs.Logits, in.inputs.Post} {
			r, err := b.Borrow(dev)
			if err != nil {
				for _, release := range releases {
					release()
				}
				t.Fatal(err)
			}
			releases = append(releases, r)
		}
		for _, b := range []cudamem.Slice[uint16]{in.inputs.X, in.inputs.Residual} {
			r, err := b.Borrow(dev)
			if err != nil {
				for _, release := range releases {
					release()
				}
				t.Fatal(err)
			}
			releases = append(releases, r)
		}
		ptrs = []driver.DevicePtr{in.inputs.Logits.Ptr(), in.inputs.X.Ptr(), in.inputs.Post.Ptr(), in.inputs.Residual.Ptr()}
		mutant, err := mhcExecute(context.Background(), q.config, ptrs, mhcBF16Ops(dev, stream, sources), releases)
		if mutant != nil {
			t.Cleanup(func() {
				if err := mutant.Close(); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal("cast mutant failed before output", err)
		}
		got, err := mutant.Readback(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 4 || math.Float32bits(got[1]) != 0x3f810000 || math.Float32bits(out[1]) != 0x3f820000 {
			t.Fatal("actual BF16 truncation mutant not independently rejected", got, out)
		}
		in.check(t)
		if err := mutant.Close(); err != nil {
			t.Fatal(err)
		}
		mhcWriteJSON(t, os.Getenv("MHC_EXTENSION_PACKET_DIR"), "bf16-final-cast-mutant.json", struct {
			ActualBits, HealthyBits []uint32
			Receipts                []KernelReceipt
			Rejected                bool
		}{mhcBits(got), mhcBits(out), mutant.Receipts(), true})
	})
}

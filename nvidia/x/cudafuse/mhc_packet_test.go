//go:build native

package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type mhcPacketCase struct {
	fixture  mhcFixture
	mutation *mhcMutation
}

func mhcBits(xs []float32) []uint32 {
	out := make([]uint32, len(xs))
	for i, x := range xs {
		out[i] = math.Float32bits(x)
	}
	return out
}

// 17 healthy two-stage cases plus five source controls: 22 cases and 44
// launches. Alias and nonfinite admission controls add no kernel launch.
func mhcPacketCases() []mhcPacketCase {
	fixtures := mhcFixtures()
	var out []mhcPacketCase
	for _, f := range fixtures {
		out = append(out, mhcPacketCase{fixture: f})
	}
	for i := range mhcMutations {
		m := &mhcMutations[i]
		for _, f := range fixtures {
			if f.name == m.fixture {
				out = append(out, mhcPacketCase{f, m})
				break
			}
		}
	}
	return out
}

func mhcPacketName(c mhcPacketCase) string {
	if c.mutation != nil {
		return "mutant-" + c.mutation.name
	}
	return c.fixture.name
}

func mhcPacketSources(c mhcPacketCase) ([2]FusedKernel, error) {
	if c.mutation != nil {
		return mhcMutated(c.fixture.config, *c.mutation)
	}
	return MHCSource(c.fixture.config)
}

func mhcWriteJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	if dir == "" {
		return
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("mhc packet directory must be absolute")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	err = json.NewEncoder(f).Encode(v)
	closeErr := f.Close()
	if err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
}

// This explicitly named preparation test invokes only NVRTC. It must never
// call CUDA initialization, allocate device storage, or execute a kernel.
func TestMHCOfflinePacketPreparation(t *testing.T) {
	dir := os.Getenv("MHC_PREP_DIR")
	if dir == "" {
		t.Skip("set MHC_PREP_DIR for offline packet compilation")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("mhc prep directory must be absolute")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual NVRTC %d.%d path=%s; offline target compute_121; no GPU calls", major, minor, nvrtc.Path())
	for _, c := range mhcPacketCases() {
		t.Run(mhcPacketName(c), func(t *testing.T) {
			sources, err := mhcPacketSources(c)
			if err != nil {
				t.Fatal(err)
			}
			var artifacts [2]*cudakernel.Artifact
			var keys [2]string
			var publicArtifacts [2]*cudakernel.Artifact
			if c.mutation == nil {
				publicArtifacts, err = CompileMHC(c.fixture.config, 12, 1)
				if err != nil {
					t.Fatal(err)
				}
			}
			for i, s := range sources {
				cm, err := nvrtcx.Compile(s.Source, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache())
				if err != nil {
					t.Fatal(err)
				}
				schema, err := mhcSchema(append(append([]string(nil), s.Inputs...), s.Outputs...))
				if err != nil {
					t.Fatal(err)
				}
				artifacts[i], err = cm.ToArtifact(s.Name, schema, "compute_121")
				if err != nil {
					t.Fatal(err)
				}
				keys[i] = cm.Key()
				if artifacts[i].Kind != cudakernel.KindPTX {
					t.Fatal("expected PTX")
				}
				if c.mutation == nil && (publicArtifacts[i].Digest != artifacts[i].Digest || publicArtifacts[i].EntryPoint != artifacts[i].EntryPoint) {
					t.Fatal("public artifact differs from fresh producer")
				}
				if err := os.WriteFile(filepath.Join(dir, mhcPacketName(c)+fmt.Sprintf("-stage%d.cu", i)), []byte(s.Source), 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, mhcPacketName(c)+fmt.Sprintf("-stage%d.ptx", i)), artifacts[i].Bytes(), 0600); err != nil {
					t.Fatal(err)
				}
			}
			want := mhcOracle(c.fixture)
			mhcWriteJSON(t, dir, mhcPacketName(c)+".json", struct {
				Config                                                 MHCConfig
				CompilerVersion, CompilerPath                          string
				Keys                                                   [2]string
				Artifacts                                              [2]*cudakernel.Artifact
				Logits, X, Post, Residual, ExpectedMix, ExpectedOutput []float32
				RowResidual, ColumnResidual                            float64
			}{c.fixture.config, fmt.Sprintf("%d.%d", major, minor), nvrtc.Path(), keys, artifacts, c.fixture.logits, c.fixture.x, c.fixture.post, c.fixture.residual, want.mix, want.out, want.rowResidual, want.columnResidual})
		})
	}
}

type mhcNativeInputs struct {
	inputs MHCInputs
	owners []cudamem.Slice[float32]
	host   [][]float32
}

func mhcAllocInputs(t *testing.T, dev *cuda.Device, f mhcFixture) mhcNativeInputs {
	t.Helper()
	var out mhcNativeInputs
	for _, xs := range [][]float32{f.logits, f.x, f.post, f.residual} {
		owner, err := cudamem.Alloc[float32](dev, len(xs)+2)
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
		host := make([]float32, len(xs)+2)
		host[0] = math.Float32frombits(0x7fc04567)
		host[len(host)-1] = host[0]
		copy(host[1:], xs)
		if err := owner.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		out.owners = append(out.owners, owner)
		out.host = append(out.host, host)
	}
	out.inputs = MHCInputs{out.owners[0].Slice(1, len(f.logits)+1), out.owners[1].Slice(1, len(f.x)+1), out.owners[2].Slice(1, len(f.post)+1), out.owners[3].Slice(1, len(f.residual)+1)}
	return out
}

func (in mhcNativeInputs) check(t *testing.T) {
	t.Helper()
	for i, b := range in.owners {
		got := make([]float32, b.Len())
		if err := b.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		for j, x := range got {
			if math.Float32bits(x) != math.Float32bits(in.host[i][j]) {
				t.Fatalf("input/guard changed at %d,%d", i, j)
			}
		}
	}
}

func TestNativeMHCTwoStageReference(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native mhc requires gpu-lock")
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
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	target := fmt.Sprintf("compute_%d%d", major, minor)
	for _, c := range mhcPacketCases() {
		t.Run(mhcPacketName(c), func(t *testing.T) {
			f := c.fixture
			in := mhcAllocInputs(t, dev, f)
			sources, err := mhcPacketSources(c)
			if err != nil {
				t.Fatal(err)
			}
			var e *Execution
			var tailOutput []float32
			if c.mutation == nil {
				e, err = ExecuteMHC(context.Background(), stream, f.config, in.inputs)
			} else {
				ops := nativeExecutionOps(dev, stream)
				compile := ops.compile
				if c.mutation.name == "tail-dropped" {
					ops.alloc = func(n int) (*executionBuffer, error) {
						storage, err := cudamem.Alloc[float32](dev, n+2)
						if storage.Ptr() == 0 {
							return nil, err
						}
						b := &executionBuffer{ptr: storage.Slice(1, n+1).Ptr(), free: storage.Free, freed: func() bool { return storage.Ptr() == 0 }, read: func(out []float32) error {
							all := make([]float32, n+2)
							if err := storage.CopyToHost(all); err != nil {
								return err
							}
							if n == len(f.residual) {
								tailOutput = append([]float32(nil), all[1:n+1]...)
							}
							if err := checkExecutionOutput(all, n); err != nil {
								return err
							}
							copy(out, all[1:n+1])
							return nil
						}}
						if err != nil {
							return b, err
						}
						poison := make([]float32, n+2)
						for i := range poison {
							poison[i] = math.Float32frombits(0x7fc01234)
						}
						return b, storage.CopyFromHost(poison)
					}
				}
				ops.compile = func(s FusedKernel) (*executionKernel, error) {
					for _, replacement := range sources {
						if replacement.Name == s.Name {
							return compile(replacement)
						}
					}
					return nil, fmt.Errorf("unexpected mhc entry %s", s.Name)
				}
				var ptrs []driver.DevicePtr
				var releases []func()
				for _, b := range []cudamem.Slice[float32]{in.inputs.Logits, in.inputs.X, in.inputs.Post, in.inputs.Residual} {
					release, err := b.Borrow(dev)
					if err != nil {
						for _, r := range releases {
							r()
						}
						t.Fatal(err)
					}
					releases = append(releases, release)
					ptrs = append(ptrs, b.Ptr())
				}
				e, err = mhcExecute(context.Background(), f.config, ptrs, ops, releases)
			}
			if e != nil {
				t.Cleanup(func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal("failed before numerical comparison", err)
			}
			out, err := e.Readback(context.Background())
			if err != nil {
				if c.mutation == nil || c.mutation.name != "tail-dropped" {
					t.Fatal(err)
				}
				if !errors.Is(err, ErrOutputUnwritten) {
					t.Fatal("unexpected tail failure", err)
				}
				out = tailOutput
			}
			mix := make([]float32, len(f.logits))
			if err := e.resources[0].read(mix); err != nil {
				t.Fatal(err)
			}
			comparison := mhcCompare(f, mhcAnswer{mix: mix, out: out}, sources[0].Source+sources[1].Source, target)
			if c.mutation == nil {
				if comparison != nil {
					t.Fatal(comparison)
				}
			} else if comparison == nil {
				t.Fatal("source mutant survived")
			}
			in.check(t)
			receipts := e.Receipts()
			if len(receipts) != 2 || e.Stats().Launches != 2 || e.Stats().Allocations != 2 {
				t.Fatal("wrong actual two-stage calls", e.Stats())
			}
			for i, r := range receipts {
				if r.SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte(sources[i].Source))) || len(r.PTXDigest) != 64 || len(r.CompilerKey) != 64 {
					t.Fatal("missing source/image/compiler binding", r)
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if e.Stats().Frees != 2 {
				t.Fatal("scratch/output not freed")
			}
			want := mhcOracle(f)
			mhcWriteJSON(t, os.Getenv("MHC_PACKET_DIR"), mhcPacketName(c)+".json", struct {
				Config                                                               MHCConfig
				Receipts                                                             []KernelReceipt
				Stats                                                                ExecutionStats
				ActualMixBits, ActualOutputBits, ExpectedMixBits, ExpectedOutputBits []uint32
				RowResidual, ColumnResidual                                          float64
				Rejected                                                             bool
				Comparison                                                           string
			}{f.config, receipts, e.Stats(), mhcBits(mix), mhcBits(out), mhcBits(want.mix), mhcBits(want.out), want.rowResidual, want.columnResidual, c.mutation != nil, fmt.Sprint(comparison)})
			t.Logf("actual launches=%d allocations=%d frees=%d; row residual=%g column residual=%g; rejected=%v", e.Stats().Launches, e.Stats().Allocations, e.Stats().Frees, want.rowResidual, want.columnResidual, c.mutation != nil)
		})
	}
	t.Run("alias-and-nonfinite-refusal", func(t *testing.T) {
		f := mhcFixtures()[0]
		in := mhcAllocInputs(t, dev, f)
		alias := in.inputs
		alias.Post = alias.Logits.Slice(0, f.config.Batch*4)
		if e, err := ExecuteMHC(context.Background(), stream, f.config, alias); err == nil || e != nil {
			if e != nil {
				_ = e.Close()
			}
			t.Fatal("forbidden alias admitted", err)
		}
		host := append([]float32(nil), in.host[0]...)
		host[1] = float32(math.NaN())
		if err := in.owners[0].CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		if e, err := ExecuteMHC(context.Background(), stream, f.config, in.inputs); err == nil || e != nil {
			if e != nil {
				_ = e.Close()
			}
			t.Fatal("nonfinite input admitted", err)
		}
	})
}

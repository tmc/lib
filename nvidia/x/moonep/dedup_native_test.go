//go:build native && !windows

package moonep

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type dedupPacketCase struct {
	name   string
	spec   DedupSpec
	mutant *dedupMutant
}

func dedupPacketCases() []dedupPacketCase {
	var out []dedupPacketCase
	for _, width := range []int{1, 17, 128} {
		for _, kind := range []string{"groups", "zero", "permuted"} {
			s := dedupFixture(width)
			if kind == "zero" {
				s.Counts = [2]int32{}
				s.Groups = nil
				s.DupLoffs = nil
			}
			if kind == "permuted" {
				s.Groups[0], s.Groups[1] = s.Groups[1], s.Groups[0]
			}
			out = append(out, dedupPacketCase{fmt.Sprintf("w%d-%s", width, kind), s, nil})
		}
	}
	for _, m := range dedupMutants() {
		m := m
		out = append(out, dedupPacketCase{"mutant-" + m.name, dedupFixture(17), &m})
	}
	return out
}
func dedupJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	if dir == "" {
		t.Fatal("absolute dedup receipt directory required")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute receipt directory required")
	}
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(dir, name+".json"), append(b, '\n'), 0600); e != nil {
		t.Fatal(e)
	}
}

func TestDedupOfflinePreparation(t *testing.T) {
	dir := os.Getenv("DEDUP_PREP_DIR")
	if dir == "" {
		t.Skip("set DEDUP_PREP_DIR for compiler-only preparation")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute prep directory required")
	}
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	for _, c := range dedupPacketCases() {
		t.Run(c.name, func(t *testing.T) {
			k := mustDedupSources(t, c.spec)
			if c.mutant != nil {
				k = dedupMutate(t, k, *c.mutant)
			}
			x, w := dedupFixtureInput(c.spec, 0)
			var images []*cudakernel.Artifact
			var keys []string
			for i, s := range k {
				cm, e := nvrtcx.Compile(s.Source, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache())
				if e != nil {
					t.Fatal(e)
				}
				a, e := cm.ToArtifact(s.Name, s.Schema, "compute_121")
				if e != nil {
					t.Fatal(e)
				}
				images = append(images, a)
				keys = append(keys, cm.Key())
				for suffix, raw := range map[string][]byte{"cu": []byte(s.Source), "ptx": []byte(cm.PTX())} {
					if e = os.WriteFile(filepath.Join(dir, fmt.Sprintf("%s-stage%d.%s", c.name, i, suffix)), raw, 0600); e != nil {
						t.Fatal(e)
					}
				}
			}
			dedupJSON(t, dir, c.name, struct {
				Spec           DedupSpec
				Input, Weights []uint32
				Expected       DedupResult
				Artifacts      []*cudakernel.Artifact
				Keys           []string
			}{c.spec, dedupBits(x), dedupBits(w), dedupReference(c.spec, 0, w), images, keys})
		})
	}
}

func dedupPopulatePlan(t *testing.T, dev *cuda.Device, s DedupSpec) *CommPlan {
	t.Helper()
	p, e := devPlan(dev, s)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := dev.WithContext(p.Close); e != nil {
			t.Error(e)
		}
	})
	m, e := dedupAdmit(s)
	if e != nil {
		t.Fatal(e)
	}
	write := func(ptr driver.DevicePtr, xs []int32) {
		if len(xs) == 0 {
			return
		}
		if e := dev.WithContext(func() error { return ptr.CopyFromHostPtr(unsafe.Pointer(&xs[0]), uint(len(xs)*4)) }); e != nil {
			t.Fatal(e)
		}
	}
	write(p.Dst, m.dst)
	groups := make([]int32, 0, len(s.Groups)*3)
	for _, g := range s.Groups {
		groups = append(groups, g[:]...)
	}
	write(p.DupGroups, groups)
	write(p.DupLoffs, s.DupLoffs)
	write(p.DupCounts, s.Counts[:])
	write(p.CuSeqlens, make([]int32, p.E+p.B))
	write(p.ExpertsToCopy, make([]int32, p.R*p.B))
	write(p.RemoteStats, []int32{0, 0})
	padding := make([]int32, (p.E+p.B)*2)
	for i, r := range s.Padding {
		copy(padding[i*2:], r[:])
	}
	write(p.ZeroFillRanges, padding)
	return p
}
func devPlan(dev *cuda.Device, s DedupSpec) (*CommPlan, error) {
	var p *CommPlan
	var err error
	e := dev.WithContext(func() error {
		p, err = AllocatePlan(&Layout{Config: Config{R: 1, E: 4, B: 4, K: s.K}, N: s.Tokens * s.K, NvS: s.Slots})
		return err
	})
	return p, e
}

type dedupNativeInput struct {
	payload, weight cudamem.Slice[float32]
	owners          []cudamem.Slice[float32]
	host            [][]float32
}

func dedupAllocInput(t *testing.T, dev *cuda.Device, s DedupSpec, epoch int) dedupNativeInput {
	t.Helper()
	x, w := dedupFixtureInput(s, epoch)
	var out dedupNativeInput
	for i, xs := range [][]float32{x, w} {
		owner, e := cudamem.Alloc[float32](dev, len(xs)+2)
		if owner.Ptr() != 0 {
			t.Cleanup(func() {
				if e := owner.Free(); e != nil {
					t.Error(e)
				}
			})
		}
		if e != nil {
			t.Fatal(e)
		}
		host := make([]float32, len(xs)+2)
		host[0], host[len(host)-1] = math.Float32frombits(0x7fc07903), math.Float32frombits(0x7fc07903)
		copy(host[1:], xs)
		if e = owner.CopyFromHost(host); e != nil {
			t.Fatal(e)
		}
		view := owner.Slice(1, len(xs)+1)
		if i == 0 {
			out.payload = view
		} else {
			out.weight = view
		}
		out.owners = append(out.owners, owner)
		out.host = append(out.host, host)
	}
	return out
}
func (in *dedupNativeInput) check(t *testing.T) {
	t.Helper()
	for i, b := range in.owners {
		xs := make([]float32, b.Len())
		if e := b.CopyToHost(xs); e != nil {
			t.Fatal(e)
		}
		for j, v := range xs {
			if math.Float32bits(v) != math.Float32bits(in.host[i][j]) {
				t.Fatal("input or guard changed", i, j)
			}
		}
	}
}
func (in *dedupNativeInput) update(t *testing.T, s DedupSpec, epoch int) {
	t.Helper()
	x, w := dedupFixtureInput(s, epoch)
	for i, xs := range [][]float32{x, w} {
		copy(in.host[i][1:], xs)
		if e := in.owners[i].CopyFromHost(in.host[i]); e != nil {
			t.Fatal(e)
		}
	}
}

func TestNativeDedupExpertConsumer(t *testing.T) {
	if os.Getenv("DEDUP_NATIVE") != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("explicit DEDUP_NATIVE=1 and gpu-lock required")
	}
	if _, e := cudalock.Inherited(os.Getenv); e != nil {
		t.Fatal(e)
	}
	dev, e := cuda.NewDevice(0)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := dev.Close(); e != nil {
			t.Error(e)
		}
	})
	stream, e := dev.NewStream()
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := stream.Close(); e != nil {
			t.Error(e)
		}
	})
	for _, c := range dedupPacketCases() {
		t.Run(c.name, func(t *testing.T) {
			s := c.spec
			p := dedupPopulatePlan(t, dev, s)
			d, e := NewDedup(dev, p, &s)
			if e != nil {
				t.Fatal(e)
			}
			in := dedupAllocInput(t, dev, s, 0)
			epochs := 1
			if c.mutant == nil {
				epochs = 2
			}
			for epoch := 0; epoch < epochs; epoch++ {
				if epoch > 0 {
					in.update(t, s, epoch)
				}
				var invocation *DedupInvocation
				if c.mutant == nil {
					invocation, e = d.Execute(context.Background(), stream, in.payload, in.weight, s.BufferGeneration)
				} else {
					release, e := dedupAcquire(p, d.snapshot)
					if e != nil {
						t.Fatal(e)
					}
					releases := []func(){release}
					for _, b := range []cudamem.Slice[float32]{in.payload, in.weight} {
						r, e := b.Borrow(dev)
						if e != nil {
							for _, r := range releases {
								r()
							}
							t.Fatal(e)
						}
						releases = append(releases, r)
					}
					ops := dedupNativeOps(dev, stream)
					compile := ops.compile
					k := dedupMutate(t, mustDedupSources(t, s), *c.mutant)
					ops.compile = func(src DedupKernel) (*dedupLoaded, error) {
						for _, replacement := range k {
							if replacement.Name == src.Name {
								return compile(replacement)
							}
						}
						return nil, errors.New("unexpected entry")
					}
					// Preserve actual mutant NaN bits for lane69's reject policy. Guards
					// remain checked by the ordinary buffer reader before this override.
					alloc := ops.alloc
					ops.alloc = func(n int) (*dedupBuffer, error) {
						b, e := alloc(n)
						if b == nil {
							return nil, e
						}
						read := b.read
						b.read = func() ([]float32, error) {
							xs, e := read()
							if e == nil || !strings.Contains(e.Error(), "unwritten or nonfinite") {
								return xs, e
							}
							xs = make([]float32, n)
							e = dev.WithContext(func() error { return b.ptr.CopyToHostPtr(unsafe.Pointer(&xs[0]), uint(n*4)) })
							return xs, e
						}
						return b, e
					}
					invocation, e = dedupRun(context.Background(), s, in.payload.Ptr(), in.weight.Ptr(), ops, releases)
				}
				if invocation != nil {
					t.Cleanup(func() {
						if e := invocation.Close(); e != nil {
							t.Error(e)
						}
					})
				}
				if e != nil {
					t.Fatal(e)
				}
				got, e := invocation.Readback(nil)
				if e != nil {
					t.Fatal(e)
				}
				_, w := dedupFixtureInput(s, epoch)
				want := dedupReference(s, epoch, w)
				receipts := invocation.Receipts()
				source := ""
				for _, r := range receipts {
					source += r.Source
				}
				comparison := dedupCompare(got, want, source, receipts[0].Target, dedupInputDigest(s, epoch))
				rejected := c.mutant != nil
				if rejected {
					var mismatch *cudatest.CUDAOutputMismatch
					if !errors.As(comparison, &mismatch) {
						t.Fatal("mutant did not fail numerical oracle", comparison)
					}
				} else if comparison != nil {
					t.Fatal(comparison)
				}
				in.check(t)
				if e = invocation.Close(); e != nil {
					t.Fatal(e)
				}
				dedupJSON(t, os.Getenv("DEDUP_PACKET_DIR"), fmt.Sprintf("%s-epoch%d", c.name, epoch), dedupMakeReceipt(s, epoch, receipts, got, want, rejected, fmt.Sprint(comparison)))
			}
			if c.mutant == nil {
				if v, e := d.Execute(nil, stream, in.payload.Slice(0, in.payload.Len()-1), in.weight, s.BufferGeneration); v != nil || e == nil {
					t.Fatal("wrong buffer shape admitted")
				}
				badCounts := [2]int32{int32(s.Slots + 1), s.Counts[1]}
				writeCounts := func(xs *[2]int32) {
					if e := dev.WithContext(func() error { return p.DupCounts.CopyFromHostPtr(unsafe.Pointer(&xs[0]), 8) }); e != nil {
						t.Fatal(e)
					}
				}
				writeCounts(&badCounts)
				if v, e := d.Execute(nil, stream, in.payload, in.weight, s.BufferGeneration); v != nil || e == nil {
					t.Fatal("mutated device compact count admitted")
				}
				writeCounts(&s.Counts)
				if v, e := d.Execute(nil, stream, in.payload, in.weight, s.BufferGeneration+1); v != nil || e == nil {
					t.Fatal("wrong generation admitted")
				}
				s.SourceID = "changed"
				if v, e := d.Execute(nil, stream, in.payload, in.weight, 1); v != nil || e == nil {
					t.Fatal("changed source admitted")
				}
				dedupJSON(t, os.Getenv("DEDUP_PACKET_DIR"), c.name+"-admission", struct{ WrongShape, WrongGeneration, ChangedSource, DeviceCountOverCapacity, NoLaunch bool }{true, true, true, true, true})
			}
		})
	}
	t.Run("incomplete-lifetime", func(t *testing.T) {
		s := dedupFixture(17)
		p := dedupPopulatePlan(t, dev, s)
		d, e := NewDedup(dev, p, &s)
		if e != nil {
			t.Fatal(e)
		}
		in := dedupAllocInput(t, dev, s, 0)
		release, e := dedupAcquire(p, d.snapshot)
		if e != nil {
			t.Fatal(e)
		}
		releases := []func(){release}
		for _, b := range []cudamem.Slice[float32]{in.payload, in.weight} {
			r, e := b.Borrow(dev)
			if e != nil {
				t.Fatal(e)
			}
			releases = append(releases, r)
		}
		ops := dedupNativeOps(dev, stream)
		wait := ops.wait
		blocked := true
		ops.wait = func(ctx context.Context) error {
			if blocked {
				return context.Canceled
			}
			return wait(ctx)
		}
		v, e := dedupRun(context.Background(), s, in.payload.Ptr(), in.weight.Ptr(), ops, releases)
		if e == nil {
			t.Fatal("forced incomplete wait succeeded")
		}
		t.Cleanup(func() {
			blocked = false
			if e := v.Close(); e != nil {
				t.Error(e)
			}
		})
		if p.Close() == nil || in.owners[0].Free() == nil {
			t.Fatal("incomplete work freed")
		}
		if r, e := dedupAcquire(p, d.snapshot); e == nil {
			r()
			t.Fatal("incomplete plan reused")
		}
		if v.Close() == nil {
			t.Fatal("failed drain erased owners")
		}
		blocked = false
		if e = v.Close(); e != nil {
			t.Fatal(e)
		}
		if e := p.Close(); e != nil {
			t.Fatal(e)
		}
		dedupJSON(t, os.Getenv("DEDUP_PACKET_DIR"), "incomplete-lifetime", struct {
			BlockedClose, BlockedFree, BlockedReuse, FailedDrainRetained, Recovery bool
			Receipts                                                               []DedupReceipt
		}{true, true, true, true, true, v.Receipts()})
	})
}

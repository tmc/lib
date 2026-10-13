package main

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type parityDevice struct {
	t   *testing.T
	dev *cuda.Device
	gm  *GPUModules
}

func newParityDevice(t *testing.T) *parityDevice {
	dev := cudatest.RequireDevice(t)
	gm, err := CompileGPUModules(dev)
	if gm != nil {
		t.Cleanup(func() {
			if err := gm.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	return &parityDevice{t: t, dev: dev, gm: gm}
}

func (d *parityDevice) upload(host []float32) cudamem.Slice[float32] {
	d.t.Helper()
	// A nonzero slice returned with an error is ours; a zero one frees as
	// a no-op.
	s, err := cudamem.Alloc[float32](d.dev, len(host))
	d.t.Cleanup(func() {
		if err := s.Free(); err != nil {
			d.t.Error(err)
		}
	})
	if err != nil {
		d.t.Fatal(err)
	}
	if err := s.CopyFromHost(host); err != nil {
		d.t.Fatal(err)
	}
	return s
}

func (d *parityDevice) download(s cudamem.Slice[float32]) []float32 {
	d.t.Helper()
	host := make([]float32, s.Len())
	if err := s.CopyToHost(host); err != nil {
		d.t.Fatal(err)
	}
	return host
}

func TestFusedRMSNormQKVParity(t *testing.T) {
	d := newParityDevice(t)
	for _, tt := range qkvParityCases {
		for _, bias := range qkvBiasModes {
			t.Run(tt.name+"/bias "+bias, func(t *testing.T) {
				f := newQKVFixture(tt.in, tt.qDim, tt.kvDim, bias)
				dx, dg := d.upload(f.x), d.upload(f.g)
				dw := []cudamem.Slice[float32]{d.upload(f.wq), d.upload(f.wk), d.upload(f.wv)}
				ptr := func(b []float32) any {
					if b == nil {
						return nullDevicePtr
					}
					return d.upload(b)
				}
				dims := []int{tt.qDim, tt.kvDim, tt.kvDim}

				// Fused kernel.
				fused := []cudamem.Slice[float32]{d.upload(make([]float32, tt.qDim)), d.upload(make([]float32, tt.kvDim)), d.upload(make([]float32, tt.kvDim))}
				if err := cudalaunch.Kernel(d.gm.fusedQKVFn).Grid(uint32(tt.qDim+2*tt.kvDim), 1, 1).Block(128, 1, 1).
					Launch(dx, dg, dw[0], dw[1], dw[2], ptr(f.bq), ptr(f.bk), ptr(f.bv), fused[0], fused[1], fused[2],
						int32(tt.in), int32(tt.qDim), int32(tt.kvDim), float32(qkvParityEps)); err != nil {
					t.Fatal(err)
				}
				// Unfused path: RMSNorm then one LinearGEMV per projection.
				dn := d.upload(make([]float32, tt.in))
				if err := cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(dx, dg, dn, int32(tt.in), float32(qkvParityEps)); err != nil {
					t.Fatal(err)
				}
				unfused := make([]cudamem.Slice[float32], 3)
				for i, b := range [][]float32{f.bq, f.bk, f.bv} {
					unfused[i] = d.upload(make([]float32, dims[i]))
					if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(dims[i]), 1, 1).Block(128, 1, 1).
						Launch(dn, dw[i], ptr(b), unfused[i], int32(tt.in), int32(dims[i])); err != nil {
						t.Fatal(err)
					}
				}

				names := []string{"q", "k", "v"}
				got := make([][]float32, 3)
				for i, want := range f.want {
					got[i] = d.download(fused[i])
					if j := parityMismatch(got[i], want, kernelTolerance); j >= 0 {
						t.Errorf("fused %s[%d]: got %g want %g", names[i], j, got[i][j], want[j])
					}
					u := d.download(unfused[i])
					if j := parityMismatch(u, want, kernelTolerance); j >= 0 {
						t.Errorf("unfused %s[%d]: got %g want %g", names[i], j, u[j], want[j])
					}
				}
				// The comparison must reject the defects of the previous fused kernel.
				for _, c := range f.controls {
					if parityMismatch(got[c.out], c.want, kernelTolerance) < 0 {
						t.Errorf("%s not detected", c.name)
					}
				}
			})
		}
	}
}

func TestFusedSwiGLUDownParity(t *testing.T) {
	d := newParityDevice(t)
	for _, tt := range swigluParityCases {
		t.Run(tt.name, func(t *testing.T) {
			f := newSwiGLUFixture(tt.hidden, tt.inter, swigluGateSalt)
			dg, du, dw := d.upload(f.gate), d.upload(f.up), d.upload(f.w)
			fused := d.upload(make([]float32, tt.hidden))
			if err := cudalaunch.Kernel(d.gm.fusedSwiGLUFn).Grid(uint32(tt.hidden), 1, 1).Block(128, 1, 1).
				Launch(dg, du, dw, fused, int32(tt.hidden), int32(tt.inter)); err != nil {
				t.Fatal(err)
			}
			da, unfused := d.upload(make([]float32, tt.inter)), d.upload(make([]float32, tt.hidden))
			if err := cudalaunch.Kernel(d.gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1).
				Launch(dg, du, da, int32(tt.inter)); err != nil {
				t.Fatal(err)
			}
			if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(tt.hidden), 1, 1).Block(128, 1, 1).
				Launch(da, dw, nullDevicePtr, unfused, int32(tt.inter), int32(tt.hidden)); err != nil {
				t.Fatal(err)
			}
			got := d.download(fused)
			if i := parityMismatch(got, f.want, kernelTolerance); i >= 0 {
				t.Errorf("fused out[%d]: got %g want %g", i, got[i], f.want[i])
			}
			if u := d.download(unfused); parityMismatch(u, f.want, kernelTolerance) >= 0 {
				i := parityMismatch(u, f.want, kernelTolerance)
				t.Errorf("unfused out[%d]: got %g want %g", i, u[i], f.want[i])
			}
			for _, c := range f.controls {
				if parityMismatch(got, c.want, kernelTolerance) < 0 {
					t.Errorf("%s not detected", c.name)
				}
			}
		})
	}
}

func TestOddRowCopies(t *testing.T) {
	d := newParityDevice(t)
	sentinel := float32(-77)

	const hidden, vocab, token = 7, 5, 3
	table := parityValues(hidden*vocab, 1)
	dTable, dOut := d.upload(table), d.upload(make([]float32, hidden))
	dTok, err := cudamem.Alloc[int32](d.dev, 1)
	t.Cleanup(func() {
		if err := dTok.Free(); err != nil {
			t.Error(err)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := dTok.CopyFromHost([]int32{token}); err != nil {
		t.Fatal(err)
	}
	if err := cudalaunch.Kernel(d.gm.embedFn).Grid(1, 1, 1).Block(256, 1, 1).
		Launch(dTable, dTok, dOut, int32(hidden)); err != nil {
		t.Fatal(err)
	}
	got := d.download(dOut)
	for i, v := range table[token*hidden : (token+1)*hidden] {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("embedding[%d]: got %g want %g", i, got[i], v)
		}
	}

	const kvHeads, headDim, maxSeq, pos = 1, 6, 5, 3
	kvDim := kvHeads * headDim
	k, v := parityValues(kvDim, 3), parityValues(kvDim, 5)
	cache := make([]float32, maxSeq*kvDim)
	for i := range cache {
		cache[i] = sentinel
	}
	dk, dv, kc, vc := d.upload(k), d.upload(v), d.upload(cache), d.upload(cache)
	if err := cudalaunch.Kernel(d.gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1).
		Launch(dk, dv, kc, vc, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
		t.Fatal(err)
	}
	for name, pair := range map[string][2][]float32{"key": {d.download(kc), k}, "value": {d.download(vc), v}} {
		got, row := pair[0], pair[1]
		for i, x := range got {
			want := sentinel
			if i/kvDim == pos {
				want = row[i%kvDim]
			}
			if math.Float32bits(x) != math.Float32bits(want) {
				t.Fatalf("%s cache[%d]: got %g want %g", name, i, x, want)
			}
		}
	}
}

func TestGPUForwardParity(t *testing.T) {
	d := newParityDevice(t)
	major, minor, err := d.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	attnCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	attnMod, err := attnCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() { attnMod.Close() })
	attnFn, err := attnMod.Function("CausalFlashAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	sampleCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	sampleMod, err := sampleCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() { sampleMod.Close() })
	sampleFn, err := sampleMod.Function("ArgmaxLogits")
	if err != nil {
		t.Fatal(err)
	}

	odd := DefaultModelConfig()
	// Query width equals hidden: CPU Forward's output projection assumes it.
	odd.HiddenSize, odd.IntermediateSize, odd.NumAttentionHeads, odd.NumKeyValueHeads, odd.VocabSize = 18, 30, 3, 1, 50
	for _, tt := range []struct {
		name  string
		cfg   ModelConfig
		fused bool
	}{
		{"default", DefaultModelConfig(), false},
		{"default fused", DefaultModelConfig(), true},
		{"odd rows", odd, false},
		{"odd rows fused", odd, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fused && (d.gm.fusedQKVFn == 0 || d.gm.fusedSwiGLUFn == 0) {
				t.Fatal("fused kernels missing from module; the fused path would silently fall back")
			}
			saveFlag, saveQualified := *fuseEGraphFlag, fusedKernelsQualified
			*fuseEGraphFlag, fusedKernelsQualified = tt.fused, tt.fused
			t.Cleanup(func() { *fuseEGraphFlag, fusedKernelsQualified = saveFlag, saveQualified })

			cfg := tt.cfg
			w := GenerateDeterministicWeights(&cfg)
			gw, err := UploadModelToGPU(d.dev, w, &cfg)
			if gw != nil {
				t.Cleanup(func() {
					if err := gw.Free(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			kv, err := cudakv.New(d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
			if kv != nil {
				t.Cleanup(func() {
					if err := kv.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			sp, err := NewGPUScratchpad(d.dev, &cfg)
			if sp != nil {
				t.Cleanup(func() {
					if err := sp.Free(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			hostKV := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())

			tok := int32(7)
			for pos := 0; pos < 6; pos++ {
				want := Forward(w, &cfg, hostKV, tok, pos)
				next, err := GPUForwardPass(gw, &cfg, kv, sp, d.gm, attnFn, sampleFn, tok, pos)
				if err != nil {
					t.Fatal(err)
				}
				got := d.download(sp.Logits)
				if i := parityMismatch(got, want, logitTolerance); i >= 0 {
					t.Fatalf("pos %d logit %d: got %g want %g", pos, i, got[i], want[i])
				}
				if hostNext := sampleArgmax(want); next != hostNext {
					t.Fatalf("pos %d: device token %d, host token %d", pos, next, hostNext)
				}
				tok = next
			}
		})
	}
}

func TestFuseEGraphRefusedUntilQualified(t *testing.T) {
	saveFlag, saveQualified := *fuseEGraphFlag, fusedKernelsQualified
	t.Cleanup(func() { *fuseEGraphFlag, fusedKernelsQualified = saveFlag, saveQualified })
	*fuseEGraphFlag, fusedKernelsQualified = true, false
	// The refusal precedes any device or argument use.
	_, err := GPUForwardPass(nil, nil, nil, nil, nil, 0, 0, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "-fuse-egraph") {
		t.Fatalf("unqualified -fuse-egraph: got %v, want refusal", err)
	}
}

var nullDevicePtr = driver.DevicePtr(0)

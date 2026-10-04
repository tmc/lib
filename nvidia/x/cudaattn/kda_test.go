package cudaattn

import (
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"math"
	"strings"
	"testing"
	"unsafe"
)

func TestKDALaunchFourByteViews(t *testing.T) {
	schema, err := KDASchema()
	if err != nil {
		t.Fatal(err)
	}
	var pointers [10]driver.DevicePtr
	for i := range pointers {
		pointers[i] = driver.DevicePtr(0x1004 + i*0x100)
	}
	args, err := kdaLaunchArguments(schema, pointers, 2, 1, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if len(args) != 13 {
		t.Fatal("wrong argument count")
	}
	for i, p := range pointers {
		if got := *(*driver.DevicePtr)(args[i]); got != p {
			t.Fatalf("pointer %d: %x want %x", i, got, p)
		}
		if uintptr(args[i])%unsafe.Alignof(driver.DevicePtr(0)) != 0 {
			t.Fatal("host pointer slot misaligned")
		}
	}
	if *(*int32)(args[10]) != 2 || *(*int32)(args[11]) != 1 || *(*float32)(args[12]) != 0.25 {
		t.Fatal("scalar ABI differs")
	}
	pointers[5]++
	if _, err := kdaLaunchArguments(schema, pointers, 2, 1, 0.25); err == nil {
		t.Fatal("misaligned FP32 view admitted")
	}
	pointers[5]--
	schema.Params[0].Offset++
	if _, err := kdaLaunchArguments(schema, pointers, 2, 1, 0.25); err == nil {
		t.Fatal("damaged ABI admitted")
	}
}

func TestKDARecurrenceReference(t *testing.T) {
	run := kdaHostKernel(t, KDASource)
	for _, n := range []int{0, 1, 15, 16, 17, 33} {
		for _, slots := range []int{1, 2} {
			t.Run(fmt.Sprintf("length%d/slots%d", n, slots), func(t *testing.T) {
				f := kdaInput(n, slots, slots)
				if err := f.c.checkOffsets(f.offsets); err != nil {
					t.Fatal(err)
				}
				if err := kdaCheckNumerics(f.c, f.q, f.key, f.value, f.decay, f.beta, f.state); err != nil {
					t.Fatal(err)
				}
				s, o, status := run(f)
				ws, wo := kdaOracle(f)
				if status != 0 {
					t.Fatal(status)
				}
				if err := kdaCompare(s, ws, f, KDASource, "host-source-shim"); err != nil {
					t.Fatal(err)
				}
				if err := kdaCompare(o, wo, f, KDASource, "host-source-shim"); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func TestKDANonzeroInitialState(t *testing.T) {
	f := kdaInput(1, 1, 1)
	saved := append([]float32(nil), f.state...)
	s, o := kdaOracle(f)
	zero := f
	zero.state = make([]float32, len(f.state))
	zs, zo := kdaOracle(zero)
	if fmt.Sprint(s) == fmt.Sprint(zs) || fmt.Sprint(o) == fmt.Sprint(zo) {
		t.Fatal("fixture does not distinguish initial state")
	}
	for i := range saved {
		if f.state[i] != saved[i] {
			t.Fatal("oracle mutated saved baseline")
		}
	}
}

func TestKDAStateOrientation(t *testing.T) {
	f := kdaInput(1, 1, 1)
	for i := range f.state {
		f.state[i] = 0
	}
	for i := range f.q {
		f.q[i] = 0
		f.key[i] = 0
		f.value[i] = 0
		f.decay[i] = 1
	}
	f.q[3] = 1
	f.key[7] = 1
	f.beta[0] = 0
	f.state[5*128+3] = 2
	f.state[3*128+5] = 7
	s, o, status := kdaHostKernel(t, KDASource)(f)
	if status != 0 || o[5] != float32(2)*f.c.Scale || s[5*128+3] != 2 {
		t.Fatal("[V,K] orientation/readout mismatch")
	}
}

func TestKDAChunkReplay(t *testing.T) {
	run := kdaHostKernel(t, KDASource)
	f := kdaInput(33, 1, 1)
	ws, wo := kdaOracle(f)
	saved := append([]float32(nil), f.state...)
	for _, split := range []int{1, 15, 16, 17} {
		t.Run(fmt.Sprint(split), func(t *testing.T) {
			state := append([]float32(nil), saved...)
			var output []float32
			for _, span := range [][2]int{{0, split}, {split, 33}} {
				part := f
				part.c.Tokens = span[1] - span[0]
				part.offsets = []int32{0, int32(part.c.Tokens)}
				part.state = state
				part.q = f.q[span[0]*128 : span[1]*128]
				part.key = f.key[span[0]*128 : span[1]*128]
				part.value = f.value[span[0]*128 : span[1]*128]
				part.decay = f.decay[span[0]*128 : span[1]*128]
				part.beta = f.beta[span[0]:span[1]]
				var o []float32
				var status int32
				state, o, status = run(part)
				if status != 0 {
					t.Fatal(status)
				}
				output = append(output, o...)
			}
			if err := kdaCompare(state, ws, f, KDASource, "host-source-shim"); err != nil {
				t.Fatal(err)
			}
			if err := kdaCompare(output, wo, f, KDASource, "host-source-shim"); err != nil {
				t.Fatal(err)
			}
		})
	}
	for i := range saved {
		if saved[i] != f.state[i] {
			t.Fatal("saved baseline changed")
		}
	}
}

func TestKDASequenceIsolation(t *testing.T) {
	f := kdaInput(17, 2, 2)
	run := kdaHostKernel(t, KDASource)
	s, o, _ := run(f)
	other := f
	other.state = append([]float32(nil), f.state...)
	for i := 0; i < 2*128*128; i++ {
		other.state[i] *= 4
	}
	ss, oo, _ := run(other)
	for i := 2 * 128 * 128; i < len(s); i++ {
		if s[i] != ss[i] {
			t.Fatal("state leaked across sequences")
		}
	}
	for i := 17 * 2 * 128; i < len(o); i++ {
		if o[i] != oo[i] {
			t.Fatal("output leaked across sequences")
		}
	}
}

func TestKDASequenceOffsets(t *testing.T) {
	c := KDAConfig{128, 1, 2, 17, 1}
	for _, a := range [][]int32{nil, {0, 18, 17}, {0, -1, 17}, {0, 17, 18}, {1, 1, 17}, {0, 17}} {
		if c.checkOffsets(a) == nil {
			t.Fatalf("accepted offsets %v", a)
		}
	}
	if c.checkOffsets([]int32{0, 0, 17}) != nil {
		t.Fatal("valid empty segment refused")
	}
	c.Tokens = 34
	if c.checkOffsets([]int32{0, 34, 34}) == nil {
		t.Fatal("overlong sequence accepted")
	}
}

func TestKDAEmptySegment(t *testing.T) {
	f := kdaInput(1, 2, 1)
	f.c.Tokens = 1
	f.offsets = []int32{0, 0, 1}
	f.q = f.q[:128]
	f.key = f.key[:128]
	f.value = f.value[:128]
	f.decay = f.decay[:128]
	f.beta = f.beta[:1]
	s, _, status := kdaHostKernel(t, KDASource)(f)
	if status != 0 {
		t.Fatal(status)
	}
	for i := 0; i < 128*128; i++ {
		if s[i] != f.state[i] {
			t.Fatal("empty segment changed initial state")
		}
	}
}

func TestKDAAdmission(t *testing.T) {
	f := kdaInput(1, 1, 1)
	for _, name := range []string{"nan", "infinity", "norm", "decay", "beta"} {
		t.Run(name, func(t *testing.T) {
			g := kdaInput(1, 1, 1)
			switch name {
			case "nan":
				g.state[0] = float32(math.NaN())
			case "infinity":
				g.value[0] = float32(math.Inf(1))
			case "norm":
				g.q[0] = 10
			case "decay":
				g.decay[0] = -1
			case "beta":
				g.beta[0] = 2
			}
			if kdaCheckNumerics(g.c, g.q, g.key, g.value, g.decay, g.beta, g.state) == nil {
				t.Fatal("bad numerical domain admitted")
			}
		})
	}
	schema, err := KDASchema()
	if err != nil || len(schema.Params) != 13 {
		t.Fatalf("schema: %v %v", schema, err)
	}
	ids := []KDAIdentity{{"a", 7, KDAFP32}}
	s := &KDAState{ids: ids, heads: 1, scale: f.c.Scale}
	next, err := kdaAdvance(f.c, s, ids, f.offsets)
	if err != nil || next[0].Prefix != 8 {
		t.Fatal(next, err)
	}
	for _, wrong := range []KDAIdentity{{"b", 7, KDAFP32}, {"a", 8, KDAFP32}, {"a", 7, "bf16"}} {
		if _, err := kdaAdvance(f.c, s, []KDAIdentity{wrong}, f.offsets); err == nil {
			t.Fatal("wrong state identity admitted")
		}
	}
	s.ids[0].Prefix = ^uint64(0)
	if _, err := kdaAdvance(f.c, s, s.ids, f.offsets); err == nil {
		t.Fatal("prefix overflow admitted")
	}
}

func kdaMutants() map[string]string {
	return map[string]string{
		"transpose":       strings.Replace(KDASource, "initial[base+i]", "initial[base+(i%128)*128+i/128]", 1),
		"late-decay":      strings.Replace(strings.Replace(KDASource, "__fmul_rn(state[row+k],decay[token+k])", "state[row+k]", 1), "state[row+k]=__fadd_rn(state[row+k],__fmul_rn(correction,key[token+k]));", "state[row+k]=__fmul_rn(__fadd_rn(state[row+k],__fmul_rn(correction,key[token+k])),decay[token+k]);", 1),
		"old-readout":     strings.Replace(KDASource, "__fmul_rn(state[row+k],q[token+k])", "__fmul_rn(initial[row+k],q[token+k])", 1),
		"drop-beta":       strings.Replace(KDASource, "beta[t*heads+h]", "1.0f", 1),
		"drop-correction": strings.Replace(KDASource, "__fsub_rn(value[token+v],dot)", "value[token+v]", 1),
		"omit-last":       strings.Replace(KDASource, "t<offsets[s+1]", "t<offsets[s+1]-1", 1),
		"no-op":           `extern "C" __global__ void kda_fp32(const float*,const float*,const float*,const float*,const float*,const float*,const int*,float*,float*,int*,int,int,float){}`,
	}
}

func TestKDABehavioralMutants(t *testing.T) {
	f := kdaInput(17, 2, 2)
	ws, wo := kdaOracle(f)
	for name, source := range kdaMutants() {
		t.Run(name, func(t *testing.T) {
			s, o, status := kdaHostKernel(t, source)(f)
			badState := kdaCompare(s, ws, f, source, "host-source-shim") != nil
			badOutput := kdaCompare(o, wo, f, source, "host-source-shim") != nil
			if status == 0 && !badState && !badOutput {
				t.Fatal("mutant survived numerical oracle")
			}
			if name == "no-op" && (!badState || !badOutput || s[0] != -12345 || o[0] != -12345) {
				t.Fatal("no-op did not fail both numerical and output-write controls")
			}
		})
	}
}

func TestKDAStateLifetime(t *testing.T) {
	for _, phase := range []string{"compute", "publication"} {
		t.Run(phase, func(t *testing.T) {
			released, copied := 0, 0
			late := true
			failed := errors.New("failed drain")
			wait := func(ctx context.Context) error {
				if ctx != nil && ctx.Err() != nil {
					return ctx.Err()
				}
				if late {
					return failed
				}
				return nil
			}
			r := &KDAResult{owner: &KDA{config: KDAConfig{128, 1, 1, 1, 1}}, ids: []KDAIdentity{{"a", 8, KDAFP32}}, releases: []func(){func() { released++ }}, readStatus: func() error { return nil }, copyResults: func() error { copied++; return nil }, computeWait: func(context.Context) error { return nil }, publishWait: func(context.Context) error { return nil }}
			if phase == "compute" {
				r.computeWait = wait
			} else {
				r.publishWait = wait
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if !errors.Is(r.Wait(ctx), context.Canceled) || released != 0 || r.done.Load() {
				t.Fatal("cancellation released queued state")
			}
			if r.Wait(nil) == nil || released != 0 || r.done.Load() {
				t.Fatal("failed drain released queued state")
			}
			if phase == "compute" && copied != 0 {
				t.Fatal("publication submitted before compute completed")
			}
			r.state = &KDAState{ids: r.ids} // A stale field must still be inaccessible.
			if _, err := r.State(); err == nil {
				t.Fatal("published before completion")
			}
			r.state = nil
			late = false
			if err := r.Wait(nil); err != nil {
				t.Fatal(err)
			}
			if released != 1 || copied != 1 {
				t.Fatal(released, copied)
			}
			if _, err := r.State(); err != nil {
				t.Fatal(err)
			}
			if err := r.Close(); err != nil || released != 1 {
				t.Fatal("repeated close released twice")
			}
		})
	}
	t.Run("partial-copy", func(t *testing.T) {
		released := 0
		late := true
		r := &KDAResult{owner: &KDA{}, releases: []func(){func() { released++ }}, computeWait: func(context.Context) error { return nil }, readStatus: func() error { return nil }, copyResults: func() error { return errors.New("second copy failed") }, publishWait: func(context.Context) error {
			if late {
				return errors.New("drain failed")
			}
			return nil
		}}
		if r.Wait(nil) == nil || released != 0 {
			t.Fatal("partial copy released without drain")
		}
		late = false
		if r.Wait(nil) == nil || released != 1 {
			t.Fatal("partial copy incorrectly published")
		}
		if _, err := r.State(); err == nil {
			t.Fatal("partial copy admitted state")
		}
	})
	t.Run("partial-submit", func(t *testing.T) {
		released, copied := 0, 0
		late := true
		submit := errors.New("partial submission failed")
		r := &KDAResult{owner: &KDA{}, submitErr: submit, releases: []func(){func() { released++ }}, computeWait: func(context.Context) error {
			if late {
				return errors.New("drain failed")
			}
			return nil
		}, copyResults: func() error { copied++; return nil }}
		if r.Wait(nil) == nil || released != 0 {
			t.Fatal("partial submit released without drain")
		}
		late = false
		if !errors.Is(r.Wait(nil), submit) || released != 1 || copied != 0 {
			t.Fatal("failed submit published")
		}
		if _, err := r.State(); err == nil {
			t.Fatal("failed submit admitted state")
		}
	})
	t.Run("missing-status", func(t *testing.T) {
		released, copied := 0, 0
		r := &KDAResult{owner: &KDA{}, releases: []func(){func() { released++ }}, computeWait: func(context.Context) error { return nil }, readStatus: func() error { return errors.New("missing status") }, copyResults: func() error { copied++; return nil }}
		if r.Wait(nil) == nil || released != 1 || copied != 0 {
			t.Fatal("missing status published or retained completed pins")
		}
	})
}

func TestKDABufferAdmission(t *testing.T) {
	dev, other := &cuda.Device{}, &cuda.Device{}
	good := []kdaBuffer{{name: "initial", ptr: 0x1000, length: 128, need: 128, dev: dev}, {name: "final", ptr: 0x2000, length: 128, need: 128, dev: dev}}
	for _, name := range []string{"valid", "alias", "adjacent", "wrong-device", "extent", "alignment", "overflow", "nil-device"} {
		t.Run(name, func(t *testing.T) {
			bufs := append([]kdaBuffer(nil), good...)
			target := dev
			switch name {
			case "alias":
				bufs[1].ptr = bufs[0].ptr + 4
			case "adjacent":
				bufs[1].ptr = bufs[0].ptr + 512
			case "wrong-device":
				bufs[1].dev = other
			case "extent":
				bufs[1].length++
			case "alignment":
				bufs[1].ptr++
			case "overflow":
				bufs[1].ptr = ^uint64(0) - 3
			case "nil-device":
				target = nil
			}
			err := kdaCheckBuffers(bufs, target)
			if (err == nil) != (name == "valid" || name == "adjacent") {
				t.Fatal(name, err)
			}
		})
	}
	for _, failAt := range []int{0, 1, 2} {
		t.Run(fmt.Sprintf("borrow%d", failAt), func(t *testing.T) {
			borrowed, released := 0, 0
			bufs := append([]kdaBuffer(nil), good...)
			for i := range bufs {
				index := i
				bufs[i].borrow = func(*cuda.Device) (func(), error) {
					if index == failAt {
						return nil, errors.New("borrow failed")
					}
					borrowed++
					return func() { released++ }, nil
				}
			}
			releases, err := kdaBorrowBuffers(bufs, dev)
			if failAt < 2 {
				if err == nil || released != borrowed {
					t.Fatal("borrow failure leaked earlier pins")
				}
			} else {
				if err != nil || released != 0 {
					t.Fatal(err)
				}
				for _, release := range releases {
					release()
				}
				if released != 2 {
					t.Fatal("missing releases")
				}
			}
		})
	}
}

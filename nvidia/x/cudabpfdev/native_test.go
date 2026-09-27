package cudabpfdev

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

const nativeNegatePTX = `.version 9.0
.target TARGET
.address_size 64
.visible .entry negate(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<3>;
    .reg .f32 %value;
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mad.lo.u32 %r0, %r1, %r2, %r0;
    mul.wide.u32 %rd2, %r0, 4;
    add.u64 %rd0, %rd0, %rd2;
    add.u64 %rd1, %rd1, %rd2;
    ld.global.f32 %value, [%rd0];
    neg.f32 %value, %value;
    st.global.f32 [%rd1], %value;
    ret;
}
`

func nativeInjectionFixture(target string, block uint32) (*ArtifactSource, Program, Budget, InjectionPolicy, error) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "u64", IsPointer: true}, cudakernel.Param{Name: "output", Type: "u64", IsPointer: true})
	if err != nil {
		return nil, Program{}, Budget{}, InjectionPolicy{}, err
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(strings.ReplaceAll(nativeNegatePTX, "TARGET", target)), "negate", schema, target)
	if err != nil {
		return nil, Program{}, Budget{}, InjectionPolicy{}, err
	}
	s, err := NewArtifactSource(a, Geometry{Grid: [3]uint32{2, 1, 1}, Block: [3]uint32{block, 1, 1}})
	p := Program{Version: 1, Registers: 4, Slots: 2, Code: []Instruction{
		{Op: Const, Dst: 1, Imm: 1}, {Op: MapAdd, B: 1},
		{Op: Const, Dst: 2, Imm: 2}, {Op: MapAdd, A: 2, B: 1},
		{Op: Lane, Dst: 3}, {Op: Leader, Dst: 3, A: 3}, {Op: Exit, A: 3},
	}}
	b := Budget{MaxInstructions: 16, MaxSteps: 16, MaxRegisters: 128, MaxSlots: 2}
	policy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{a.Digest, a.EntryPoint, a.Target}}, Entry: true, Exit: true, MemoryInstructions: []int{11}}
	return s, p, b, policy, err
}

func nativeStubInjectionFixture(target string, block uint32) (*ArtifactSource, Program, Budget, InjectionPolicy, error) {
	ptx := strings.Replace(nativeNegatePTX, "    ret;\n}", "    call.uni trace_stub;\n    ret;\n}\n\n.func trace_stub\n{\n    ret;\n}", 1)
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "u64", IsPointer: true}, cudakernel.Param{Name: "output", Type: "u64", IsPointer: true})
	if err != nil {
		return nil, Program{}, Budget{}, InjectionPolicy{}, err
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(strings.ReplaceAll(ptx, "TARGET", target)), "negate", schema, target)
	if err != nil {
		return nil, Program{}, Budget{}, InjectionPolicy{}, err
	}
	s, err := NewArtifactSource(a, Geometry{Grid: [3]uint32{2, 1, 1}, Block: [3]uint32{block, 1, 1}})
	p := Program{Version: 1, Registers: 1, Slots: 1, Code: []Instruction{{Op: Const, Dst: 0, Imm: 1}, {Op: MapAdd, B: 0}, {Op: Exit}}}
	b := Budget{MaxInstructions: 8, MaxSteps: 8, MaxRegisters: 128, MaxSlots: 1}
	policy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{a.Digest, a.EntryPoint, a.Target}}, StubFunctions: []string{"trace_stub"}}
	return s, p, b, policy, err
}

// This host check ensures the gated workload actually exercises candidate
// admission; a silently falling-back workload is not a native injection test.
func TestNativeInjectionFixture(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_121a"} {
		for _, block := range []uint32{17, 32, 64} {
			s, p, b, policy, err := nativeInjectionFixture(target, block)
			if err != nil {
				t.Fatal(err)
			}
			i, err := PrepareInjection(s, p, b, policy)
			if err != nil {
				t.Fatal(err)
			}
			if a, reason := i.Candidate(); a == nil {
				t.Fatal(reason)
			}
		}
	}
	for _, block := range []uint32{17, 32} {
		s, p, b, policy, err := nativeStubInjectionFixture("sm_121a", block)
		if err != nil {
			t.Fatal(err)
		}
		i, err := PrepareInjection(s, p, b, policy)
		if err != nil {
			t.Fatal(err)
		}
		a, reason := i.Candidate()
		if a == nil {
			t.Fatal(reason)
		}
		if _, err := cudaptx.ParseBytes(a.Raw); err != nil {
			t.Fatalf("stub candidate is not parseable by cudaptx: %v", err)
		}
	}
}

// TestNativeInjection requires explicit hardware authorization outside this
// test. The environment opts into the named workload; it is not authorization
// by itself. Once opted in, missing tools/device/lease fail instead of skipping.
func TestNativeInjection(t *testing.T) {
	if os.Getenv("CUDABPFDEV_NATIVE") != "1" {
		t.Skip("CUDABPFDEV_NATIVE=1 requires scoped hardware authorization")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDABPFDEV_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("CUDABPFDEV_DEVICE must name an explicit nonnegative ordinal")
	}
	target := os.Getenv("CUDABPFDEV_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("CUDABPFDEV_TARGET must be sm_100a or sm_121a")
	}
	repetitions := 5
	if s := os.Getenv("CUDABPFDEV_REPETITIONS"); s != "" {
		repetitions, err = strconv.Atoi(s)
		if err != nil || repetitions < 3 || repetitions > 100 {
			t.Fatal("CUDABPFDEV_REPETITIONS must be in [3,100]")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	var dev *cuda.Device
	var loaded []*LoadedInjection
	var allocations []driver.DevicePtr
	var events []driver.Event
	// All cleanup is ordered here. If a wait fails, retain buffers, contexts and
	// the lease until process exit; never free caller storage underneath work.
	t.Cleanup(func() {
		failed := false
		for _, l := range loaded {
			if err := l.Close(); err != nil {
				t.Errorf("retained invocation: %v", err)
				failed = true
			}
		}
		if failed {
			return
		}
		if dev != nil {
			err := dev.WithContext(func() error {
				for n, e := range events {
					if e != 0 {
						if err := e.Close(); err != nil {
							return err
						}
						events[n] = 0
					}
				}
				for n, p := range allocations {
					if p != 0 {
						if err := p.Free(); err != nil {
							return err
						}
						allocations[n] = 0
					}
				}
				return nil
			})
			if err != nil {
				t.Errorf("retained native resources: %v", err)
				return
			}
			if err := dev.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	dev, err = cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatalf("device SM %d.%d does not match %s", major, minor, target)
	}
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	linkMajor, linkMinor, err := nvjitlink.Version()
	if err != nil {
		t.Fatal(err)
	}
	rtcMajor, rtcMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	tool, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Fatal(err)
	}
	toolVersion, err := cudatool.Run(ctx, tool, "", nil, "--version")
	if err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	logNative(t, map[string]any{"kind": "environment", "host": host, "device": name, "ordinal": ordinal, "uuid": uuid, "target": target,
		"driver_cuda_version": driverVersion, "driver_library": driver.Path(), "nvjitlink_version": fmt.Sprintf("%d.%d", linkMajor, linkMinor),
		"nvrtc_version": fmt.Sprintf("%d.%d", rtcMajor, rtcMinor), "ptxas": toolVersion.Path, "ptxas_version": string(toolVersion.Stdout) + string(toolVersion.Stderr),
		"nvrtc_library": nvrtc.Path(), "nvjitlink_library": nvjitlink.Path(),
		"gpu_lock_held": os.Getenv("GPU_LOCK_HELD"), "lease": lease.Owner(), "lease_dir": lease.Dir(), "nested_lease": lease.IsNested(), "repetitions": repetitions, "go": runtime.Version()})
	alloc := func(bytes uint) driver.DevicePtr {
		var p driver.DevicePtr
		err := dev.WithContext(func() error {
			var err error
			p, err = driver.MemAllocBytes(bytes)
			if p != 0 {
				allocations = append(allocations, p)
			}
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	for _, block := range []uint32{17, 32, 64} {
		source, program, budget, policy, err := nativeInjectionFixture(target, block)
		if err != nil {
			t.Fatal(err)
		}
		n := int(block) * 2
		input := make([]float32, n)
		for j := range input {
			input[j] = float32(j-17) / 4
		}
		input[0], input[1] = 0, math.Float32frombits(1<<31)
		inputBits := make([]uint32, n)
		for j, value := range input {
			inputBits[j] = math.Float32bits(value)
		}
		logNative(t, map[string]any{"kind": "workload", "block": block, "original_artifact": source.launch.Artifact,
			"program": program, "budget": budget, "policy": policy, "input_float32_bits": inputBits, "reference_operation": "cudaref neg", "numerical_policy": "exact"})
		in, originalOut, candidateOut := alloc(uint(n*4)), alloc(uint(n*4)), alloc(uint(n*4))
		if err := nativeFloatTransfer(dev, in, input, false); err != nil {
			t.Fatal(err)
		}
		graph := cudair.NewGraph()
		x := graph.AddInput("x", []int{n}, cudair.Float32)
		y := graph.AddUnary("neg", x)
		if err := graph.SetOutputs(y); err != nil {
			t.Fatal(err)
		}
		for repetition := -1; repetition < repetitions; repetition++ {
			outputs := [2][]float32{}
			// Alternate pair order to avoid assigning every first-launch effect
			// to the original. Repetition -1 is reported as warmup, not discarded.
			order := []int{0, 1}
			if repetition%2 == 0 {
				order = []int{1, 0}
			}
			for _, mode := range order {
				candidate := mode == 1
				selectedPolicy := policy
				selectedPolicy.Enabled = candidate
				start := time.Now()
				if _, err := Translate(program, budget, target); err != nil {
					t.Fatal(err)
				}
				translationProbe := time.Since(start)
				start = time.Now()
				injection, err := PrepareInjection(source, program, budget, selectedPolicy)
				prepare := time.Since(start)
				if err != nil {
					t.Fatal(err)
				}
				ops := nativeInjectionLoadOps(dev)
				if candidate && repetition == -1 {
					a, reason := injection.Candidate()
					logNative(t, map[string]any{"kind": "prepared_candidate", "block": block, "artifact": a, "fallback_reason": reason})
				}
				compile := ops.compile
				var linkTime time.Duration
				ops.compile = func(a *cudakernel.Artifact) (*cudakernel.Artifact, *nvjitlinkx.Result, error) {
					start := time.Now()
					artifact, result, err := compile(a)
					linkTime = time.Since(start)
					if result != nil {
						logNative(t, map[string]any{"kind": "link_diagnostics", "info": result.InfoLog, "error": result.ErrorLog})
					}
					if artifact != nil {
						logNative(t, map[string]any{"kind": "linked_artifact", "block": block, "repetition": repetition, "artifact": artifact})
					}
					return artifact, result, err
				}
				var begin, end driver.Event
				launch := ops.launch
				ops.launch = func(fn driver.Function, geometry Geometry, stream driver.Stream, schema cudakernel.Schema, packed []byte) error {
					var err error
					begin, err = driver.NewEvent(0)
					if begin != 0 {
						events = append(events, begin)
					}
					if err != nil {
						return err
					}
					end, err = driver.NewEvent(0)
					if end != 0 {
						events = append(events, end)
					}
					if err != nil {
						return err
					}
					if err := begin.Record(stream); err != nil {
						return err
					}
					if err := launch(fn, geometry, stream, schema, packed); err != nil {
						return err
					}
					return end.Record(stream)
				}
				start = time.Now()
				owner, err := injection.load(ops)
				loadTime := time.Since(start)
				if owner != nil {
					loaded = append(loaded, owner)
				}
				if err != nil {
					t.Fatal(err)
				}
				selection, err := owner.Selection()
				if err != nil || selection.Candidate != candidate {
					t.Fatalf("unexpected selection: %+v %v", selection, err)
				}
				out := originalOut
				if candidate {
					out = candidateOut
				}
				// Distinct poison makes missing stores visible, including zeros.
				poison := make([]float32, n)
				for j := range poison {
					poison[j] = 12345
				}
				if err := nativeFloatTransfer(dev, out, poison, false); err != nil {
					t.Fatal(err)
				}
				start = time.Now()
				err = owner.Launch(ctx, in, out)
				launchWall := time.Since(start)
				if err != nil {
					t.Fatal(err)
				}
				var elapsed float32
				if err := dev.WithContext(func() error { var err error; elapsed, err = begin.ElapsedTime(end); return err }); err != nil {
					t.Fatal(err)
				}
				outputs[mode] = make([]float32, n)
				if err := nativeFloatTransfer(dev, out, outputs[mode], true); err != nil {
					t.Fatal(err)
				}
				snapshot, available := owner.Snapshot()
				if available != candidate {
					t.Fatal("snapshot selection mismatch")
				}
				if candidate {
					want := uint32(2 * ((block + 31) / 32) * 3)
					if len(snapshot.Values) != 2 || snapshot.Values[0] != want || snapshot.Values[1] != 0 || snapshot.Misses != uint64(want) {
						t.Fatalf("counter oracle: got %+v want %d", snapshot, want)
					}
				}
				record, err := owner.Record()
				if err != nil {
					t.Fatal(err)
				}
				logNative(t, map[string]any{"kind": "sample", "block": block, "grid": 2, "repetition": repetition, "warmup": repetition < 0,
					"selection": selection, "record": record, "map": snapshot, "translation_probe_ns": translationProbe.Nanoseconds(),
					"prepare_including_translation_ns": prepare.Nanoseconds(), "candidate_link_ns": linkTime.Nanoseconds(), "load_including_link_ns": loadTime.Nanoseconds(),
					"launch_including_sync_readback_ns": launchWall.Nanoseconds(), "gpu_event_ms": elapsed, "timings_additive": false})
				if err := owner.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if err := CompareOutputs(graph, map[string][]float32{"x": input}, map[string][]float32{y.Name: outputs[0]}, map[string][]float32{y.Name: outputs[1]}, NumericalPolicy{Mode: "exact"}); err != nil {
				t.Fatal(err)
			}
			stubSource, stubProgram, stubBudget, stubPolicy, err := nativeStubInjectionFixture(target, block)
			if err != nil {
				t.Fatal(err)
			}
			stubInjection, err := PrepareInjection(stubSource, stubProgram, stubBudget, stubPolicy)
			if err != nil {
				t.Fatal(err)
			}
			stubOwner, err := stubInjection.load(nativeInjectionLoadOps(dev))
			if stubOwner != nil {
				loaded = append(loaded, stubOwner)
			}
			if err != nil {
				t.Fatal(err)
			}
			stubSelection, err := stubOwner.Selection()
			if err != nil || !stubSelection.Candidate {
				t.Fatalf("stub candidate was not selected: %+v %v", stubSelection, err)
			}
			if err := stubOwner.Launch(ctx, in, candidateOut); err != nil {
				t.Fatal(err)
			}
			stubSnapshot, ok := stubOwner.Snapshot()
			if !ok {
				t.Fatal("stub candidate snapshot unavailable")
			}
			active := uint32(0xffffffff)
			if block < 32 {
				active = (uint32(1) << block) - 1
			}
			perWarp, err := Evaluate(ctx, stubProgram, stubBudget, active)
			if err != nil {
				t.Fatal(err)
			}
			warpCount := uint32(2 * ((block + 31) / 32))
			wantCount := perWarp.Map[0] * warpCount
			wantMisses := perWarp.Misses * uint64(warpCount)
			if len(stubSnapshot.Values) != 1 || stubSnapshot.Values[0] != wantCount || stubSnapshot.Misses != wantMisses {
				t.Fatalf("stub map differs from Evaluate per warp: got %+v want count=%d misses=%d", stubSnapshot, wantCount, wantMisses)
			}
			stubOutput := make([]float32, n)
			if err := nativeFloatTransfer(dev, candidateOut, stubOutput, true); err != nil {
				t.Fatal(err)
			}
			if err := CompareOutputs(graph, map[string][]float32{"x": input}, map[string][]float32{y.Name: outputs[0]}, map[string][]float32{y.Name: stubOutput}, NumericalPolicy{Mode: "exact"}); err != nil {
				t.Fatalf("stub candidate changed kernel output: %v", err)
			}
			if err := stubOwner.Close(); err != nil {
				t.Fatal(err)
			}
			// A physical-output corruption must fail the same qualification gate.
			outputs[1][n-1]++
			if err := CompareOutputs(graph, map[string][]float32{"x": input}, map[string][]float32{y.Name: outputs[0]}, map[string][]float32{y.Name: outputs[1]}, NumericalPolicy{Mode: "exact"}); err == nil {
				t.Fatal("corrupted native output passed")
			}
		}
	}
}

func nativeFloatTransfer(dev *cuda.Device, ptr driver.DevicePtr, data []float32, read bool) error {
	var pin runtime.Pinner
	pin.Pin(&data[0])
	defer pin.Unpin()
	return dev.WithContext(func() error {
		if read {
			return ptr.CopyToHostPtr(unsafe.Pointer(&data[0]), uint(len(data)*4))
		}
		return ptr.CopyFromHostPtr(unsafe.Pointer(&data[0]), uint(len(data)*4))
	})
}

func logNative(t *testing.T, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(data))
}

package nvl72

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

// TestDecideColdMoEPlacement is the crossover oracle: on VR200 a 36 MB FP8
// expert hit by one decode token runs in place on the Vera cores and keeps
// nearly all 36 MB off C2C, while a 64-token prefill batch goes to the GPU.
func TestDecideColdMoEPlacement(t *testing.T) {
	vr, err := C2CPreset("vr200")
	if err != nil {
		t.Fatal(err)
	}
	one, err := DecideColdMoEPlacement(vr, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !one.RunOnArmInPlace || one.ArmInPlaceTimeUs >= one.GPUDMATimeUs {
		t.Fatalf("1 token: %+v, want CPU in place", one)
	}
	if one.BandwidthSavedMB <= 35.9 {
		t.Fatalf("1 token saves %.3f MB of C2C, want > 35.9", one.BandwidthSavedMB)
	}
	prefill, err := DecideColdMoEPlacement(vr, 64)
	if err != nil {
		t.Fatal(err)
	}
	if prefill.RunOnArmInPlace || prefill.BandwidthSavedMB != 0 {
		t.Fatalf("64 tokens: %+v, want GPU", prefill)
	}
	t.Logf("vr200: 1 token CPU %.1f us vs GPU %.1f us; 64 tokens CPU %.1f us vs GPU %.1f us",
		one.ArmInPlaceTimeUs, one.GPUDMATimeUs, prefill.ArmInPlaceTimeUs, prefill.GPUDMATimeUs)

	for _, arch := range []string{"gb200", "gb300", "vr200"} {
		cfg, err := C2CPreset(arch)
		if err != nil {
			t.Fatal(err)
		}
		k, err := ColdMoECrossover(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if k < 2 || k > 64 {
			t.Fatalf("%s crossover %d tokens, want in 2..64", arch, k)
		}
		below, _ := DecideColdMoEPlacement(cfg, k-1)
		at, _ := DecideColdMoEPlacement(cfg, k)
		if !below.RunOnArmInPlace || at.RunOnArmInPlace {
			t.Fatalf("%s: crossover %d not where the decision flips", arch, k)
		}
		t.Logf("%s: CPU in place below %d tokens", arch, k)
	}
	if _, err := C2CPreset("h100"); err == nil {
		t.Fatal("C2CPreset(h100) succeeded")
	}
	if _, err := DecideColdMoEPlacement(vr, 0); err == nil {
		t.Fatal("DecideColdMoEPlacement(0 tokens) succeeded")
	}
}

func TestPlanGrammarOverlap(t *testing.T) {
	cfg, _ := C2CPreset("gb200")
	tests := []struct {
		layers, task float64
		stall        float64
	}{
		{layers: 400, task: 2, stall: 0},
		{layers: 10, task: 20, stall: 10.5},
	}
	for _, tc := range tests {
		o, err := PlanGrammarOverlap(cfg, tc.layers, tc.task)
		if err != nil {
			t.Fatal(err)
		}
		if !(math.Abs(o.StallUs-tc.stall) <= 1e-9) {
			t.Fatalf("layers %v task %v: stall %v, want %v", tc.layers, tc.task, o.StallUs, tc.stall)
		}
	}
}

func TestC2CDoorbellSlotLayout(t *testing.T) {
	var s C2CDoorbellSlot
	offsets := []struct {
		got, want uintptr
	}{
		{unsafe.Offsetof(s.SeqNo), C2CSlotSeqNo},
		{unsafe.Offsetof(s.TaskKind), C2CSlotTaskKind},
		{unsafe.Offsetof(s.NumTokens), C2CSlotNumTokens},
		{unsafe.Offsetof(s.InputAddr), C2CSlotInputAddr},
		{unsafe.Offsetof(s.WeightAddr), C2CSlotWeightAddr},
		{unsafe.Offsetof(s.OutputAddr), C2CSlotOutputAddr},
		{unsafe.Offsetof(s.CompletionSeq), C2CSlotCompletionSeq},
	}
	for i, o := range offsets {
		if o.got != o.want {
			t.Fatalf("field %d at %d, want %d", i, o.got, o.want)
		}
	}
	in := C2CDoorbellSlot{SeqNo: 7, TaskKind: TaskGrammarBitmaskDFA, NumTokens: 64, InputAddr: 0x1000, WeightAddr: 0x2000, OutputAddr: 0x3000, CompletionSeq: 6}
	b, err := in.MarshalBinary()
	if err != nil || len(b) != C2CDoorbellSlotBytes {
		t.Fatalf("MarshalBinary = %d bytes, %v", len(b), err)
	}
	var out C2CDoorbellSlot
	if err := out.UnmarshalBinary(b); err != nil || out != in {
		t.Fatalf("round trip %+v, %v; want %+v", out, err, in)
	}
	if err := out.UnmarshalBinary(b[:48]); err == nil {
		t.Fatal("UnmarshalBinary accepted 48 bytes")
	}
}

// TestAuditC2CCoherence: the emitted doorbell passes, and narrowing either
// handoff to .gpu or .cta scope, or dropping the scope, is rejected.
func TestAuditC2CCoherence(t *testing.T) {
	ptx, err := EmitC2CSysDoorbellPTX("c2c_doorbell")
	if err != nil {
		t.Fatal(err)
	}
	if err := AuditC2CCoherence(ptx); err != nil {
		t.Fatalf("emitted PTX: %v", err)
	}
	src, err := C2CDoorbellKernel{Rounds: 2, Rows: 4, Cols: 4}.Source()
	if err != nil {
		t.Fatal(err)
	}
	if err := AuditC2CCoherence(src); err != nil {
		t.Fatalf("emitted kernel: %v", err)
	}
	if strings.Index(ptx, "st.global.u64 [%slot+32]") > strings.Index(ptx, "st.release.sys") {
		t.Fatal("descriptor written after the release that publishes it")
	}
	mutants := map[string]string{
		"release_gpu":   strings.Replace(ptx, "st.release.sys", "st.release.gpu", 1),
		"acquire_cta":   strings.Replace(ptx, "ld.acquire.sys", "ld.acquire.cta", 1),
		"acquire_none":  strings.Replace(ptx, "ld.acquire.sys.global", "ld.acquire.global", 1),
		"no_release":    strings.Replace(ptx, "st.release.sys", "st", 1),
		"gpu_atomic":    ptx + "\tatom.add.gpu.global.u64 %t, [%slot], 1;\n",
		"default_atom":  ptx + "\tatom.global.add.u64 %t, [%slot], 1;\n",
		"cluster_fence": ptx + "\tfence.acq_rel.cluster;\n",
	}
	for name, m := range mutants {
		if err := AuditC2CCoherence(m); err == nil {
			t.Errorf("%s: audit passed", name)
		}
	}
	// Multimem and .sys atomics are fine.
	if err := AuditC2CCoherence(ptx + "\tmultimem.red.relaxed.sys.global.add.u32 [%slot], 1;\n\tfence.sc.sys;\n"); err != nil {
		t.Fatalf("sys-scoped additions rejected: %v", err)
	}
}

// TestC2CDoorbellKernelHostEmu runs the persistent kernel against the CPU
// worker on a host thread: every round's output must be the worker's GEMV
// of that round's input, delivered through acquire/release handoffs only.
func TestC2CDoorbellKernelHostEmu(t *testing.T) {
	const rounds, rows, cols = 8, 6, 10
	k := C2CDoorbellKernel{Name: "c2c_doorbell", Rounds: rounds, Rows: rows, Cols: cols}
	src, err := k.Source()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(51, 52))
	x := make([]float32, rounds*cols)
	w := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	out, err := cudahostemu.Run(context.Background(), t.TempDir(), cudahostemu.Program{
		Source: src + "\n#include <thread>\n",
		Launch: `std::thread worker([&] { rc[0] = eg_c2c_worker(slot, w); });
    eg_emu_launch(dim3(1), dim3(32), 0, c2c_doorbell, slot, x, y, lat, ysum, status);
    worker.join()`,
		Buffers: []cudahostemu.Buffer{
			{Name: "slot", CType: "unsigned long long", Data: make([]byte, C2CDoorbellSlotBytes)},
			{Name: "x", CType: "float", Data: f32Bytes(x)},
			{Name: "w", CType: "float", Data: f32Bytes(w)},
			{Name: "y", CType: "float", Data: cudahostemu.Poison(4 * rounds * rows)},
			{Name: "lat", CType: "unsigned long long", Data: make([]byte, 8*rounds)},
			{Name: "ysum", CType: "float", Data: cudahostemu.Poison(4 * rounds)},
			{Name: "status", CType: "unsigned int", Data: cudahostemu.Poison(4)},
			{Name: "rc", CType: "int", Data: cudahostemu.Poison(4)},
		},
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if st := bytesU32(out["status"])[0]; st != 0 {
		t.Fatalf("kernel status %d, want 0", st)
	}
	if rc := bytesI32(out["rc"])[0]; rc != 0 {
		t.Fatalf("worker returned %d, want 0", rc)
	}
	var slot C2CDoorbellSlot
	if err := slot.UnmarshalBinary(out["slot"]); err != nil {
		t.Fatal(err)
	}
	if slot.SeqNo != rounds || slot.CompletionSeq != rounds || slot.TaskKind != TaskColdMoESingleToken || slot.NumTokens != 1 {
		t.Fatalf("final slot %+v", slot)
	}
	y, ysum := bytesF32(out["y"]), bytesF32(out["ysum"])
	for r := range rounds {
		var sum float64
		for i := range rows {
			var want float32
			for j := range cols {
				want += w[i*cols+j] * x[r*cols+j]
			}
			if !(math.Abs(float64(y[r*rows+i]-want)) <= 1e-5) {
				t.Fatalf("round %d row %d: %v, want %v", r, i, y[r*rows+i], want)
			}
			sum += float64(want)
		}
		if !(math.Abs(float64(ysum[r])-sum) <= 1e-4) {
			t.Fatalf("round %d: kernel read sum %v after the handoff, want %v", r, ysum[r], sum)
		}
	}
}

// TestC2CDoorbellNative runs the ping-pong between a CPU thread and a
// running GPU kernel through mapped host memory and reports the median
// round trip; the spec's acceptance bound is 1.5 us.
func TestC2CDoorbellNative(t *testing.T) {
	nvcc := requireNative(t)
	const rounds, rows, cols = 1000, 16, 64
	k := C2CDoorbellKernel{Name: "c2c_doorbell", Rounds: rounds, Rows: rows, Cols: cols}
	src, err := k.Source()
	if err != nil {
		t.Fatal(err)
	}
	src += `
#include <cmath>
#include <cstdio>
#include <thread>
int main() {
    unsigned long long *slot, *lat; float *x, *w, *y, *ysum; unsigned int* status;
    if (cudaHostAlloc((void**)&slot, 64, cudaHostAllocMapped) != cudaSuccess) return 2;
    cudaHostAlloc((void**)&x, sizeof(float) * EG_C2C_ROUNDS * EG_C2C_COLS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&y, sizeof(float) * EG_C2C_ROUNDS * EG_C2C_ROWS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&lat, sizeof(unsigned long long) * EG_C2C_ROUNDS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&ysum, sizeof(float) * EG_C2C_ROUNDS, cudaHostAllocMapped);
    cudaHostAlloc((void**)&status, sizeof(unsigned int), cudaHostAllocMapped);
    w = new float[EG_C2C_ROWS * EG_C2C_COLS];
    for (int i = 0; i < 64 / 8; ++i) slot[i] = 0;
    for (int i = 0; i < EG_C2C_ROUNDS * EG_C2C_COLS; ++i) x[i] = (float)((i * 7) % 13) - 6.0f;
    for (int i = 0; i < EG_C2C_ROWS * EG_C2C_COLS; ++i) w[i] = (float)((i * 5) % 11) * 0.125f;
    int rc = -1;
    std::thread worker([&] { rc = eg_c2c_worker(slot, w); });
    c2c_doorbell<<<1, 32>>>(slot, x, y, lat, ysum, status);
    cudaError_t err = cudaDeviceSynchronize();
    worker.join();
    if (err != cudaSuccess) { printf("cuda %s\n", cudaGetErrorString(err)); return 3; }
    printf("status %u\nrc %d\n", *status, rc);
    printf("lat"); for (int i = 0; i < EG_C2C_ROUNDS; ++i) printf(" %llu", lat[i]); printf("\n");
    double bad = 0;
    for (int r = 0; r < EG_C2C_ROUNDS; ++r) {
        double sum = 0;
        for (int i = 0; i < EG_C2C_ROWS; ++i) {
            float want = 0;
            for (int j = 0; j < EG_C2C_COLS; ++j) want += w[i * EG_C2C_COLS + j] * x[r * EG_C2C_COLS + j];
            bad += !(fabs(y[r * EG_C2C_ROWS + i] - want) <= 1e-4);
            sum += want;
        }
        bad += !(fabs(ysum[r] - sum) <= 1e-3 * (1 + fabs(sum)));
    }
    printf("bad %g\n", bad);
    return 0;
}
`
	out := nativeRun(t, nvcc, src)
	if st := parseFields(t, out, "status")[0]; st != 0 {
		t.Fatalf("kernel timed out waiting for the CPU (status %v)\n%s", st, out)
	}
	if rc := parseFields(t, out, "rc")[0]; rc != 0 {
		t.Fatalf("worker failed (rc %v)", rc)
	}
	if bad := parseFields(t, out, "bad")[0]; bad != 0 {
		t.Fatalf("%v wrong outputs", bad)
	}
	lat := parseFields(t, out, "lat")
	slices.Sort(lat)
	med, p99 := lat[len(lat)/2], lat[len(lat)*99/100]
	t.Logf("round trip over %d rounds: median %.0f ns, p99 %.0f ns", len(lat), med, p99)
	if med > 1500 {
		t.Errorf("median round trip %.0f ns exceeds the 1.5 us acceptance bound", med)
	}
}

func ExampleDecideColdMoEPlacement() {
	cfg, err := C2CPreset("vr200")
	if err != nil {
		panic(err)
	}
	for _, m := range []int{1, 64} {
		d, err := DecideColdMoEPlacement(cfg, m)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%d tokens: cpu=%v saved=%.1fMB\n", m, d.RunOnArmInPlace, d.BandwidthSavedMB)
	}
	// Output:
	// 1 tokens: cpu=true saved=36.0MB
	// 64 tokens: cpu=false saved=0.0MB
}

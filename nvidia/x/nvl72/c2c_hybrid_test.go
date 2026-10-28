package nvl72

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
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
// of that round's input, delivered through acquire/release handoffs only,
// with the completion word in the slot's line or, split, on its own line.
func TestC2CDoorbellKernelHostEmu(t *testing.T) {
	for _, split := range []bool{false, true} {
		t.Run(fmt.Sprintf("split=%v", split), func(t *testing.T) {
			testC2CDoorbellKernelHostEmu(t, split)
		})
	}
}

func testC2CDoorbellKernelHostEmu(t *testing.T, split bool) {
	const rounds, rows, cols = 8, 6, 10
	k := C2CDoorbellKernel{Name: "c2c_doorbell", Rounds: rounds, Rows: rows, Cols: cols, SplitCompletion: split}
	slotBytes := C2CDoorbellSlotBytes
	if split {
		slotBytes = C2CSplitSlotBytes
	}
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
			{Name: "slot", CType: "unsigned long long", Data: make([]byte, slotBytes)},
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
	if err := slot.UnmarshalBinary(out["slot"][:C2CDoorbellSlotBytes]); err != nil {
		t.Fatal(err)
	}
	done := slot.CompletionSeq
	if split {
		if slot.CompletionSeq != 0 {
			t.Fatalf("split completion also written in the slot line: %d", slot.CompletionSeq)
		}
		done = binary.LittleEndian.Uint64(out["slot"][C2CSplitCompletionOffset:])
	}
	if slot.SeqNo != rounds || done != rounds || slot.TaskKind != TaskColdMoESingleToken || slot.NumTokens != 1 {
		t.Fatalf("final slot %+v, completion %d", slot, done)
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

// c2cDoorbellBoundEnv, if set, is the median round trip in nanoseconds the
// baseline doorbell must not exceed, on any device.
const c2cDoorbellBoundEnv = "CUDA_NVL72_DOORBELL_BOUND_NS"

// c2cDoorbellBound returns the median round-trip bound in nanoseconds to
// enforce on device, and whether to enforce one. The spec's 1.5 us target
// is for GB200, GB300 and VR200-class C2C; elsewhere (GB10 included) the
// test only logs, unless c2cDoorbellBoundEnv sets a bound.
func c2cDoorbellBound(device string, getenv func(string) string) (float64, bool, error) {
	if s := getenv(c2cDoorbellBoundEnv); s != "" {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || !(v > 0) {
			return 0, false, fmt.Errorf("%s=%q: want a positive number of nanoseconds", c2cDoorbellBoundEnv, s)
		}
		return v, true, nil
	}
	d := strings.ToUpper(device)
	for _, class := range []string{"GB200", "GB300", "VR200", "RUBIN"} {
		if strings.Contains(d, class) {
			return 1500, true, nil
		}
	}
	return 0, false, nil
}

func TestC2CDoorbellBound(t *testing.T) {
	tests := []struct {
		device, env string
		want        float64
		ok, err     bool
	}{
		{device: "NVIDIA GB10"},
		{device: "NVIDIA B200"},
		{device: "NVIDIA GH200 480GB"},
		{device: "NVIDIA GB200", want: 1500, ok: true},
		{device: "NVIDIA GB300", want: 1500, ok: true},
		{device: "NVIDIA Vera Rubin VR200", want: 1500, ok: true},
		{device: "NVIDIA GB10", env: "5000", want: 5000, ok: true},
		{device: "NVIDIA GB200", env: "900", want: 900, ok: true},
		{device: "NVIDIA GB10", env: "0", err: true},
		{device: "NVIDIA GB10", env: "fast", err: true},
	}
	for _, tt := range tests {
		getenv := func(k string) string {
			if k == c2cDoorbellBoundEnv {
				return tt.env
			}
			return ""
		}
		got, ok, err := c2cDoorbellBound(tt.device, getenv)
		if (err != nil) != tt.err || ok != tt.ok || got != tt.want {
			t.Errorf("c2cDoorbellBound(%q, %q) = %v, %v, %v; want %v, %v, err %v", tt.device, tt.env, got, ok, err, tt.want, tt.ok, tt.err)
		}
	}
}

// c2cWorkerCPU returns the CPU to pin the doorbell worker to: the value of
// CUDA_NVL72_DOORBELL_CPU, else the highest-numbered CPU with the highest
// cpuinfo_max_freq (a performance core on a big.LITTLE part such as GB10),
// else -1.
func c2cWorkerCPU(t *testing.T) int {
	if s := os.Getenv("CUDA_NVL72_DOORBELL_CPU"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			t.Fatalf("CUDA_NVL72_DOORBELL_CPU=%q: want a CPU number", s)
		}
		return n
	}
	files, _ := filepath.Glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq")
	cpu, best := -1, int64(-1)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		freq, err1 := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		n, err2 := strconv.Atoi(strings.TrimPrefix(filepath.Base(filepath.Dir(filepath.Dir(f))), "cpu"))
		if err1 != nil || err2 != nil {
			continue
		}
		if freq > best || freq == best && n > cpu {
			cpu, best = n, freq
		}
	}
	return cpu
}

// c2cDoorbellMain is the host program for TestC2CDoorbellNative. Usage:
// prog mapped|pageable CPU. With "pageable" every buffer the kernel and
// worker share comes from aligned_alloc, which the GPU reaches through
// ATS/HMM; with "mapped" it comes from cudaHostAllocMapped. CPU >= 0 pins
// the worker thread to that CPU and makes the launching thread block in
// cudaDeviceSynchronize instead of spinning. Before the ping-pong, a probe
// kernel times EG_C2C_ROUNDS dependent ld.acquire.sys loads of the idle
// slot and prints the mean.
const c2cDoorbellMain = `
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
// eg_c2c_probe chains the loads through the slot's zero SeqNo, so each
// load's address waits for the previous load's value.
__global__ void eg_c2c_probe(const unsigned long long* p, unsigned long long* out) {
    const unsigned long long t0 = eg_c2c_globaltimer();
    for (int i = 0; i < EG_C2C_ROUNDS; ++i) p += eg_c2c_ld_acquire_sys_u64(p);
    out[0] = eg_c2c_globaltimer() - t0;
    out[1] = (unsigned long long)p;
}
static bool eg_pageable;
template <class T> static T* eg_alloc(size_t n) {
    void* p = nullptr;
    const size_t bytes = (n * sizeof(T) + 4095) & ~size_t(4095);
    if (eg_pageable) p = aligned_alloc(4096, bytes);
    else if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped) != cudaSuccess) p = nullptr;
    if (p == nullptr) { printf("alloc failed\n"); exit(2); }
    memset(p, 0, bytes);
    return (T*)p;
}
static void eg_print(const char* key, const unsigned long long* v) {
    printf("%s", key); for (int i = 0; i < EG_C2C_ROUNDS; ++i) printf(" %llu", v[i]); printf("\n");
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    eg_pageable = strcmp(argv[1], "pageable") == 0;
    const int cpu = atoi(argv[2]);
    if (cpu >= 0) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;
    int pma = 0;
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, 0);
    printf("device %s\npageable_access %d\n", prop.name, pma);
    if (eg_pageable && !pma) { printf("skip\n"); return 0; }
    unsigned long long* slot = eg_alloc<unsigned long long>(256 / 8);
    unsigned long long* lat = eg_alloc<unsigned long long>(EG_C2C_ROUNDS);
    unsigned long long* probe = eg_alloc<unsigned long long>(2);
    float* x = eg_alloc<float>(EG_C2C_ROUNDS * EG_C2C_COLS);
    float* y = eg_alloc<float>(EG_C2C_ROUNDS * EG_C2C_ROWS);
    float* ysum = eg_alloc<float>(EG_C2C_ROUNDS);
    unsigned int* status = eg_alloc<unsigned int>(1);
    float* w = new float[EG_C2C_ROWS * EG_C2C_COLS];
    for (int i = 0; i < EG_C2C_ROUNDS * EG_C2C_COLS; ++i) x[i] = (float)((i * 7) % 13) - 6.0f;
    for (int i = 0; i < EG_C2C_ROWS * EG_C2C_COLS; ++i) w[i] = (float)((i * 5) % 11) * 0.125f;
    eg_c2c_probe<<<1, 1>>>(slot, probe);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("cuda %s\n", cudaGetErrorString(err)); return 3; }
    int rc = -1, wcpu = -1;
    std::thread worker([&] {
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) { rc = -2; return; }
        }
        wcpu = sched_getcpu();
        rc = eg_c2c_worker(slot, w);
    });
    c2c_doorbell<<<1, 32>>>(slot, x, y, lat, ysum, status);
    err = cudaDeviceSynchronize();
    worker.join();
    if (err != cudaSuccess) { printf("cuda %s\n", cudaGetErrorString(err)); return 3; }
    printf("status %u\nrc %d\nworker_cpu %d\n", *status, rc, wcpu);
    eg_print("lat", lat);
    printf("probe %g\n", (double)probe[0] / EG_C2C_ROUNDS);
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

// TestC2CDoorbellNative runs the ping-pong between a CPU thread and a
// running GPU kernel and logs the median and p99 round trip of each
// variant. Only the baseline ("mapped": the default kernel, mapped host
// memory, unpinned spinning worker) is held to a bound, and only where
// c2cDoorbellBound gives one. The other variants separate the hardware
// from the implementation: pageable memory through ATS/HMM, a pinned
// worker on a performance core, __nanosleep polling, the completion word on
// its own cache line, and an empty 1x1 task that leaves only the doorbell.
//
// GB10 measurements are in the [C2CDoorbellKernel] doc comment.
func TestC2CDoorbellNative(t *testing.T) {
	nvcc := requireNative(t)
	const rounds = 1000
	gemv := C2CDoorbellKernel{Rows: 16, Cols: 64}
	empty := C2CDoorbellKernel{Rows: 1, Cols: 1}
	with := func(k C2CDoorbellKernel, f func(*C2CDoorbellKernel)) C2CDoorbellKernel {
		f(&k)
		return k
	}
	split := func(k *C2CDoorbellKernel) { k.SplitCompletion = true }
	variants := []struct {
		name   string
		kernel C2CDoorbellKernel
		mem    string
		pin    bool
	}{
		{"mapped", gemv, "mapped", false},
		{"pageable", gemv, "pageable", false},
		{"pinned", gemv, "mapped", true},
		{"nanosleep", with(gemv, func(k *C2CDoorbellKernel) { k.PollSleepNs = 100 }), "mapped", true},
		{"split", with(gemv, split), "mapped", true},
		{"empty", empty, "mapped", true},
		{"empty_split", with(empty, split), "mapped", true},
		{"empty_split_pageable", with(empty, split), "pageable", true},
	}
	cpu := c2cWorkerCPU(t)
	// Build in the parent test, whose TempDir outlives every subtest.
	bins := make(map[C2CDoorbellKernel]string)
	for i := range variants {
		k := &variants[i].kernel
		k.Name, k.Rounds = "c2c_doorbell", rounds
		if _, ok := bins[*k]; ok {
			continue
		}
		src, err := k.Source()
		if err != nil {
			t.Fatal(err)
		}
		bins[*k] = nativeBuild(t, nvcc, src+c2cDoorbellMain)
	}
	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			k, bin := v.kernel, bins[v.kernel]
			pin := -1
			if v.pin {
				pin = cpu
			}
			out := nativeExec(t, bin, v.mem, strconv.Itoa(pin))
			if strings.Contains(out, "\nskip\n") {
				t.Skipf("device has no pageable memory access\n%s", out)
			}
			if st := parseFields(t, out, "status")[0]; st != 0 {
				t.Fatalf("kernel timed out waiting for the CPU (status %v)\n%s", st, out)
			}
			if rc := parseFields(t, out, "rc")[0]; rc != 0 {
				t.Fatalf("worker failed (rc %v)", rc)
			}
			if bad := parseFields(t, out, "bad")[0]; bad != 0 {
				t.Fatalf("%v wrong outputs", bad)
			}
			var device string
			for line := range strings.Lines(out) {
				if s, ok := strings.CutPrefix(line, "device "); ok {
					device = strings.TrimSpace(s)
				}
			}
			lat := parseFields(t, out, "lat")
			slices.Sort(lat)
			med, p99 := lat[len(lat)/2], lat[len(lat)*99/100]
			t.Logf("%s on %s: %s memory, worker cpu %v (asked %d), poll sleep %d ns, split %v, %dx%d task: round trip median %.0f ns, p99 %.0f ns; idle dependent ld.acquire.sys %.0f ns",
				v.name, device, v.mem, parseFields(t, out, "worker_cpu")[0], pin, k.PollSleepNs, k.SplitCompletion, k.Rows, k.Cols, med, p99, parseFields(t, out, "probe")[0])
			if v.name != "mapped" {
				return
			}
			bound, ok, err := c2cDoorbellBound(device, os.Getenv)
			if err != nil {
				t.Fatal(err)
			}
			if ok && med > bound {
				t.Errorf("median round trip %.0f ns exceeds the %.0f ns bound for %s", med, bound, device)
			}
		})
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

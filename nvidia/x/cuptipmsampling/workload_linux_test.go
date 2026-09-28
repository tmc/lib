//go:build linux && cupti_physical

package cuptipmsampling_test

import (
	"crypto/sha256"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestPhysicalWorkload measures device-wide intervals, not per-kernel metrics.
// The build tag explicitly requests hardware qualification; failures never skip.
func TestPhysicalWorkload(t *testing.T) {
	d, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	const source = `extern "C" __global__ void pm_work(unsigned *out) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x, x=i;
 for(unsigned j=0;j<100000;j++) x=x*1664525u+1013904223u;
 out[i]=x;
 }`
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(d), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	cubin := cm.CUBIN()
	if len(cubin) == 0 {
		t.Fatal("missing native artifact")
	}
	if parent := os.Getenv("CUPTI_QUALIFICATION_DIR"); parent != "" {
		dir, err := os.MkdirTemp(parent, "pm-workload-")
		if err != nil {
			t.Fatal(err)
		}
		flags, err := nvrtcx.CompileFlags(nvrtcx.WithDeviceArch(d), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		for name, data := range map[string][]byte{"kernel.cu": []byte(source), "kernel.cubin": cubin, "kernel.ptx": []byte(cm.PTX()), "flags.txt": []byte(strings.Join(flags, "\n") + "\n")} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("retained source, PTX, loaded CUBIN and compiler flags: %s", dir)
	}

	mod, err := cm.LoadModule(d)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mod.Close(); err != nil {
			t.Error(err)
		}
	})
	kernel, err := mod.Function("pm_work")
	if err != nil {
		t.Fatal(err)
	}
	out, err := cudamem.Alloc[uint32](d, 256)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	})
	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		t.Fatal(err)
	}
	const metric = "sm__cycles_elapsed.avg"
	host, config := pmConfiguration(t, chip, metric)
	sess, err := cuptipmsampling.NewSession(cuptipmsampling.Config{ConfigImage: config, MetricNames: []string{metric}, MaxSamples: 65536, SamplingInterval: 100000, TriggerMode: cuptipmsampling.TriggerModeTimeInterval, AppendMode: cuptipmsampling.AppendModeKeepOldest})
	if sess != nil {
		t.Cleanup(func() {
			if err := sess.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Start(); err != nil {
		t.Fatal(err)
	}
	start, err := cupti.GetTimestamp()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if err := cudalaunch.OwnedKernel(kernel).Block(256, 1, 1).Launch(out); err != nil {
			t.Fatal(err)
		}
	}
	values := make([]uint32, 256)
	if err := out.CopyToHost(values); err != nil {
		t.Fatal(err)
	}
	end, err := cupti.GetTimestamp()
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Stop(); err != nil {
		t.Fatal(err)
	}
	for i, v := range values {
		x := uint32(i)
		for j := 0; j < 100000; j++ {
			x = x*1664525 + 1013904223
		}
		if v != x {
			t.Fatalf("output[%d]=%d want %d", i, v, x)
		}
	}
	decoded, err := sess.Decode()
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Overflow || decoded.StopReason != cuptipmsampling.DecodeStopReasonEndOfRecords || decoded.Info.NumCompletedSamples == 0 {
		t.Fatalf("incomplete capture: %+v", decoded)
	}
	image := sess.CounterData()
	var overlap uint
	var total float64
	var previous uint64
	for i := uint(0); i < decoded.Info.NumCompletedSamples; i++ {
		span, err := sess.SampleInfo(i)
		if err != nil {
			t.Fatal(err)
		}
		if span.StartTimestamp == 0 || span.EndTimestamp <= span.StartTimestamp || span.StartTimestamp < previous {
			t.Fatalf("invalid interval %+v previous end=%d", span, previous)
		}
		previous = span.EndTimestamp
		v := pmEvaluate(t, host, image, i, metric)
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			t.Fatalf("invalid cycles %g", v)
		}
		if span.StartTimestamp < end && span.EndTimestamp > start {
			overlap++
			total += v
		}
	}
	if overlap == 0 || total <= 0 {
		t.Fatalf("no nonempty measurements overlap workload: intervals=%d cycles=%g", overlap, total)
	}
	t.Logf("chip=%s source_sha256=%x cubin_sha256=%x launches=20 samples=%d overlap=%d cycles=%g window=[%d,%d] overflow=false; device-wide intervals overlapping submission-through-copy window, NOT per-kernel attribution", chip, sha256.Sum256([]byte(source)), sha256.Sum256(cubin), decoded.Info.NumCompletedSamples, overlap, total, start, end)
}

// pmConfiguration deliberately uses a PM host. A range-profiler plan is not a
// substitute: profilerType controls native scheduling and image evaluation.
func pmConfiguration(t *testing.T, chip, metric string) (uintptr, []byte) {
	t.Helper()
	v, err := cuptipmsampling.CheckVersion()
	if err != nil {
		t.Fatal(err)
	}
	availability, err := cuptipmsampling.CounterAvailability(0)
	if err != nil {
		t.Fatal(err)
	}
	chipBytes := append([]byte(chip), 0)
	var pin runtime.Pinner
	pin.Pin(&chipBytes[0])
	pin.Pin(&availability[0])
	t.Cleanup(pin.Unpin)
	init := cupti.ProfilerHostInitializeParams{ProfilerType: cupti.ProfilerTypePmSampling, PChipName: uintptr(unsafe.Pointer(&chipBytes[0])), PCounterAvailabilityImage: uintptr(unsafe.Pointer(&availability[0]))}
	init.StructSize = uint(unsafe.Offsetof(init.PHostObject) + unsafe.Sizeof(init.PHostObject))
	switch v {
	case 130200, 130202, 130301, 130400:
		init.StructSize = uint(unsafe.Sizeof(init))
	}
	if err := cupti.ProfilerHostInitialize(&init); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		p := cupti.ProfilerHostDeinitializeParams{PHostObject: init.PHostObject}
		p.StructSize = uint(unsafe.Sizeof(p))
		if err := cupti.ProfilerHostDeinitialize(&p); err != nil {
			t.Error(err)
		}
	})
	name := append([]byte(metric), 0)
	names := []uintptr{uintptr(unsafe.Pointer(&name[0]))}
	pin.Pin(&name[0])
	pin.Pin(&names[0])
	add := cupti.ProfilerHostConfigAddMetricsParams{PHostObject: init.PHostObject, PpMetricNames: uintptr(unsafe.Pointer(&names[0])), NumMetrics: 1}
	add.StructSize = uint(unsafe.Sizeof(add))
	if err := cupti.ProfilerHostConfigAddMetrics(&add); err != nil {
		t.Fatal(err)
	}
	size := cupti.ProfilerHostGetConfigImageSizeParams{PHostObject: init.PHostObject}
	size.StructSize = uint(unsafe.Sizeof(size))
	if err := cupti.ProfilerHostGetConfigImageSize(&size); err != nil {
		t.Fatal(err)
	}
	if size.ConfigImageSize == 0 || size.ConfigImageSize > 16<<20 {
		t.Fatalf("invalid config size %d", size.ConfigImageSize)
	}
	config := make([]byte, size.ConfigImageSize)
	pin.Pin(&config[0])
	get := cupti.ProfilerHostGetConfigImageParams{PHostObject: init.PHostObject, ConfigImageSize: uint(len(config)), PConfigImage: uintptr(unsafe.Pointer(&config[0]))}
	get.StructSize = uint(unsafe.Sizeof(get))
	if err := cupti.ProfilerHostGetConfigImage(&get); err != nil {
		t.Fatal(err)
	}
	passes := cupti.ProfilerHostGetNumOfPassesParams{ConfigImageSize: uint(len(config)), PConfigImage: get.PConfigImage}
	passes.StructSize = uint(unsafe.Sizeof(passes))
	if err := cupti.ProfilerHostGetNumOfPasses(&passes); err != nil {
		t.Fatal(err)
	}
	if passes.NumOfPasses != 1 {
		t.Fatalf("PM requires one pass, got %d", passes.NumOfPasses)
	}
	return init.PHostObject, config
}

func pmEvaluate(t *testing.T, host uintptr, image []byte, index uint, metric string) float64 {
	t.Helper()
	name := append([]byte(metric), 0)
	names := []uintptr{uintptr(unsafe.Pointer(&name[0]))}
	value := math.NaN()
	var pin runtime.Pinner
	pin.Pin(&name[0])
	pin.Pin(&names[0])
	pin.Pin(&value)
	pin.Pin(&image[0])
	defer pin.Unpin()
	p := cupti.ProfilerHostEvaluateToGpuValuesParams{PHostObject: host, PCounterDataImage: uintptr(unsafe.Pointer(&image[0])), CounterDataImageSize: uint(len(image)), RangeIndex: index, PpMetricNames: uintptr(unsafe.Pointer(&names[0])), NumMetrics: 1, PMetricValues: uintptr(unsafe.Pointer(&value))}
	p.StructSize = uint(unsafe.Sizeof(p))
	if err := cupti.ProfilerHostEvaluateToGpuValues(&p); err != nil {
		t.Fatal(err)
	}
	return value
}

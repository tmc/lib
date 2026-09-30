package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

type probeControl struct {
	WallNS  int64  `json:"wall_ns"`
	Output  uint32 `json:"output"`
	Correct bool   `json:"correct"`
}
type probeNativeResult struct {
	Kind         string `json:"kind"`
	Error        string `json:"error"`
	DisableError string `json:"disable_error"`
	Code         *int32 `json:"native_result_code"`
	DisableCode  *int32 `json:"disable_native_result_code"`
}
type probeReport struct {
	Build               buildStamp                    `json:"build"`
	DriverVersion       int                           `json:"driver_version"`
	DriverVersionError  string                        `json:"driver_version_error,omitempty"`
	ReplayMode          string                        `json:"replay_mode"`
	CounterOverhead     string                        `json:"counter_overhead"`
	Version             int                           `json:"version"`
	Device              int                           `json:"device"`
	Name                string                        `json:"name"`
	Chip                string                        `json:"chip"`
	Architecture        string                        `json:"host_architecture"`
	Allocation          string                        `json:"allocation"`
	CUPTI               uint32                        `json:"cupti_version"`
	Attributes          map[string]int                `json:"attributes"`
	AttributeErrors     map[string]string             `json:"attribute_errors"`
	UMPolicy            string                        `json:"um_policy"`
	UMResults           []probeNativeResult           `json:"um_results"`
	Residency           memoryEvidence                `json:"residency"`
	Marker              string                        `json:"marker"`
	Controls            []probeControl                `json:"controls"`
	Definitions         []cuptirange.MetricProperties `json:"definitions"`
	Values              [][]float64                   `json:"values"`
	Quality             *cuptirange.CollectionQuality `json:"quality"`
	ConfigurationPasses uint                          `json:"configuration_passes"`
	ReplayOutputs       []uint32                      `json:"replay_outputs"`
	Error               string                        `json:"error"`
	Limits              []string                      `json:"limits"`
}

// runProbeCommand collects a controlled native receipt under caller-owned GPU
// isolation. UM configuration runs separately from range profiling.
func runProbeCommand(args []string) error {
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: cudaprof probe [-device N] [-metrics names] [-um] [-managed]")
		fs.PrintDefaults()
	}
	device := fs.Int("device", 0, "CUDA device ordinal")
	metrics := fs.String("metrics", "", "comma-separated native metric names")
	um := fs.Bool("um", false, "probe native UM counter configuration only (separate process)")
	managed := fs.Bool("managed", false, "use managed allocation for the controlled kernel; does not observe residency")
	passes := fs.Uint("max-passes", 16, "maximum user replay passes")
	if e := fs.Parse(args); e != nil {
		if errors.Is(e, flag.ErrHelp) {
			return nil
		}
		return e
	}
	if fs.NArg() != 0 || *device < 0 || *passes == 0 || *passes > 1024 || (*um && (*metrics != "" || *managed)) {
		return fmt.Errorf("probe: invalid arguments")
	}
	if *metrics != "" {
		if _, err := probeMetricNames(*metrics); err != nil {
			return err
		}
	}
	r := probeReport{Build: currentBuild(), ReplayMode: "uncollected", CounterOverhead: "unmeasured", Version: 1, Device: *device, Architecture: runtime.GOARCH, Attributes: map[string]int{}, AttributeErrors: map[string]string{}, Residency: memoryEvidence{"unavailable", "allocation support and configuration success do not observe page residency"}, Limits: []string{"controlled kernel is not a qualified application shape", "profiling changes execution; wall time includes reset, launch and synchronization", "UM configuration success is not collection, migration, or page-fault evidence"}}
	if runtime.GOARCH == "arm64" {
		r.UMPolicy = "NVIDIA release notes say UM profiling unsupported on ARM; native configuration probe remains separate: https://docs.nvidia.com/cupti/release-notes/release-notes.html"
	}
	err := collectProbe(&r, *metrics, *passes, *um, *managed)
	if err != nil {
		r.Error = err.Error()
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if e := enc.Encode(r); e != nil {
		return e
	}
	return err
}

func collectProbe(r *probeReport, metrics string, passes uint, um, managed bool) (err error) {
	dev, err := cuda.NewDevice(r.Device)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, dev.Close()) }()
	r.DriverVersion, err = driver.DriverGetVersion()
	if err != nil {
		r.DriverVersionError = err.Error()
	}
	r.Name, err = dev.Name()
	if err != nil {
		return err
	}
	for _, a := range []driver.DeviceAttribute{driver.DevAttrManagedMemory, driver.DevAttrConcurrentManagedAccess, driver.DevAttrPageableMemoryAccess, driver.DevAttrPageableMemoryAccessUsesHostPageTables, driver.DevAttrDirectManagedMemAccessFromHost} {
		v, e := dev.Attribute(a)
		if e != nil {
			r.AttributeErrors[a.String()] = e.Error()
		} else {
			r.Attributes[a.String()] = v
		}
	}
	r.CUPTI, err = cupti.GetVersion()
	if err != nil {
		return err
	}
	if um {
		return dev.WithContext(func() error {
			var failures []error
			for _, kind := range []cupti.ActivityUnifiedMemoryCounterKind{cupti.ActivityUnifiedMemoryCounterKindBytesTransferHtod, cupti.ActivityUnifiedMemoryCounterKindBytesTransferDtoh, cupti.ActivityUnifiedMemoryCounterKindCpuPageFaultCount, cupti.ActivityUnifiedMemoryCounterKindGpuPageFault} {
				c := cupti.ActivityUnifiedMemoryCounterConfig{Scope: cupti.ActivityUnifiedMemoryCounterScopeProcessSingleDevice, Kind: kind, DeviceId: uint32(r.Device), Enable: 1}
				result := probeNativeResult{Kind: kind.String()}
				e := cupti.ActivityConfigureUnifiedMemoryCounter(&c, 1)
				if e != nil {
					result.Error = e.Error()
					result.Code = probeResultCode(e)
					failures = append(failures, e)
				} else {
					result.Code = probeResultCode(nil)
					c.Enable = 0
					e = cupti.ActivityConfigureUnifiedMemoryCounter(&c, 1)
					result.DisableCode = probeResultCode(e)
					if e != nil {
						result.DisableError = e.Error()
						failures = append(failures, e)
					}
				}
				r.UMResults = append(r.UMResults, result)
			}
			return errors.Join(failures...)
		})
	}
	cm, err := nvrtcx.Compile(`extern "C" __global__ void cudaprof_probe(unsigned* out) { out[0] = 7; }`, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		return err
	}
	module, err := cm.Load(dev)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, dev.WithContext(module.Close)) }()
	var fn driver.Function
	if err = dev.WithContext(func() error { var e error; fn, e = module.Function("cudaprof_probe"); return e }); err != nil {
		return err
	}
	var out cudamem.Slice[uint32]
	r.Allocation = "device"
	if managed {
		r.Allocation = "managed"
		out, err = cudamem.AllocManaged[uint32](dev, 1)
	} else {
		out, err = cudamem.Alloc[uint32](dev, 1)
	}
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, out.Free()) }()
	token, layer, cache := 0, 0, 0
	dtype := "u32"
	disabled := false
	r.Marker, err = marker.Encode(marker.Metadata{Phase: "decode", Token: &token, Layer: &layer, Operation: "probe_write", DType: &dtype, CacheLength: &cache, Lookahead: &disabled, Materialization: &disabled})
	if err != nil {
		return err
	}
	run := func() error {
		if err := out.CopyFromHost([]uint32{0}); err != nil {
			return err
		}
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		nvtx.RangePush(r.Marker)
		defer nvtx.RangePop()
		return cudalaunch.Kernel(fn).Launch(out)
	}
	read := func() (uint32, error) { var values [1]uint32; e := out.CopyToHost(values[:]); return values[0], e }
	control := func() error {
		start := time.Now()
		e := dev.WithContext(func() error {
			if e := run(); e != nil {
				return e
			}
			return driver.CtxSynchronize()
		})
		elapsed := time.Since(start).Nanoseconds()
		if e != nil {
			return e
		}
		v, e := read()
		r.Controls = append(r.Controls, probeControl{elapsed, v, v == 7})
		if e != nil {
			return e
		}
		if v != 7 {
			return fmt.Errorf("probe: output %d, want 7", v)
		}
		return nil
	}
	if err = control(); err != nil {
		return err
	}
	if metrics == "" {
		return control()
	}
	names, err := probeMetricNames(metrics)
	if err != nil {
		return err
	}
	var available []byte
	if err = dev.WithContext(func() error {
		var e error
		r.Chip, e = cuptisass.DeviceChipName(uint32(r.Device))
		if e != nil {
			return e
		}
		available, e = cuptirange.CounterAvailability(dev, 16<<20)
		return e
	}); err != nil {
		return err
	}
	plan, err := cuptirange.NewPlan(r.Chip, available, names, 16<<20)
	if plan != nil {
		defer func() { err = errors.Join(err, plan.Close()) }()
	}
	if err != nil {
		return err
	}
	_, r.ConfigurationPasses = plan.Configuration()
	for _, n := range names {
		p, e := plan.Properties(n, 4096)
		if e != nil {
			return e
		}
		r.Definitions = append(r.Definitions, p)
	}
	r.ReplayMode = "user_replay"
	capture, collectErr := plan.Capture(dev, cuptirange.Limits{MaxRanges: 4, MaxRangeTreeNodes: 8, MaxCounterBytes: 16 << 20, MaxPasses: passes}, func(pass uint) error {
		if e := run(); e != nil {
			return e
		}
		if e := driver.CtxSynchronize(); e != nil {
			return e
		}
		v, e := read()
		r.ReplayOutputs = append(r.ReplayOutputs, v)
		if e != nil {
			return e
		}
		if v != 7 {
			return fmt.Errorf("probe: replay %d output %d, want 7", pass, v)
		}
		return nil
	})
	if capture != nil {
		r.Quality = &capture.Quality
		defer func() { err = errors.Join(err, capture.Close()) }()
		if cleanup := capture.Close(); cleanup != nil {
			return errors.Join(collectErr, fmt.Errorf("probe: cleanup before unprofiled control: %w", cleanup))
		}
	}
	afterErr := control()
	if collectErr != nil || afterErr != nil {
		return errors.Join(collectErr, afterErr)
	}
	for i := uint(0); i < capture.Ranges; i++ {
		v, e := plan.Evaluate(capture.Image, i)
		if e != nil {
			return e
		}
		r.Values = append(r.Values, v)
	}
	return nil
}

// A missing library symbol has no native result code.
func probeResultCode(err error) *int32 {
	var code int32
	if err != nil {
		var result cupti.Result
		if !errors.As(err, &result) {
			return nil
		}
		code = int32(result)
	}
	return &code
}

func probeMetricNames(metrics string) ([]string, error) {
	names := strings.Split(metrics, ",")
	if len(names) > 32 {
		return nil, fmt.Errorf("probe: metric count exceeds 32")
	}
	seen := make(map[string]bool)
	for _, name := range names {
		if name == "" || strings.TrimSpace(name) != name || strings.ContainsRune(name, 0) || seen[name] {
			return nil, fmt.Errorf("probe: invalid or duplicate metric name")
		}
		seen[name] = true
	}
	return names, nil
}

//go:build linux

// capture-plan-native prepares a real compiled workload and cooperates with
// cudaprof's opt-in capture plan. Preparation also requires the GPU lock.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type packet struct {
	Artifact    *cudakernel.Artifact       `json:"artifact"`
	Workload    cudacache.WorkloadIdentity `json:"workload"`
	CompilerKey string                     `json:"compiler_key"`
	Source      string                     `json:"source"`
	PTX         string                     `json:"ptx"`
	SASS        string                     `json:"sass"`
	Domain      string                     `json:"domain"`
	Chip        string                     `json:"chip"`
	RangeImage  []byte                     `json:"range_image"`
	PM          *cuptipmsampling.Config    `json:"pm,omitempty"`
}

func main() {
	prepare := flag.Bool("prepare", false, "prepare a plan on the locked GPU")
	dir := flag.String("dir", "", "existing task-scoped output directory")
	pmPath := flag.String("pm-config", "", "actual independently prepared PM Config JSON; no range-image substitution")
	pmMetric := flag.String("pm-metric", "", "actual PM metric name; build a distinct PM host configuration")
	flag.Parse()
	if err := run(*prepare, *dir, *pmPath, *pmMetric); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(prepare bool, dir, pmPath, pmMetric string) (err error) {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return fmt.Errorf("inherited gpu-lock required: %w", err)
	}
	if dir == "" {
		return errors.New("-dir is required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, dev.Close()) }()
	if prepare {
		return preparePlan(dev, dir, pmPath, pmMetric)
	}
	return collect(dev, dir)
}

func observe(dev *cuda.Device, domain string) (cudaprof.CaptureEnvironment, error) {
	name, err := dev.Name()
	if err != nil {
		return cudaprof.CaptureEnvironment{}, err
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		return cudaprof.CaptureEnvironment{}, err
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		return cudaprof.CaptureEnvironment{}, err
	}
	cv, err := cupti.GetVersion()
	if err != nil {
		return cudaprof.CaptureEnvironment{}, err
	}
	// Current clock observations are receipts, not assertions that clocks are locked.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	clocks, err := boundedCommand(ctx, 4096, "nvidia-smi", "-i", "0", "--query-gpu=clocks.current.sm,clocks.current.memory", "--format=csv,noheader,nounits")
	if err != nil {
		return cudaprof.CaptureEnvironment{}, fmt.Errorf("observe clocks: %w", err)
	}
	return cudaprof.CaptureEnvironment{Device: name, Driver: strconv.Itoa(version), CUDA: fmt.Sprintf("NVRTC %d.%d", major, minor), CUPTI: strconv.FormatUint(uint64(cv), 10), ClockMode: "observed MHz: " + strings.TrimSpace(string(clocks)), CorrelationDomain: domain}, nil
}

func preparePlan(dev *cuda.Device, dir, pmPath, pmMetric string) (err error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	target := fmt.Sprintf("sm_%d%d", major, minor)
	cm, err := nvrtcx.Compile(source, nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithNoCache(), nvrtcx.WithLineInfo())
	if err != nil {
		return err
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "ptr", ByteWidth: 8, Alignment: 8, IsPointer: true}, cudakernel.Param{Name: "trips", Type: "u32", ByteWidth: 4, Alignment: 4})
	if err != nil {
		return err
	}
	artifact, err := cm.ToArtifact("capture_work", schema, target)
	if err != nil {
		return err
	}
	if artifact.Kind != cudakernel.KindCUBIN {
		return errors.New("compiler produced no CUBIN; exact loaded-image fixture unavailable")
	}
	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		return err
	}
	names := []string{"sm__inst_executed.sum"}
	rp, err := cuptirange.NewPlan(chip, nil, names, 4<<20)
	if rp != nil {
		defer func() { err = errors.Join(err, rp.Close()) }()
	}
	if err != nil {
		return err
	}
	image, passes := rp.Configuration()
	metrics, sassErr := cuptisass.SupportedMetrics(chip)
	if len(artifact.Raw) > 4<<20 || len(cm.PTX()) > 4<<20 {
		return errors.New("compiled code packet exceeds byte limits")
	}
	if sassErr == nil && len(metrics) == 0 {
		sassErr = errors.New("no actual SASS metric available")
	}
	nonce := make([]byte, 16)
	if _, err := rand.Read(nonce); err != nil {
		return err
	}
	p := packet{Artifact: artifact, Workload: cudacache.WorkloadIdentity{ProgramID: "capture-plan-native/lcg-v1", ShapeDims: []int{elements, int(trips)}, NumericalMode: "uint32-wrap"}, CompilerKey: cm.Key(), Source: source, PTX: cm.PTX(), Domain: "fixture/" + hex.EncodeToString(nonce), Chip: chip, RangeImage: image}
	if pmPath != "" {
		var cfg cuptipmsampling.Config
		if err := readJSON(pmPath, &cfg); err != nil {
			return err
		}
		p.PM = &cfg
	}
	if pmMetric != "" {
		if pmPath != "" {
			return errors.New("choose -pm-config or -pm-metric")
		}
		cfg, err := actualPMConfig(chip, pmMetric)
		if err != nil {
			if e := os.WriteFile(filepath.Join(dir, "pm-unavailable.txt"), []byte(err.Error()+"\n"), 0600); e != nil {
				return e
			}
		} else {
			p.PM = &cfg
		}
	}

	cubinPath := filepath.Join(dir, "kernel.cubin")
	if err := os.WriteFile(cubinPath, artifact.Raw, 0600); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	sass, err := boundedCommand(ctx, 4<<20, "nvdisasm", cubinPath)
	if err != nil {
		return fmt.Errorf("disassemble actual CUBIN: %w", err)
	}
	if len(sass) > 4<<20 {
		return errors.New("SASS receipt exceeds limit")
	}
	p.SASS = string(sass)
	id, err := cudaprof.BindEvidence(artifact, p.Workload, p.CompilerKey)
	if err != nil {
		return err
	}
	env, err := observe(dev, p.Domain)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(image)
	plan := cudaprof.CapturePlan{Version: 1, Identity: id, Environment: env, Budget: cudaprof.CaptureBudget{MaxRecords: 1024, MaxBytes: byteLimit, MaxPasses: int(passes)*2 + 4, MaxWallNanos: int64(60 * time.Second)}, Passes: []cudaprof.CapturePass{
		{ID: "activity", Replay: "launch-reference", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity}},
		{ID: "range", Replay: "range-replay", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureRange}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureRange: {Metrics: names, ConfigDigest: hex.EncodeToString(digest[:])}}},
	}}
	if sassErr == nil {
		plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "sass", Replay: "sass-replay", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureSASS}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureSASS: {SASSMetrics: []cudaprof.CaptureSASSMetric{{MetricID: metrics[0].ID, Granularity: uint8(cuptisass.GranularityGPU)}}}}})
	} else {
		pass := plan.Passes[1]
		pass.ID = "range-second"
		pass.Replay = "range-replay-second"
		plan.Passes = append(plan.Passes, pass)
		if err := os.WriteFile(filepath.Join(dir, "sass-unavailable.txt"), []byte(sassErr.Error()+"\n"), 0600); err != nil {
			return err
		}
	}
	if p.PM != nil {
		d := sha256.Sum256(p.PM.ConfigImage)
		cfg := cudaprof.CaptureProviderConfig{Metrics: p.PM.MetricNames, ConfigDigest: hex.EncodeToString(d[:]), SamplingInterval: p.PM.SamplingInterval, TriggerMode: int32(p.PM.TriggerMode), AppendMode: int32(p.PM.AppendMode)}
		plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "pm", Replay: "pm-replay", Providers: []cudaprof.CaptureProvider{cudaprof.CapturePM}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CapturePM: cfg}})
	}
	if err := plan.Validate(); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(dir, "packet.json"), p); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "kernel.cu"), []byte(source), 0600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "kernel.ptx"), []byte(p.PTX), 0600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "kernel.sass"), sass, 0600); err != nil {
		return err
	}
	return writeJSON(filepath.Join(dir, "plan.json"), plan)
}

func collect(dev *cuda.Device, dir string) (err error) {
	var p packet
	if err := readJSON(filepath.Join(dir, "packet.json"), &p); err != nil {
		return err
	}
	f, err := os.Open(os.Getenv("CUDAPROF_CAPTURE_PLAN"))
	if err != nil {
		return err
	}
	plan, err := cudaprof.ReadCapturePlan(f, 1<<20)
	err = errors.Join(err, f.Close())
	if err != nil {
		return err
	}
	var pass *cudaprof.CapturePass
	for i := range plan.Passes {
		if plan.Passes[i].ID == os.Getenv("CUDAPROF_CAPTURE_PASS") {
			pass = &plan.Passes[i]
		}
	}
	if pass == nil {
		return errors.New("requested pass not found")
	}
	if p.Artifact == nil {
		return errors.New("fixture artifact unavailable")
	}
	if err := verifyWorkload(p.Workload, p.Source); err != nil {
		return err
	}
	if err := p.Artifact.Validate(); err != nil {
		return err
	}
	id, err := cudaprof.BindEvidence(p.Artifact, p.Workload, p.CompilerKey)
	if err != nil {
		return err
	}
	env, err := observe(dev, p.Domain)
	if err != nil {
		return err
	}
	primary, err := dev.PrimaryContext()
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, primary.Release()) }()
	var contextID uint32
	err = dev.WithContext(func() error { var e error; contextID, e = cupti.GetContextId(primary.Raw()); return e })
	if err != nil {
		return err
	}
	generation, err := processStart()
	if err != nil {
		return err
	}
	var raw driver.Module
	err = dev.WithContext(func() error { var e error; raw, e = driver.ModuleLoadData(p.Artifact.Raw); return e })
	if err != nil {
		return err
	}
	module, err := cuda.OwnModule(dev, raw)
	if err != nil {
		return errors.Join(err, dev.WithContext(raw.Close))
	}
	defer func() { err = errors.Join(err, module.Close()) }()
	kernel, err := module.Function("capture_work")
	if err != nil {
		return err
	}
	out, err := cudamem.Alloc[uint32](dev, elements)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, out.Free()) }()
	limits, err := runtimeLimits(plan)
	if err != nil {
		return err
	}
	deadline := time.Unix(0, limits.deadline)
	launch := func() error {
		return dev.WithContext(func() error {
			if time.Now().After(deadline) {
				return errors.New("fixture wall budget exhausted")
			}
			if err := cudalaunch.Kernel(kernel.Raw()).Grid(1, 1, 1).Block(elements, 1, 1).Launch(out, trips); err != nil {
				return err
			}
			got := make([]uint32, elements)
			if err := out.CopyToHost(got); err != nil {
				return err
			}
			for i, v := range got {
				want := uint32(i + 1)
				for j := uint32(0); j < trips; j++ {
					want = want*1664525 + 1013904223
				}
				if v != want {
					return fmt.Errorf("workload output[%d]=%d want%d", i, v, want)
				}
			}
			return nil
		})
	}
	if len(pass.Providers) == 1 && pass.Providers[0] == cudaprof.CaptureActivity {
		return launch()
	}
	start, err := cupti.GetTimestamp()
	if err != nil {
		return err
	}
	r := cudaprof.CaptureProviderResult{PassID: pass.ID, Replay: pass.Replay, Identity: id, Environment: env, Process: uint64(os.Getpid()), Generation: generation, Context: strconv.FormatUint(uint64(contextID), 10), Start: start, Status: "unavailable", Reason: "provider not collected", EvidenceClass: "unread"}
	provider := pass.Providers[0]
	var collectionErr error
	switch provider {
	case cudaprof.CaptureRange:
		metrics := pass.Configs[provider].Metrics
		rp, e := cuptirange.NewPlan(p.Chip, nil, metrics, min(4<<20, limits.bytes))
		collectionErr = e
		var capture *cuptirange.Capture
		if e == nil {
			image, _ := rp.Configuration()
			if !bytes.Equal(image, p.RangeImage) {
				collectionErr = errors.New("actual range configuration differs from preparation")
			} else {
				capture, collectionErr = rp.CaptureReplay(dev, cuptirange.Limits{MaxRanges: uint(limits.records), MaxRangeTreeNodes: 1024, MaxCounterBytes: min(4<<20, limits.bytes), MaxPasses: uint(limits.passes)}, cuptirange.CaptureOptions{Mode: cuptirange.Regions, MaxDepth: 2}, cuptirange.Replay{Synchronize: driver.CtxSynchronize, Run: func(_ uint, s *cuptirange.Scope) error { return s.Region("work", launch) }})
			}
		}
		cleanup := error(nil)
		if capture != nil {
			cleanup = errors.Join(cleanup, capture.Close())
		}
		if rp != nil {
			cleanup = errors.Join(cleanup, rp.Close())
		}
		collectionErr = errors.Join(collectionErr, cleanup)
		r.CleanupComplete = cleanup == nil
		r.Stop, err = cupti.GetTimestamp()
		if err != nil {
			return err
		}
		r, err = cudaprof.RangeCaptureProviderResult(r, capture, p.RangeImage, metrics, collectionErr)
	case cudaprof.CaptureSASS:
		cfg := cuptisass.Config{DeviceIndex: 0}
		for _, m := range pass.Configs[provider].SASSMetrics {
			cfg.Metrics = append(cfg.Metrics, cuptisass.MetricConfig{MetricID: m.MetricID, Granularity: cuptisass.OutputGranularity(m.Granularity)})
		}
		var records []cuptisass.Record
		var dropped *uint64
		var session *cuptisass.Session
		collectionErr = dev.WithContext(func() error {
			var e error
			session, e = cuptisass.NewSession(uintptr(primary.Raw()), cfg)
			if e == nil {
				e = session.Enable()
			}
			if e == nil {
				e = launch()
			}
			if session != nil {
				if e == nil {
					records, e = session.FlushBounded(cuptisass.FlushLimits{MaxRecords: limits.records, MaxInstances: uint64(limits.bytes), MaxBytes: uint64(limits.bytes), MaxFunctionNameBytes: 4096})
				}
				n, de := session.Disable()
				if de == nil {
					d := uint64(n)
					dropped = &d
				}
				e = errors.Join(e, de, session.Close())
				r.CleanupComplete = e == nil
			}
			return e
		})
		r.Stop, err = cupti.GetTimestamp()
		if err != nil {
			return err
		}
		r, err = cudaprof.SASSCaptureProviderResult(r, cfg, records, dropped, collectionErr)
	case cudaprof.CapturePM:
		if p.PM == nil {
			return errors.New("actual PM configuration unavailable")
		}
		cfg, err := boundedPMConfig(*p.PM, limits)
		if err != nil {
			return err
		}
		var decoded cuptipmsampling.DecodeResult
		var samples []cuptipmsampling.SampleInfo
		var image []byte
		session, e := cuptipmsampling.NewSession(cfg)
		collectionErr = e
		if e == nil {
			collectionErr = session.Start()
			if collectionErr == nil {
				collectionErr = launch()
			}
			collectionErr = errors.Join(collectionErr, session.Stop())
			if collectionErr == nil {
				decoded, collectionErr = session.Decode()
			}
			if collectionErr == nil && (uint64(decoded.Info.NumCompletedSamples) > limits.records || decoded.Info.NumCompletedSamples > uint(cfg.MaxSamples) || decoded.Info.NumCompletedSamples > uint(limits.bytes/32)) {
				collectionErr = errors.New("PM decoded sample budget exceeded")
			}
			if collectionErr == nil {
				for i := uint(0); i < decoded.Info.NumCompletedSamples; i++ {
					s, e := session.SampleInfo(i)
					if e != nil {
						collectionErr = e
						break
					}
					samples = append(samples, s)
				}
			}
			image = session.CounterData()
		}
		cleanup := error(nil)
		if session != nil {
			cleanup = session.Close()
		}
		collectionErr = errors.Join(collectionErr, cleanup)
		r.CleanupComplete = cleanup == nil
		r.Stop, err = cupti.GetTimestamp()
		if err != nil {
			return err
		}
		r, err = cudaprof.PMCaptureProviderResult(r, decoded, samples, image, cfg, collectionErr)
	default:
		return fmt.Errorf("unsupported cooperative provider %q", provider)
	}
	if err != nil {
		return err
	}
	publication := cudaprof.CapturePublication{Version: 1, Result: r, Artifact: p.Artifact, Workload: p.Workload, CompilerKey: p.CompilerKey}
	transport := os.Getenv("CUDAPROF_TRANSPORT")
	if transport == "" {
		return errors.New("actual controller transport directory unavailable")
	}
	var encoded bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&encoded, plan, publication); err != nil {
		return err
	}
	if err := writeCodePublication(transport, provider, p, raw, encoded.Bytes(), limits.bytes); err != nil {
		return err
	}

	fmt.Printf("provider=%s status=%s reason=%s process=%d generation=%d context=%s\n", provider, r.Status, r.Reason, r.Process, r.Generation, r.Context)
	return nil
}

func processStart() (uint64, error) {
	b, e := os.ReadFile("/proc/self/stat")
	if e != nil {
		return 0, e
	}
	end := strings.LastIndexByte(string(b), ')')
	if end < 0 {
		return 0, errors.New("malformed process stat")
	}
	fields := strings.Fields(string(b[end+1:]))
	if len(fields) < 20 {
		return 0, errors.New("short process stat")
	}
	return strconv.ParseUint(fields[19], 10, 64)
}
func readJSON(path string, v any) error {
	f, e := os.Open(path)
	if e != nil {
		return e
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		return e
	}
	if !info.Mode().IsRegular() || info.Size() > byteLimit {
		return errors.New("fixture input is not a bounded regular file")
	}
	d := json.NewDecoder(io.LimitReader(f, byteLimit+1))
	d.DisallowUnknownFields()
	if e := d.Decode(v); e != nil {
		return e
	}
	var extra any
	if e := d.Decode(&extra); e != io.EOF {
		return errors.New("trailing fixture JSON")
	}
	return nil
}
func writeJSON(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	if len(b) > byteLimit {
		return errors.New("fixture packet byte limit exceeded")
	}
	return os.WriteFile(path, b, 0600)
}

func writeCodePublication(dir string, provider cudaprof.CaptureProvider, p packet, loaded driver.Module, publication []byte, limit int) error {
	code := cudacapture.CodeFiles{Source: "kernel.cu", PTX: "kernel.ptx", CUBIN: "kernel.cubin", SASS: "kernel.sass"}
	capture := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "code_receipt", Unit: "bytes"}}, Comments: []string{"code-only producer receipt; no GPU timing samples"}}, Code: code, Modules: []cudacapture.Module{{ID: fmt.Sprintf("producer-loaded-%x", uint64(loaded)), Generation: 1, Code: code}}, Files: map[string][]byte{"kernel.cu": []byte(p.Source), "kernel.ptx": []byte(p.PTX), "kernel.cubin": p.Artifact.Raw, "kernel.sass": []byte(p.SASS)}}
	var codeBytes int
	for _, data := range capture.Files {
		if len(data) > limit/4-codeBytes {
			return errors.New("code packet exceeds aggregate budget")
		}
		codeBytes += len(data)
	}
	if len(publication) > limit/4 {
		return errors.New("publication exceeds aggregate encoded budget")
	}
	remaining := limit/4 - len(publication)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			continue
		}
		if info.Size() > int64(remaining) {
			return errors.New("existing transport exceeds aggregate budget")
		}
		remaining -= int(info.Size())
	}
	output := &limitedOutput{remaining: remaining}
	if err := cudacapture.WriteLimits(output, capture, cudacapture.Limits{Files: 4, FileBytes: int64(remaining), TotalBytes: int64(remaining), InputBytes: int64(remaining)}); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "capture-provider-"+string(provider)+".pprof"), output.Bytes(), 0600); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "capture-provider-"+string(provider)+".json"), publication, 0600)
}

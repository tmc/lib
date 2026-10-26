//go:build linux && native

package nccltest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
	"github.com/tmc/lib/nvidia/x/ncclras"
	"golang.org/x/sys/unix"
)

const (
	collectorCostMaxPluginBytes  = 128 << 20
	collectorCostMaxCaptureBytes = 4 << 20
)

type nativeCollectorCostConfig struct {
	Source    string
	Devices   []int32
	RASAddr   string
	Plugin    collectorPluginPin
	OutputDir string
	OffDir    string
	OnDir     string
}

type nativeCollectorArmReceipt struct {
	Mode                collectorArmMode   `json:"mode"`
	PID                 int                `json:"pid"`
	Hostname            string             `json:"hostname"`
	BootID              string             `json:"boot_id"`
	ProcessStartTicks   uint64             `json:"process_start_ticks"`
	CollectorEnabled    bool               `json:"collector_enabled"`
	LiveCommunicator    bool               `json:"live_communicator"`
	ObservedFiles       int                `json:"observed_files"`
	RASCommunicatorHash string             `json:"ras_communicator_hash"`
	Plugin              collectorPluginPin `json:"plugin"`
	OutputDir           string             `json:"output_dir"`
	DumpDir             string             `json:"dump_dir"`
	Env                 map[string]string  `json:"env"`
	Result              Result             `json:"result"`
	CyclesCompleted     int                `json:"cycles_completed"`
	Timing              collectorArmTiming `json:"timing"`
}

// TestNativeCollectorCostComparison runs a fresh child process per arm (off
// then on) under an inherited GPU lock and compares collector-disabled versus
// pinned-Inspector-enabled execution using the shared collector cost contract.
func TestNativeCollectorCostComparison(t *testing.T) {
	if os.Getenv("NCCLCOST_NATIVE") != "1" {
		t.Skip("collector cost native qualification pending: explicit NCCLCOST_NATIVE=1 admission required")
	}
	cfg, err := preflightNativeCollectorCost(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", cfg.RASAddr)
	if err != nil {
		t.Fatalf("loopback RAS address %s unavailable: %v", cfg.RASAddr, err)
	}
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	plan, err := defaultNativeCollectorCostWorkload(cfg.Devices)
	if err != nil {
		t.Fatal(err)
	}
	scope := nccljoin.Scope{
		Capture: "native-collector-cost",
		Job:     cfg.Source,
		Epoch:   "on-arm",
	}
	var onRASCommHash string
	spec := collectorCostComparisonSpec{
		ParentPID:    os.Getpid(),
		Case:         plan.Case,
		Options:      plan.Options,
		Cycles:       plan.Cycles,
		OffOutputDir: cfg.OffDir,
		OnOutputDir:  cfg.OnDir,
		RASAddr:      cfg.RASAddr,
		Plugin:       cfg.Plugin,
		Expected:     plan.expectedCapture(scope, "", [2]int{}, [2]string{}),
	}
	spec.ResolveOnExpected = func(onExec collectorArmExecution) (collectorExpectedCapture, error) {
		if err := ncclinspector.ValidateCommunicatorHash(onRASCommHash); err != nil {
			return collectorExpectedCapture{}, fmt.Errorf("live RAS communicator hash invalid: %w", err)
		}
		return plan.expectedCapture(
			spec.Expected.Scope,
			onRASCommHash,
			[2]int{onExec.PID, onExec.PID},
			[2]string{onExec.Hostname, onExec.Hostname},
		), nil
	}

	receipt, err := runCollectorCostComparison(ctx, spec, func(armCtx context.Context, armCfg collectorArmConfig) (collectorArmExecution, error) {
		execState, rasHash, runErr := runNativeCollectorArmSubprocess(armCtx, armCfg)
		if armCfg.Mode == collectorArmOn {
			onRASCommHash = rasHash
		}
		return execState, runErr
	})
	if err != nil {
		t.Fatal(err)
	}
	summary := struct {
		Source           string             `json:"source"`
		CommunicatorHash string             `json:"communicator_hash"`
		OffPID           int                `json:"off_pid"`
		OnPID            int                `json:"on_pid"`
		OffTiming        collectorArmTiming `json:"off_timing"`
		OnTiming         collectorArmTiming `json:"on_timing"`
		LossKnown        bool               `json:"loss_known"`
		LossStatus       string             `json:"loss_status"`
		RankCounts       [2]uint64          `json:"rank_counts"`
		Plugin           collectorPluginPin `json:"plugin"`
	}{
		Source:           cfg.Source,
		CommunicatorHash: receipt.Capture.CommunicatorHash,
		OffPID:           receipt.Off.PID,
		OnPID:            receipt.On.PID,
		OffTiming:        receipt.Off.Timing,
		OnTiming:         receipt.On.Timing,
		LossKnown:        receipt.Capture.LossKnown,
		LossStatus:       receipt.Capture.LossStatus,
		RankCounts:       receipt.Capture.RankCounts,
		Plugin:           cfg.Plugin,
	}
	if err := stallWrite(filepath.Join(cfg.OutputDir, "comparison-summary.json"), summary); err != nil {
		t.Fatal(err)
	}
}

func preflightNativeCollectorCost(getenv func(string) string) (nativeCollectorCostConfig, error) {
	if driver.Loaded() || nccl.Loaded() || nvml.Loaded() {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: vendor library loaded before inherited GPU lock admission")
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return nativeCollectorCostConfig{}, fmt.Errorf("collector cost native lock: %w", err)
	}
	source := getenv("NCCLCOST_SOURCE")
	if raw, err := hex.DecodeString(source); err != nil || len(raw) != 20 || strings.ToLower(source) != source {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: canonical 40-hex NCCLCOST_SOURCE revision required")
	}
	parts := strings.Split(getenv("NCCLCOST_DEVICES"), ",")
	if len(parts) != 2 {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: NCCLCOST_DEVICES must name two distinct device ordinals")
	}
	devices := make([]int32, 2)
	for i, part := range parts {
		ordinal, err := strconv.ParseInt(part, 10, 32)
		if err != nil || ordinal < 0 {
			return nativeCollectorCostConfig{}, errors.New("collector cost native: invalid device ordinal")
		}
		devices[i] = int32(ordinal)
	}
	if devices[0] == devices[1] {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: distinct device ordinals required")
	}
	address := getenv("NCCLCOST_RAS_ADDR")
	host, portStr, err := net.SplitHostPort(address)
	port, portErr := strconv.Atoi(portStr)
	if err != nil || host != "127.0.0.1" || portErr != nil || port < 1024 || port > 65535 {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: explicit 127.0.0.1 loopback NCCLCOST_RAS_ADDR required")
	}
	plugin, err := verifyPinnedInspectorPlugin(getenv("NCCLCOST_INSPECTOR_PLUGIN"), getenv("NCCLCOST_INSPECTOR_SHA256"))
	if err != nil {
		return nativeCollectorCostConfig{}, err
	}
	out := getenv("NCCLCOST_OUTPUT")
	if !filepath.IsAbs(out) || filepath.Clean(out) != out {
		return nativeCollectorCostConfig{}, errors.New("collector cost native: clean absolute fresh NCCLCOST_OUTPUT directory required")
	}
	if err := os.Mkdir(out, 0700); err != nil {
		return nativeCollectorCostConfig{}, err
	}
	offDir := filepath.Join(out, "off")
	onDir := filepath.Join(out, "on")
	if err := os.Mkdir(offDir, 0700); err != nil {
		return nativeCollectorCostConfig{}, err
	}
	if err := os.Mkdir(onDir, 0700); err != nil {
		return nativeCollectorCostConfig{}, err
	}
	return nativeCollectorCostConfig{
		Source:    source,
		Devices:   devices,
		RASAddr:   address,
		Plugin:    plugin,
		OutputDir: out,
		OffDir:    offDir,
		OnDir:     onDir,
	}, nil
}

func verifyPinnedInspectorPlugin(path, wantHex string) (collectorPluginPin, error) {
	pin := collectorPluginPin{Path: path, SHA256: wantHex}
	if err := validateCollectorPluginPin(pin); err != nil {
		return collectorPluginPin{}, err
	}
	wantBytes, err := hex.DecodeString(wantHex)
	if err != nil || len(wantBytes) != sha256.Size {
		return collectorPluginPin{}, errors.New("collector cost native: invalid Inspector plugin SHA-256")
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > collectorCostMaxPluginBytes {
		return collectorPluginPin{}, errors.New("collector cost native: bounded regular Inspector plugin file required")
	}
	f, err := openBoundedCollectorFile(path, collectorCostMaxPluginBytes)
	if err != nil {
		return collectorPluginPin{}, err
	}
	opened, statErr := f.Stat()
	if statErr != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		f.Close()
		return collectorPluginPin{}, errors.New("collector cost native: Inspector plugin descriptor changed")
	}
	hash := sha256.New()
	n, readErr := io.Copy(hash, io.LimitReader(f, collectorCostMaxPluginBytes+1))
	closeErr := f.Close()
	if readErr != nil || closeErr != nil || n != info.Size() || !bytes.Equal(hash.Sum(nil), wantBytes) {
		return collectorPluginPin{}, errors.New("collector cost native: Inspector plugin SHA-256 digest mismatch")
	}
	return pin, nil
}

func openBoundedCollectorFile(path string, maxBytes int64) (*os.File, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NONBLOCK|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > maxBytes {
		f.Close()
		return nil, fmt.Errorf("collector cost native: bounded regular file required: %s", path)
	}
	return f, nil
}

func runNativeCollectorArmSubprocess(parentCtx context.Context, armCfg collectorArmConfig) (execState collectorArmExecution, rasCommHash string, err error) {
	ctx, cancel := context.WithTimeout(parentCtx, 20*time.Second)
	defer cancel()

	if armCfg.Mode == collectorArmOn {
		if err := os.Mkdir(armCfg.DumpDir, 0700); err != nil {
			return execState, "", err
		}
	}
	if err := stallWrite(filepath.Join(armCfg.OutputDir, "arm-config.json"), armCfg); err != nil {
		return execState, "", err
	}
	stdoutFile, err := os.OpenFile(filepath.Join(armCfg.OutputDir, "arm.stdout"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return execState, "", err
	}
	stderrFile, err := os.OpenFile(filepath.Join(armCfg.OutputDir, "arm.stderr"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		stdoutFile.Close()
		return execState, "", err
	}
	outLog := &stallLog{w: stdoutFile, remaining: 1 << 20, cancel: cancel}
	errLog := &stallLog{w: stderrFile, remaining: 1 << 20, cancel: cancel}
	defer func() {
		err = errors.Join(err, outLog.Err(), errLog.Err(), stdoutFile.Close(), stderrFile.Close())
	}()

	cmd := exec.CommandContext(ctx, os.Args[0], "-test.v", "-test.run=^TestNativeCollectorCostArm$")
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if key == "NCCL_DEBUG_FILE" || key == "NCCL_DEBUG" || key == "NCCL_DEBUG_SUBSYS" ||
			key == "NCCL_PROFILER_PLUGIN" || strings.HasPrefix(key, "NCCL_INSPECTOR_") {
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	for k, v := range armCfg.Env {
		if v != "" {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}
	cmd.Env = append(cmd.Env,
		"NCCLCOST_NATIVE=1",
		"NCCLCOST_ARM_DIR="+armCfg.OutputDir,
		"NCCL_DEBUG=INFO",
		"NCCL_DEBUG_SUBSYS=INIT,ENV",
	)
	cmd.Stdout = outLog
	cmd.Stderr = errLog
	if err := cmd.Start(); err != nil {
		return execState, "", err
	}
	child := stallWatch(cmd, cancel)
	var childReceipt nativeCollectorArmReceipt
	waitErr := stallWaitChildren(ctx, filepath.Join(armCfg.OutputDir, "arm-receipt.json"), &childReceipt, []*stallChild{child})
	<-child.done
	if waitErr != nil || child.err != nil {
		return execState, "", errors.Join(waitErr, child.err)
	}
	if childReceipt.PID != cmd.Process.Pid {
		return execState, "", fmt.Errorf("collector cost native: child receipt PID %d != process PID %d", childReceipt.PID, cmd.Process.Pid)
	}
	execState = collectorArmExecution{
		Mode:              childReceipt.Mode,
		PID:               childReceipt.PID,
		Hostname:          childReceipt.Hostname,
		BootID:            childReceipt.BootID,
		ProcessStartTicks: childReceipt.ProcessStartTicks,
		CollectorEnabled:  childReceipt.CollectorEnabled,
		LiveCommunicator:  childReceipt.LiveCommunicator,
		ObservedFiles:     childReceipt.ObservedFiles,
		Plugin:            childReceipt.Plugin,
		OutputDir:         childReceipt.OutputDir,
		DumpDir:           childReceipt.DumpDir,
		Env:               childReceipt.Env,
		Result:            childReceipt.Result,
		CyclesCompleted:   childReceipt.CyclesCompleted,
		Timing:            childReceipt.Timing,
	}
	if armCfg.Mode == collectorArmOn {
		capturePath := filepath.Join(armCfg.OutputDir, "live-inspector-capture.jsonl")
		execState.ReadCapture = func() ([]byte, error) {
			f, err := openBoundedCollectorFile(capturePath, collectorCostMaxCaptureBytes)
			if err != nil {
				return nil, err
			}
			data, readErr := io.ReadAll(io.LimitReader(f, collectorCostMaxCaptureBytes+1))
			closeErr := f.Close()
			if readErr != nil || closeErr != nil {
				return nil, errors.Join(readErr, closeErr)
			}
			if len(data) > collectorCostMaxCaptureBytes {
				return nil, errors.New("collector cost native: retained capture exceeds byte bound")
			}
			return data, nil
		}
		execState.Sink = func(name string, data []byte) error {
			return os.WriteFile(filepath.Join(armCfg.OutputDir, name), data, 0600)
		}
	}
	return execState, childReceipt.RASCommunicatorHash, nil
}

// TestNativeCollectorCostArm is the internal per-arm child subprocess entry point.
func TestNativeCollectorCostArm(t *testing.T) {
	armDir := os.Getenv("NCCLCOST_ARM_DIR")
	if os.Getenv("NCCLCOST_NATIVE") != "1" || armDir == "" {
		t.Skip("internal native collector cost arm entry point")
	}
	if driver.Loaded() || nccl.Loaded() || nvml.Loaded() {
		t.Fatal("vendor library loaded before inherited GPU lock admission")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(armDir) || filepath.Clean(armDir) != armDir {
		t.Fatal("clean absolute NCCLCOST_ARM_DIR required")
	}
	var armCfg collectorArmConfig
	if err := stallRead(filepath.Join(armDir, "arm-config.json"), &armCfg); err != nil {
		t.Fatal(err)
	}
	if err := executeNativeCollectorArmChild(armCfg); err != nil {
		t.Fatal(err)
	}
}

func executeNativeCollectorArmChild(armCfg collectorArmConfig) error {
	plan, err := planNativeCollectorCostWorkload(armCfg.Case, armCfg.Options, armCfg.Cycles)
	if err != nil {
		return err
	}
	setupStart, err := stallNow()
	if err != nil {
		return err
	}
	for _, key := range []string{
		"NCCL_PROFILER_PLUGIN",
		"NCCL_INSPECTOR_ENABLE",
		"NCCL_INSPECTOR_DUMP_DIR",
		"NCCL_INSPECTOR_DUMP_THREAD_ENABLE",
		"NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS",
		"NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES",
		"NCCL_INSPECTOR_PROM_DUMP",
		"NCCL_INSPECTOR_OTEL_EXPORT",
		"NCCL_INSPECTOR_DUMP_VERBOSE",
	} {
		if err := os.Unsetenv(key); err != nil {
			return err
		}
	}
	if armCfg.Mode == collectorArmOn {
		if _, err := verifyPinnedInspectorPlugin(armCfg.Plugin.Path, armCfg.Plugin.SHA256); err != nil {
			return err
		}
		info, err := os.Lstat(armCfg.DumpDir)
		if err != nil || !info.IsDir() {
			return errors.New("collector cost native: on-arm dump directory missing")
		}
	}
	appliedEnv := make(map[string]string, len(armCfg.Env))
	for k, v := range armCfg.Env {
		appliedEnv[k] = v
		if v == "" {
			if err := os.Unsetenv(k); err != nil {
				return err
			}
			continue
		}
		if err := os.Setenv(k, v); err != nil {
			return err
		}
	}

	hostname, err := os.Hostname()
	if err != nil || strings.TrimSpace(hostname) == "" {
		return errors.New("collector cost native: non-empty hostname required")
	}
	bootIDBytes, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
	if err != nil {
		return err
	}
	startTicks, err := currentProcessStartTicks()
	if err != nil {
		return err
	}
	if err := driver.Init(0); err != nil {
		return err
	}
	count, err := driver.DeviceGetCount()
	if err != nil || count < 2 {
		return fmt.Errorf("collector cost native: two CUDA devices required (count=%d err=%v)", count, err)
	}
	for _, ordinal := range armCfg.Options.Devices {
		if int(ordinal) >= count {
			return fmt.Errorf("collector cost native: device ordinal %d unavailable", ordinal)
		}
	}
	setupEnd, err := stallNow()
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 18*time.Second)
	defer cancel()
	client := ncclras.Client{Address: armCfg.RASAddr, Timeout: 5 * time.Second}

	wallStart := setupEnd
	var flushDuration time.Duration
	var lastResult Result
	var totalGPUTime time.Duration
	var commHash string
	var observedFiles int
	cycles := 0

	err = RunCycles(plan.Case, plan.Options, plan.Cycles, func(result Result) error {
		if result.Ranks != 2 || result.Comparisons <= 0 || result.Wrong != 0 {
			return fmt.Errorf("collector cost native numerical check failed: %+v", result)
		}
		status, err := client.Status(ctx)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(armCfg.OutputDir, fmt.Sprintf("ras-status-%d.json", cycles)), status.Raw, 0600); err != nil {
			return err
		}
		nextHash, err := plan.validateRASCycleStatus(status, cycles, result.NCCLVersion, os.Getpid(), commHash)
		if err != nil {
			return err
		}
		commHash = nextHash
		cycles++
		totalGPUTime += result.GPUTime
		lastResult = result
		lastResult.GPUTime = totalGPUTime

		if cycles == plan.Cycles {
			flushStart, err := stallNow()
			if err != nil {
				return err
			}
			if armCfg.Mode == collectorArmOff {
				entries, err := os.ReadDir(armCfg.OutputDir)
				if err != nil {
					return err
				}
				for _, entry := range entries {
					if strings.HasSuffix(entry.Name(), ".log") || strings.Contains(entry.Name(), "inspector") {
						observedFiles++
					}
				}
				if observedFiles != 0 {
					return errors.New("collector cost native: off arm produced unexpected Inspector output")
				}
			} else {
				dumpPath := filepath.Join(armCfg.DumpDir, fmt.Sprintf("%s-pid%d.log", hostname, os.Getpid()))
				expected := plan.expectedCapture(
					nccljoin.Scope{
						Capture: "native-collector-cost",
						Job:     "live-child",
						Epoch:   "on-arm",
					},
					commHash,
					[2]int{os.Getpid(), os.Getpid()},
					[2]string{hostname, hostname},
				)
				captureCtx, stop := context.WithTimeout(ctx, 8*time.Second)
				defer stop()
				capture, err := pollLiveCollectorCapture(captureCtx, dumpPath, expected, armCfg.Plugin, 100*time.Millisecond)
				if err != nil {
					return err
				}
				observedFiles = 1
				if err := os.WriteFile(filepath.Join(armCfg.OutputDir, "live-inspector-capture.jsonl"), capture, 0600); err != nil {
					return err
				}
			}
			flushEnd, err := stallNow()
			if err != nil {
				return err
			}
			flushDuration = time.Duration(flushEnd - flushStart)
		}
		return nil
	})
	if err != nil {
		return err
	}
	wallEnd, err := stallNow()
	if err != nil {
		return err
	}
	receipt := nativeCollectorArmReceipt{
		Mode:                armCfg.Mode,
		PID:                 os.Getpid(),
		Hostname:            hostname,
		BootID:              strings.TrimSpace(string(bootIDBytes)),
		ProcessStartTicks:   startTicks,
		CollectorEnabled:    armCfg.Mode == collectorArmOn,
		LiveCommunicator:    armCfg.Mode == collectorArmOn,
		ObservedFiles:       observedFiles,
		RASCommunicatorHash: commHash,
		Plugin:              armCfg.Plugin,
		OutputDir:           armCfg.OutputDir,
		DumpDir:             armCfg.DumpDir,
		Env:                 appliedEnv,
		Result:              lastResult,
		CyclesCompleted:     cycles,
		Timing: collectorArmTiming{
			SetupDuration: time.Duration(setupEnd - setupStart),
			WallDuration:  time.Duration(wallEnd - wallStart),
			FlushDuration: flushDuration,
			GPUTime:       lastResult.GPUTime,
			RankTimes:     lastResult.RankTimes,
		},
	}
	return stallWrite(filepath.Join(armCfg.OutputDir, "arm-receipt.json"), receipt)
}

func pollLiveCollectorCapture(
	ctx context.Context,
	path string,
	expected collectorExpectedCapture,
	plugin collectorPluginPin,
	interval time.Duration,
) ([]byte, error) {
	return pollLiveCollectorCaptureReader(ctx, func() ([]byte, error) {
		f, err := openBoundedCollectorFile(path, collectorCostMaxCaptureBytes)
		if os.IsNotExist(err) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		b, readErr := io.ReadAll(io.LimitReader(f, collectorCostMaxCaptureBytes+1))
		closeErr := f.Close()
		if readErr != nil || closeErr != nil {
			return nil, errors.Join(readErr, closeErr)
		}
		return b, nil
	}, expected, plugin, interval)
}

func currentProcessStartTicks() (uint64, error) {
	stat, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return 0, err
	}
	closeParen := strings.LastIndexByte(string(stat), ')')
	if closeParen < 0 {
		return 0, errors.New("collector cost native: invalid /proc/self/stat")
	}
	fields := strings.Fields(string(stat[closeParen+1:]))
	if len(fields) < 20 {
		return 0, errors.New("collector cost native: missing start ticks in /proc/self/stat")
	}
	ticks, err := strconv.ParseUint(fields[19], 10, 64)
	if err != nil || ticks == 0 {
		return 0, errors.New("collector cost native: invalid process start ticks")
	}
	return ticks, nil
}

// TestNativeCollectorCostPreflightAndChildControls exercises CPU-only preflight
// and subprocess early-exit refusals on Linux native builds without touching GPUs.
func TestNativeCollectorCostPreflightAndChildControls(t *testing.T) {
	if os.Getenv("NCCLCOST_EXIT_CONTROL") == "1" {
		fmt.Fprintln(os.Stderr, "cpu-only collector arm preflight refusal")
		os.Exit(4)
	}
	if _, err := preflightNativeCollectorCost(func(string) string { return "" }); err == nil {
		t.Fatal("preflight succeeded without inherited GPU lock")
	}

	tmp := t.TempDir()
	pluginPath := filepath.Join(tmp, "libnccl-profiler-inspector.so")
	pluginContent := []byte("constructed-plugin-bytes")
	if err := os.WriteFile(pluginPath, pluginContent, 0600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(pluginContent)
	sumHex := hex.EncodeToString(sum[:])

	if _, err := verifyPinnedInspectorPlugin(pluginPath, sumHex); err != nil {
		t.Fatalf("verifyPinnedInspectorPlugin failed: %v", err)
	}
	if _, err := verifyPinnedInspectorPlugin(pluginPath, strings.Repeat("00", sha256.Size)); err == nil {
		t.Fatal("verifyPinnedInspectorPlugin accepted wrong SHA-256")
	}
	symlinkPath := filepath.Join(tmp, "symlink-plugin.so")
	if err := os.Symlink(pluginPath, symlinkPath); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyPinnedInspectorPlugin(symlinkPath, sumHex); err == nil {
		t.Fatal("verifyPinnedInspectorPlugin accepted symlinked plugin")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var raw bytes.Buffer
	log := &stallLog{w: &raw, remaining: 1 << 20, cancel: cancel}
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestNativeCollectorCostPreflightAndChildControls$")
	cmd.Env = append(os.Environ(), "NCCLCOST_EXIT_CONTROL=1")
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	child := stallWatch(cmd, cancel)
	var receipt nativeCollectorArmReceipt
	err := stallWaitChildren(ctx, filepath.Join(tmp, "missing-arm-receipt.json"), &receipt, []*stallChild{child})
	<-child.done
	if err == nil || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("child exit became timeout: %v", err)
	}
	if !strings.Contains(raw.String(), "cpu-only collector arm preflight refusal") || log.Err() != nil {
		t.Fatalf("lost child stderr diagnostic: %q (%v)", raw.String(), log.Err())
	}
}

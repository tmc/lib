package cudaregress

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestCommandReplayRunnerFixture(t *testing.T) {
	dir := t.TempDir()
	currentBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "runner.test")
	source, err := os.Open(currentBinary)
	if err != nil {
		t.Fatal(err)
	}
	dest, err := os.OpenFile(binary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0700)
	if err != nil {
		source.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dest, source)
	closeDestErr := dest.Close()
	closeSourceErr := source.Close()
	if err := errors.Join(copyErr, closeDestErr, closeSourceErr); err != nil {
		t.Fatalf("copy task-owned helper executable: %v", err)
	}
	if err := os.Chmod(binary, 0700); err != nil {
		t.Fatal(err)
	}
	digest, size, err := ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	plan, _, _, _ := historyFixture(t)
	plan.BinaryArtifacts = []ArtifactRef{{Name: "runner", Path: binary, Digest: digest, SizeBytes: size, Role: "binary"}}
	plan.PlanDigest, err = ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"command-replay-child-good"}
	spec := ExecutionSpec{
		Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "host_fixture",
		WorkingDirectory: dir, Environment: map[string]string{"MODE": "fixed"},
		TimeoutNanos: int64(10 * time.Second), MaxOutputBytes: 4096,
		Baseline:  CommandArm{Path: binary, SHA256: digest, Args: args},
		Candidate: CommandArm{Path: binary, SHA256: digest, Args: args},
		Inputs:    []CommandArtifact{}, Libraries: []CommandArtifact{},
	}
	spec.SpecDigest, err = ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	for _, constraint := range []string{"clock", "power", "mig"} {
		t.Run("unsupported-"+constraint, func(t *testing.T) {
			unsupported := *plan
			switch constraint {
			case "clock":
				unsupported.DeviceConfig.TargetClockMHz = 1200
			case "power":
				unsupported.DeviceConfig.PowerLimitWatts = 250
			case "mig":
				unsupported.DeviceConfig.MIGMode = "1g.10gb"
			}
			unsupported.PlanDigest, err = ComputePlanDigest(&unsupported)
			if err != nil {
				t.Fatal(err)
			}
			unsupportedSpec := spec
			unsupportedSpec.PlanDigest = unsupported.PlanDigest
			unsupportedSpec.Environment = map[string]string{"Clock": "untyped/free-form fixture"}
			unsupportedSpec.SpecDigest, err = ComputeExecutionSpecDigest(unsupportedSpec)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := NewCommandReplayRunner(&unsupported, unsupportedSpec); err == nil || !strings.Contains(err.Error(), "unsupported") {
				t.Fatalf("typed %s constraint did not refuse at public admission: %v", constraint, err)
			}
		})
	}
	runner, err := NewCommandReplayRunner(plan, spec)
	if err != nil {
		t.Fatal(err)
	}
	args[0] = "unexpected-mutation"
	plan.PlanID = "mutated"
	receipt, got, err := runner.RunArm(context.Background(), "candidate", 3)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Complete || !receipt.HasTiming(cudatrial.TimingWallElapsed) || receipt.WallNs != 17 || len(got) != 2 || got[1] != 2.5 {
		t.Fatalf("unexpected receipt or values: %+v %v", receipt, got)
	}
	zeroSpec := spec
	zeroSpec.Baseline.Args = []string{"command-replay-child-zero"}
	zeroSpec.Candidate.Args = []string{"command-replay-child-zero"}
	zeroSpec.SpecDigest, err = ComputeExecutionSpecDigest(zeroSpec)
	if err != nil {
		t.Fatal(err)
	}
	zeroRunner, err := NewCommandReplayRunner(runner.plan, zeroSpec)
	if err != nil {
		t.Fatal(err)
	}
	zeroReceipt, zeroValues, err := zeroRunner.RunArm(context.Background(), "baseline", 1)
	if err != nil || !zeroReceipt.HasTiming(cudatrial.TimingWallElapsed) || zeroReceipt.WallNs != 0 || len(zeroValues) != 2 || zeroValues[0] != 0 {
		t.Fatalf("explicit measured zero lost at runner API: receipt=%+v values=%v err=%v", zeroReceipt, zeroValues, err)
	}
	var wg sync.WaitGroup
	for round := 4; round < 8; round++ {
		wg.Add(1)
		go func(round int) {
			defer wg.Done()
			if _, _, err := runner.RunArm(context.Background(), "baseline", round); err != nil {
				t.Errorf("concurrent replay: %v", err)
			}
		}(round)
	}
	wg.Wait()
	attempts := runner.Attempts()
	if len(attempts) != 5 || attempts[0].Round != 3 || attempts[0].Result == nil || attempts[0].StdoutSHA256 == "" || attempts[0].StderrSHA256 == "" {
		t.Fatalf("missing attempt evidence: %+v", attempts)
	}
	got[0] = 99
	attempts[0].Result.OutputValues[0] = 98
	attempts[0].Result.Timing.Count = 97
	retained := runner.Attempts()[0].Result
	if retained.OutputValues[0] != 1 || retained.Timing.Count != 17 {
		t.Fatalf("public result mutations changed retained evidence: %+v", retained)
	}
}

func init() {
	if len(os.Args) == 0 || !strings.HasPrefix(os.Args[len(os.Args)-1], "command-replay-child-") {
		return
	}
	mode := strings.TrimPrefix(os.Args[len(os.Args)-1], "command-replay-child-")
	if mode == "wait" {
		time.Sleep(time.Second)
		os.Exit(0)
	}
	if mode == "overflow" {
		fmt.Fprint(os.Stdout, strings.Repeat("x", 8192))
		os.Exit(0)
	}
	if mode == "nonzero" {
		fmt.Fprint(os.Stdout, "partial")
		os.Exit(9)
	}
	if mode == "malformed" {
		fmt.Fprint(os.Stdout, "{")
		os.Exit(0)
	}
	if mode == "zero" {
		digest, err := OutputDigest([]float64{0, 2.5})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stdout, "{\"version\":1,\"output_values\":[0,2.5],\"output_sha256\":%q,\"timing\":{\"kind\":\"wall_elapsed\",\"status\":\"measured\",\"unit\":\"ns\",\"count\":0}}\n", digest)
		os.Exit(0)
	}
	digest, err := OutputDigest([]float64{1, 2.5})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	status := "measured"
	if mode == "unmeasured" {
		status = "unmeasured"
	}
	fmt.Fprintf(os.Stdout, "{\"version\":1,\"output_values\":[1,2.5],\"output_sha256\":%q,\"timing\":{\"kind\":\"wall_elapsed\",\"status\":%q,\"unit\":\"ns\",\"count\":17}}\n", digest, status)
	os.Exit(0)
}

func TestCommandReplayRunnerRetainsFailureAttempts(t *testing.T) {
	dir := t.TempDir()
	currentBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "runner.test")
	source, err := os.Open(currentBinary)
	if err != nil {
		t.Fatal(err)
	}
	dest, err := os.OpenFile(binary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0700)
	if err != nil {
		source.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dest, source)
	closeDestErr := dest.Close()
	closeSourceErr := source.Close()
	if err := errors.Join(copyErr, closeDestErr, closeSourceErr); err != nil {
		t.Fatal(err)
	}
	digest, size, err := ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	plan, _, _, _ := historyFixture(t)
	plan.BinaryArtifacts = []ArtifactRef{{Name: "runner", Path: binary, Digest: digest, SizeBytes: size}}
	plan.PlanDigest, err = ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		timeout   time.Duration
		maxOutput int
		wantExit  int
	}{
		{name: "malformed", maxOutput: 4096, wantExit: 0},
		{name: "nonzero", maxOutput: 4096, wantExit: 9},
		{name: "overflow", maxOutput: 128, wantExit: -2},
		{name: "timeout", timeout: 10 * time.Millisecond, maxOutput: 4096, wantExit: -1},
		{name: "unmeasured", maxOutput: 4096, wantExit: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			timeout := test.timeout
			if timeout == 0 {
				timeout = 10 * time.Second
			}
			args := []string{"command-replay-child-" + test.name}
			spec := ExecutionSpec{Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "host_fixture", WorkingDirectory: dir, Environment: map[string]string{}, TimeoutNanos: int64(timeout), MaxOutputBytes: test.maxOutput, Baseline: CommandArm{Path: binary, SHA256: digest, Args: args}, Candidate: CommandArm{Path: binary, SHA256: digest, Args: args}, Inputs: []CommandArtifact{}, Libraries: []CommandArtifact{}}
			spec.SpecDigest, err = ComputeExecutionSpecDigest(spec)
			if err != nil {
				t.Fatal(err)
			}
			runner, err := NewCommandReplayRunner(plan, spec)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := runner.RunArm(context.Background(), "baseline", 1); err == nil {
				t.Fatal("invalid child result accepted")
			}
			attempts := runner.Attempts()
			if len(attempts) != 1 || !attempts[0].Partial || attempts[0].StdoutSHA256 == "" || (test.wantExit != -2 && attempts[0].ExitCode != test.wantExit) {
				t.Fatalf("failure evidence missing: %+v", attempts)
			}
		})
	}
}

func TestCommandSpecAndResultRefuseMutation(t *testing.T) {
	plan, _, _, _ := historyFixture(t)
	spec := ExecutionSpec{Version: commandSpecVersion, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "host_fixture", WorkingDirectory: t.TempDir(), Environment: map[string]string{}, TimeoutNanos: int64(time.Second), MaxOutputBytes: 1024}
	spec.SpecDigest, _ = ComputeExecutionSpecDigest(spec)
	if err := ValidateExecutionSpec(plan, &spec); err == nil {
		t.Fatal("spec without bound arms accepted")
	}

	for _, input := range []string{
		`{"version":1,"version":1,"output_values":[1],"output_sha256":"0000000000000000000000000000000000000000000000000000000000000000","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":1}}`,
		`{"version":1,"Version":1,"output_values":[1],"output_sha256":"0000000000000000000000000000000000000000000000000000000000000000","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":1}}`,
		`{"version":1,"output_values":[1],"output_sha256":"0000000000000000000000000000000000000000000000000000000000000000","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":1},"extra":true}`,
		`{"version":1,"output_values":[1],"output_sha256":"0000000000000000000000000000000000000000000000000000000000000000","timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":1}} {}`,
	} {
		var result CommandResult
		if err := decodeCommandResult([]byte(input), &result); err == nil {
			t.Fatalf("malformed result accepted: %s", input)
		}
	}

	digest, err := OutputDigest([]float64{math.NaN()})
	if err == nil || digest != "" {
		t.Fatal("non-finite output accepted")
	}
}

func TestCommandResultRequiresNumbersButPreservesMeasuredZero(t *testing.T) {
	digest, err := OutputDigest([]float64{0, 2.5})
	if err != nil {
		t.Fatal(err)
	}
	valid := fmt.Sprintf(`{"version":1,"output_values":[0,2.5],"output_sha256":%q,"timing":{"kind":"wall_elapsed","status":"measured","unit":"ns","count":0}}`, digest)
	var result CommandResult
	if err := decodeCommandResult([]byte(valid), &result); err != nil {
		t.Fatalf("explicit measured zero rejected: %v", err)
	}
	if result.Timing.Count != 0 || result.OutputValues[0] != 0 {
		t.Fatalf("explicit zero changed during decode: %+v", result)
	}
	plan, _, _, _ := historyFixture(t)
	if err := validateCommandResult(plan, result); err != nil {
		t.Fatalf("explicit measured zero rejected by public result contract: %v", err)
	}
	for _, input := range []string{
		strings.Replace(valid, `"count":0`, `"count":null`, 1),
		strings.Replace(valid, `,"count":0`, ``, 1),
		strings.Replace(valid, `"output_values":[0,2.5]`, `"output_values":[null,2.5]`, 1),
	} {
		if err := decodeCommandResult([]byte(input), &result); err == nil {
			t.Fatalf("null or missing result number accepted: %s", input)
		}
	}
}

func TestReadExecutionSpecEnforcesPublicBounds(t *testing.T) {
	valid := ExecutionSpec{Version: 1, Environment: map[string]string{}, Inputs: []CommandArtifact{}, Libraries: []CommandArtifact{}, Baseline: CommandArm{Args: []string{}}, Candidate: CommandArm{Args: []string{}}}
	encode := func(spec ExecutionSpec) []byte {
		t.Helper()
		b, err := json.Marshal(spec)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	if _, err := ReadExecutionSpec(bytes.NewReader(encode(valid))); err != nil {
		t.Fatalf("empty boundary spec rejected: %v", err)
	}
	exact := valid
	exact.Environment = make(map[string]string, maxCommandEnv)
	for i := 0; i < maxCommandEnv; i++ {
		exact.Environment[fmt.Sprintf("K%d", i)] = "v"
	}
	exact.Baseline.Args = make([]string, maxCommandArgs)
	exact.Candidate.Args = make([]string, maxCommandArgs)
	exact.Inputs = make([]CommandArtifact, maxCommandArtifacts)
	exact.Libraries = make([]CommandArtifact, maxCommandArtifacts)
	if _, err := ReadExecutionSpec(bytes.NewReader(encode(exact))); err != nil {
		t.Fatalf("exact documented collection bounds rejected: %v", err)
	}
	tests := []struct {
		name string
		edit func(*ExecutionSpec)
	}{
		{"args", func(spec *ExecutionSpec) { spec.Baseline.Args = make([]string, maxCommandArgs+1) }},
		{"environment", func(spec *ExecutionSpec) {
			for i := 0; i <= maxCommandEnv; i++ {
				spec.Environment[fmt.Sprintf("K%d", i)] = "v"
			}
		}},
		{"inputs", func(spec *ExecutionSpec) { spec.Inputs = make([]CommandArtifact, maxCommandArtifacts+1) }},
		{"libraries", func(spec *ExecutionSpec) { spec.Libraries = make([]CommandArtifact, maxCommandArtifacts+1) }},
		{"string", func(spec *ExecutionSpec) { spec.WorkingDirectory = strings.Repeat("x", maxCommandString+1) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := valid
			spec.Environment = map[string]string{}
			spec.Inputs = []CommandArtifact{}
			spec.Libraries = []CommandArtifact{}
			test.edit(&spec)
			if _, err := ReadExecutionSpec(bytes.NewReader(encode(spec))); err == nil {
				t.Fatal("over-limit execution spec accepted by public reader")
			}
		})
	}
}

func TestCommandRunnerRequiresInheritedLockBeforeGPUExec(t *testing.T) {
	lockDir := t.TempDir()
	home := t.TempDir()
	canonical := filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
	if err := os.MkdirAll(canonical, 0700); err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	if dot := strings.IndexByte(host, '.'); dot >= 0 {
		host = host[:dot]
	}
	owner := fmt.Sprintf("pid=%d host=%s label=fixture epoch=1 at=%s\n", os.Getpid(), host, time.Now().Format(time.RFC3339))
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(owner), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("GPU_LOCK_HELD", fmt.Sprint(os.Getpid()))
	t.Setenv("GPU_LOCKDIR", lockDir)
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	digest, size, err := ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	plan, _, _, _ := historyFixture(t)
	plan.BinaryArtifacts = []ArtifactRef{{Name: "runner", Path: binary, Digest: digest, SizeBytes: size}}
	plan.PlanDigest, err = ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	spec := ExecutionSpec{Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "gpu", WorkingDirectory: t.TempDir(), Environment: map[string]string{"GPU_LOCK_HELD": fmt.Sprint(os.Getpid()), "GPU_LOCKDIR": lockDir}, TimeoutNanos: int64(time.Second), MaxOutputBytes: 1024, Baseline: CommandArm{Path: binary, SHA256: digest, Args: []string{}}, Candidate: CommandArm{Path: binary, SHA256: digest, Args: []string{}}, Inputs: []CommandArtifact{}, Libraries: []CommandArtifact{}}
	spec.SpecDigest, err = ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewCommandReplayRunner(plan, spec); err == nil {
		t.Fatal("valid ambient decoy lock accepted without canonical HOME owner record")
	}
}

func TestCommandRunnerUsesCanonicalHomeLockDespiteAmbientOverride(t *testing.T) {
	home := t.TempDir()
	canonical := filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
	if err := os.MkdirAll(canonical, 0700); err != nil {
		t.Fatal(err)
	}
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	if dot := strings.IndexByte(host, '.'); dot >= 0 {
		host = host[:dot]
	}
	owner := fmt.Sprintf("pid=%d host=%s label=fixture epoch=1 at=%s\n", os.Getpid(), host, time.Now().Format(time.RFC3339))
	if err := os.WriteFile(filepath.Join(canonical, "owner"), []byte(owner), 0600); err != nil {
		t.Fatal(err)
	}
	ambient := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("GPU_LOCKDIR", ambient)
	t.Setenv("GPU_LOCK_HELD", fmt.Sprint(os.Getpid()))
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	digest, size, err := ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	plan, _, _, _ := historyFixture(t)
	plan.BinaryArtifacts = []ArtifactRef{{Name: "runner", Path: binary, Digest: digest, SizeBytes: size}}
	plan.PlanDigest, err = ComputePlanDigest(plan)
	if err != nil {
		t.Fatal(err)
	}
	spec := ExecutionSpec{Version: 1, PlanID: plan.PlanID, PlanDigest: plan.PlanDigest, Mode: "gpu", WorkingDirectory: t.TempDir(), Environment: map[string]string{"GPU_LOCK_HELD": fmt.Sprint(os.Getpid()), "GPU_LOCKDIR": canonical}, TimeoutNanos: int64(time.Second), MaxOutputBytes: 1024, Baseline: CommandArm{Path: binary, SHA256: digest, Args: []string{}}, Candidate: CommandArm{Path: binary, SHA256: digest, Args: []string{}}, Inputs: []CommandArtifact{}, Libraries: []CommandArtifact{}}
	spec.SpecDigest, err = ComputeExecutionSpecDigest(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewCommandReplayRunner(plan, spec); err != nil {
		t.Fatalf("canonical HOME lock rejected with unrelated ambient GPU_LOCKDIR: %v", err)
	}
}

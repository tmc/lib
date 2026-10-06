package cudahotswap

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
)

func publicGoRunCPUFixture(t *testing.T) publicGoRunBinding {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "examples/kernelharness"), 0700); err != nil {
		t.Fatal(err)
	}
	sum, err := os.ReadFile(filepath.Join("..", "..", "go.sum"))
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"go.mod":                        "module github.com/tmc/lib/nvidia\n\ngo 1.27.0\n",
		"go.sum":                        string(sum),
		"examples/kernelharness/run.go": "package main\n",
		"examples/kernelharness/main.go": `package main
import("encoding/json";"os")
func main(){cwd,_:=os.Getwd();json.NewEncoder(os.Stdout).Encode(struct{PID,Parent int;Dir string;Args []string}{os.Getpid(),os.Getppid(),cwd,os.Args[1:]})}
`,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	tool := filepath.Join(runtime.GOROOT(), "bin/go")
	tool, err = filepath.EvalSymlinks(tool)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := publicGoRunHash(tool)
	if err != nil {
		t.Fatal(err)
	}
	binding := publicGoRunBinding{GoBinary: tool, GoSHA256: digest, Workdir: root, Files: map[string]string{}}
	for name := range files {
		digest, err := publicGoRunHash(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		binding.Files[name] = digest
	}
	return binding
}

func TestPublicActualGoRunCPU(t *testing.T) {
	binding := publicGoRunCPUFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	candidate := filepath.Join(t.TempDir(), "candidate.cu")
	receipt := filepath.Join(t.TempDir(), "receipt.json")
	command, err := publicGoRunCommand(ctx, binding, candidate, receipt)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{binding.GoBinary, "run", "./examples/kernelharness", "-task", "axpby", "-kernel", candidate, "-json", "-native-receipt", receipt}
	if !reflect.DeepEqual(command.Args, want) || command.Dir != binding.Workdir || command.SysProcAttr != nil || command.WaitDelay != 2*time.Second || command.Cancel == nil {
		t.Fatalf("wrong command: %+v", command)
	}
	start := time.Now()
	output, err := command.Output()
	if err != nil {
		t.Fatalf("actual CPU go run: %v %s", err, output)
	}
	wall := time.Since(start)
	var worker struct {
		PID, Parent int
		Dir         string
		Args        []string
	}
	if err := json.Unmarshal(output, &worker); err != nil {
		t.Fatal(err)
	}
	if worker.PID <= 1 || worker.Parent != command.ProcessState.Pid() || worker.Dir != binding.Workdir || !reflect.DeepEqual(worker.Args, want[3:]) || wall <= 0 {
		t.Fatalf("actual child not launcher descendant: %+v wall=%v", worker, wall)
	}
}

func TestPublicGoRunBindingsRefuse(t *testing.T) {
	for _, kind := range []string{"tool", "source", "missing", "extra", "relative", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			binding := publicGoRunCPUFixture(t)
			switch kind {
			case "tool":
				binding.GoSHA256 = string(make([]byte, 64))
			case "source":
				os.WriteFile(filepath.Join(binding.Workdir, "examples/kernelharness/run.go"), []byte("package main // changed"), 0600)
			case "missing":
				delete(binding.Files, "go.sum")
			case "extra":
				os.WriteFile(filepath.Join(binding.Workdir, "extra.txt"), []byte("unbound"), 0600)
			case "relative":
				binding.Workdir = "relative"
			case "symlink":
				os.Remove(filepath.Join(binding.Workdir, "examples/kernelharness/run.go"))
				os.Symlink("main.go", filepath.Join(binding.Workdir, "examples/kernelharness/run.go"))
			}
			if err := binding.check(); err == nil {
				t.Fatal("unbound Go run accepted")
			}
		})
	}
}

func TestPublicGoRunProducerIdentity(t *testing.T) {
	parent := nativereceipt.Process{PID: 10, Group: 10}
	launcher := nativereceipt.Process{PID: 11, Parent: 10, Group: 10, Start: 3, ExecutableSHA256: "tool"}
	worker := nativereceipt.Process{PID: 12, Parent: 11, Group: 10, Start: 4, ExecutableSHA256: "actual-built-worker"}
	if err := publicGoRunProducer(launcher, parent, worker); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"launcher", "parent", "group", "start", "image"} {
		t.Run(kind, func(t *testing.T) {
			bad := worker
			switch kind {
			case "launcher":
				bad.PID = 11
			case "parent":
				bad.Parent = 10
			case "group":
				bad.Group = 12
			case "start":
				bad.Start = 0
			case "image":
				bad.ExecutableSHA256 = ""
			}
			if err := publicGoRunProducer(launcher, parent, bad); err == nil {
				t.Fatal("wrong worker identity accepted")
			}
		})
	}
}

func TestPublicGoRunBindingJSON(t *testing.T) {
	for _, input := range []string{`{"GoBinary":"a","GoBinary":"b"}`, `{"goBinary":"a"}`, `{"Files":{"a":"1","a":"2"}}`, `{} {}`, `{"Files":[]}`, "{}" + strings.Repeat(" ", 2<<20)} {
		if _, err := publicReadGoRunBinding(strings.NewReader(input)); err == nil {
			t.Fatalf("invalid binding accepted: %.80s", input)
		}
	}
	original := publicGoRunCPUFixture(t)
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	got, err := publicReadGoRunBinding(bytes.NewReader(data))
	if err != nil || !reflect.DeepEqual(got, original) {
		t.Fatalf("binding roundtrip: %+v %v", got, err)
	}
}

func TestPublicGoRunStopChild(t *testing.T) {
	dir := os.Getenv("HOTSWAP_CPU_STOP_CHILD")
	if dir == "" {
		return
	}
	publicColdEdits(t, func(t *testing.T, index int) {
		if err := os.WriteFile(filepath.Join(dir, strconv.Itoa(index)), []byte("entered"), 0600); err != nil {
			t.Fatal(err)
		}
		if index == 0 {
			t.Fatal("owned CPU first-request refusal")
		}
	})
}

func TestPublicGoRunStopOnRefusalCPU(t *testing.T) {
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPublicGoRunStopChild$")
	command.Env = append(os.Environ(), "HOTSWAP_CPU_STOP_CHILD="+dir)
	output, err := command.CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 1 {
		t.Fatalf("owned failing child: %v %s", err, output)
	}
	if _, err := os.Stat(filepath.Join(dir, "0")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "1")); !os.IsNotExist(err) {
		t.Fatal("second edit started after refusal")
	}
}

func TestPublicColdPushLiteralOutcomeCPU(t *testing.T) {
	const source = "owned candidate"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(source)))
	prior := State{Name: "kernel", Kind: "kernel", Generation: 4, Version: 3, ConfigRevision: 7, SourceDigest: "prior"}
	for _, wrong := range []bool{false, true} {
		current := prior
		result := &PushResult{Report: Report{Outcome: OutcomeOK, Recognized: true, Compile: &CompileReport{SourceSHA256: digest}, Replay: &ReplayReport{Native: true, Passed: true}}}
		var pushErr error
		if wrong {
			result.Report.Outcome = OutcomeWrongAnswer
			result.Report.Replay.Passed = false
			pushErr = errors.New("recorded wrong result")
		} else {
			current.Generation++
			current.Version++
			current.SourceDigest = digest
			result.Published = true
		}
		result.State = current
		result.Selected = current
		if err := publicColdPushCheck(prior, current, result, pushErr, source, wrong); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []Outcome{OutcomeHarnessError, OutcomeTimeout, OutcomeDeviceFault, OutcomeCompileError} {
			original := result.Report.Outcome
			result.Report.Outcome = bad
			if err := publicColdPushCheck(prior, current, result, pushErr, source, wrong); err == nil {
				t.Fatalf("wrong=%t accepted actual outcome%s", wrong, bad)
			}
			result.Report.Outcome = original
		}
		original := result.Selected
		result.Selected.Version++
		if err := publicColdPushCheck(prior, current, result, pushErr, source, wrong); err == nil {
			t.Fatal("selected receipt differs from actual state")
		}
		result.Selected = original
		if wrong {
			badCurrent := current
			badCurrent.Generation++
			result.Selected = badCurrent
			if err := publicColdPushCheck(prior, badCurrent, result, pushErr, source, wrong); err == nil {
				t.Fatal("refused candidate changed actual selected code")
			}
		}
	}
}

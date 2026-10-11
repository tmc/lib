package tileiras_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

const nativeEnv = "TILEIRAS_NATIVE"

func nativeUUID(uuid string) error {
	if len(uuid) != 40 || !strings.HasPrefix(uuid, "GPU-") {
		return errors.New("missing or invalid expected GPU UUID")
	}
	for i, c := range uuid[4:] {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return errors.New("invalid expected GPU UUID")
			}
		} else if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return errors.New("invalid expected GPU UUID")
		}
	}
	return nil
}

// nativeTarget admits one exact UUID and compute capability from nvidia-smi.
// Neither an ordinal nor a default architecture substitutes for this identity.
func nativeTarget(expected, report string) (major, minor int, err error) {
	if err := nativeUUID(expected); err != nil {
		return 0, 0, err
	}
	if len(report) > 256 || strings.Contains(strings.TrimSpace(report), "\n") {
		return 0, 0, errors.New("ambiguous or oversized GPU report")
	}
	fields := strings.Split(strings.TrimSpace(report), ",")
	if len(fields) != 2 || strings.TrimSpace(fields[0]) != expected {
		return 0, 0, errors.New("GPU UUID mismatch")
	}
	capability := strings.Split(strings.TrimSpace(fields[1]), ".")
	if len(capability) != 2 {
		return 0, 0, errors.New("invalid compute capability")
	}
	major, err = strconv.Atoi(capability[0])
	if err != nil || major < 8 || major > 99 || strconv.Itoa(major) != capability[0] {
		return 0, 0, errors.New("unsupported compute capability")
	}
	minor, err = strconv.Atoi(capability[1])
	if err != nil || minor < 0 || minor > 9 || strconv.Itoa(minor) != capability[1] {
		return 0, 0, errors.New("invalid compute capability minor")
	}
	return major, minor, nil
}

func nativeAdmission(env func(string) string, probe func(string) (string, error)) (int, int, error) {
	home := env("HOME")
	if !filepath.IsAbs(home) {
		return 0, 0, errors.New("canonical GPU lock home unavailable")
	}
	canonical := filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
	if dir := env("GPU_LOCKDIR"); dir != "" && filepath.Clean(dir) != canonical {
		return 0, 0, errors.New("noncanonical GPU lock directory")
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return 0, 0, fmt.Errorf("GPU lease admission: %w", err)
	}
	uuid := env("TILEIRAS_GPU_UUID")
	if err := nativeUUID(uuid); err != nil {
		return 0, 0, err
	}
	report, err := probe(uuid)
	if err != nil {
		return 0, 0, fmt.Errorf("GPU identity admission: %w", err)
	}
	return nativeTarget(uuid, report)
}

// TestAssembleNative compiles a real semantic kernel into NVIDIA Tile IR,
// assembles that artifact, and checks the resulting executable container.
// TILEIRAS_NATIVE=1 requires an inherited canonical cudalock lease and an exact
// TILEIRAS_GPU_UUID. Any failed admission, compiler or assembler call is fatal;
// there is no retry or unavailable-tool skip. This test creates no CUDA context
// and does not load or execute the output, so it is not numerical qualification.
func TestAssembleNative(t *testing.T) {
	if os.Getenv(nativeEnv) == "" {
		t.Skipf("%s=1 enables the compiler/assembler gate", nativeEnv)
	}
	if os.Getenv(nativeEnv) != "1" {
		t.Fatalf("%s must be 1", nativeEnv)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	major, minor, err := nativeAdmission(os.Getenv, func(uuid string) (string, error) {
		probeCtx, stop := context.WithTimeout(ctx, 10*time.Second)
		defer stop()
		data, err := exec.CommandContext(probeCtx, "nvidia-smi", "-i", uuid,
			"--query-gpu=uuid,compute_cap", "--format=csv,noheader,nounits").CombinedOutput()
		return string(data), err
	})
	if err != nil {
		t.Fatalf("native admission: %v", err)
	}
	graph := cudair.NewGraph()
	a := graph.AddInput("A", []int{4}, cudair.Float32)
	b := graph.AddInput("B", []int{4}, cudair.Float32)
	graph.AddBinary("sub", a, b)
	snapshot, err := graph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	artifact, log, err := tileir.CompileSemanticTile(ctx, snapshot, tileir.LowerOptions{KernelName: "native_subtract"}, major, minor)
	if err != nil {
		t.Fatalf("compile semantic fixture: %v", err)
	}
	if err := artifact.Validate(); err != nil {
		t.Fatalf("compiled artifact: %v", err)
	}
	if artifact.Kind != cudakernel.KindTileIR || len(artifact.Raw) == 0 || len(artifact.Schema.Params) != 3 || !tileiras.IsBytecode(artifact.Raw) {
		t.Fatal("compiler did not return a Tile IR artifact with the expected pointer ABI")
	}
	result, err := tileiras.Assemble(ctx, artifact.Raw, tileiras.WithArch(major, minor), tileiras.WithRemarks("all"))
	if err != nil {
		t.Fatalf("assemble compiler output: %v", err)
	}
	if err := nativeCUBIN(result.CUBIN); err != nil {
		t.Fatalf("assembler output: %v", err)
	}

	t.Logf("uuid=%s target=%s compiler=%s options=%q TileIR=%s CUBIN=%x remarks=%d TMAfailures=%d log=%q",
		os.Getenv("TILEIRAS_GPU_UUID"), artifact.Target, artifact.Toolchain, artifact.CompilerOpts,
		artifact.Digest, sha256.Sum256(result.CUBIN), len(result.Remarks), len(result.TMAFailures()), log)
}

func TestNativeTargetRefusals(t *testing.T) {
	const uuid = "GPU-00000000-0000-0000-0000-000000000001"
	for _, tt := range []struct {
		name, expected, report string
	}{
		{"missing", "", uuid + ", 12.1"},
		{"ordinal", "0", uuid + ", 12.1"},
		{"malformed UUID", "GPU-00000000x0000-0000-0000-000000000001", uuid + ", 12.1"},
		{"different UUID", uuid, "GPU-00000000-0000-0000-0000-000000000002, 12.1"},
		{"ambiguous", uuid, uuid + ", 12.1\n" + uuid + ", 12.1"},
		{"unknown capability", uuid, uuid + ", N/A"},
		{"unsupported capability", uuid, uuid + ", 7.5"},
		{"suffixed capability", uuid, uuid + ", 12.1a"},
		{"noncanonical capability", uuid, uuid + ", +12.1"},
		{"oversized", uuid, strings.Repeat("x", 257)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := nativeTarget(tt.expected, tt.report); err == nil {
				t.Fatal("accepted invalid native identity or target")
			}
		})
	}
	major, minor, err := nativeTarget(uuid, uuid+", 12.1\n")
	if err != nil || major != 12 || minor != 1 {
		t.Fatalf("exact identity = %d.%d, %v", major, minor, err)
	}
}

func TestNativeAdmissionRejectsBeforeProbe(t *testing.T) {
	for _, tt := range []struct {
		name, home, dir, held string
	}{
		{"missing home", "", "", ""},
		{"relative home", "relative", "", ""},
		{"noncanonical directory", t.TempDir(), t.TempDir(), "1"},
		{"missing lease", t.TempDir(), "", ""},
		{"presence flag", t.TempDir(), "", "1"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			env := map[string]string{"HOME": tt.home, "GPU_LOCKDIR": tt.dir, "GPU_LOCK_HELD": tt.held}
			calls := 0
			_, _, err := nativeAdmission(func(k string) string { return env[k] }, func(string) (string, error) {
				calls++
				return "", nil
			})
			if err == nil || calls != 0 {
				t.Fatalf("invalid admission = %v, probe calls=%d", err, calls)
			}
		})
	}
}

func TestNativeAdmissionHeldLeaseAndSingleProbe(t *testing.T) {
	// This is an isolated host lock plus a fake identity probe, not GPU admission.
	home := t.TempDir()
	dir := filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
	if err := os.MkdirAll(filepath.Dir(dir), 0755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	handle, err := cudalock.Acquire(ctx, cudalock.WithDir(dir), cudalock.WithLabel("assembler-host-control"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := handle.Release(); err != nil {
			t.Error(err)
		}
	}()
	const uuid = "GPU-00000000-0000-0000-0000-000000000001"
	env := map[string]string{"HOME": home, "GPU_LOCKDIR": dir,
		"GPU_LOCK_HELD": strconv.Itoa(handle.Owner().PID), "TILEIRAS_GPU_UUID": uuid}
	lookup := func(k string) string { return env[k] }
	calls := 0
	probe := func(got string) (string, error) {
		calls++
		if got != uuid {
			t.Fatalf("probe UUID = %q", got)
		}
		return uuid + ", 12.1", nil
	}
	major, minor, err := nativeAdmission(lookup, probe)
	if err != nil || major != 12 || minor != 1 || calls != 1 {
		t.Fatalf("held admission = %d.%d, %v; probes=%d", major, minor, err, calls)
	}
	calls = 0
	if _, _, err := nativeAdmission(lookup, func(string) (string, error) {
		calls++
		return "", errors.New("probe failed")
	}); err == nil || calls != 1 {
		t.Fatalf("failed probe admission = %v; probes=%d", err, calls)
	}
	// A live directory does not authorize a different claimed owner.
	env["GPU_LOCK_HELD"] = strconv.Itoa(handle.Owner().PID + 1)
	calls = 0
	if _, _, err := nativeAdmission(lookup, probe); err == nil || calls != 0 {
		t.Fatalf("forged owner admission = %v; probes=%d", err, calls)
	}
	env["GPU_LOCK_HELD"] = strconv.Itoa(handle.Owner().PID)
	// Missing UUID must refuse even with a real owner record, before the probe.
	env["TILEIRAS_GPU_UUID"] = ""
	calls = 0
	if _, _, err := nativeAdmission(lookup, probe); err == nil || calls != 0 {
		t.Fatalf("missing UUID admission = %v; probes=%d", err, calls)
	}
}

func nativeCUBIN(data []byte) error {
	file, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("invalid ELF CUBIN: %w", err)
	}
	defer file.Close()
	if file.Machine != elf.EM_CUDA || file.Type != elf.ET_EXEC {
		return errors.New("assembler output is not a CUDA executable")
	}
	return nil
}

func TestNativeCUBINRejectsMagicOnlyOutput(t *testing.T) {
	for _, data := range [][]byte{nil, {0x7f, 'E', 'L', 'F'}, make([]byte, 64)} {
		if err := nativeCUBIN(data); err == nil {
			t.Fatal("accepted incomplete executable container")
		}
	}
}

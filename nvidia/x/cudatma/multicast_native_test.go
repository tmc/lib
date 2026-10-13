//go:build cuda_tma_multicast_qualification

package cudatma

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This separate opt-in compiler packet is source only until the coordinator
// admits a pinned toolkit/host lease. It performs no GPU execution. The native
// numerical/participation packet still needs a concrete launch/allocation owner
// and physical completion receipts; the compiler packet cannot establish that.
func TestMulticastCompilePTX(t *testing.T) {
	if os.Getenv("CUDA_TMA_MULTICAST_COMPILE") != "1" {
		t.Skip("compiler qualification requires explicit opt-in")
	}
	// No discovery/default compiler: a qualified caller supplies the exact path.
	nvcc := os.Getenv("CUDA_TMA_MULTICAST_NVCC")
	if !filepath.IsAbs(nvcc) {
		t.Fatal("explicit absolute qualified nvcc path required")
	}
	for rank := 1; rank <= 5; rank++ {
		p, err := NewMulticastPlan(multicastConfig(rank, 4, 128))
		if err != nil {
			t.Fatal(err)
		}
		s, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		dir := t.TempDir()
		input := filepath.Join(dir, "multicast.cu")
		output := filepath.Join(dir, "multicast.ptx")
		if err := os.WriteFile(input, []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		result, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_100a", "-ptx", input, "-o", output).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("rank%d compile: %v\n%s", rank, err, result)
		}
		b, err := os.ReadFile(output)
		if err != nil {
			t.Fatal(err)
		}
		text := string(b)
		for _, want := range []string{".multicast::cluster", "cvta.param", "fence.mbarrier_init.release.cluster", "mbarrier.arrive.expect_tx", "mbarrier.try_wait.parity", "barrier.cluster"} {
			if !strings.Contains(text, want) {
				t.Fatalf("rank%d PTX missing %s", rank, want)
			}
		}
		// A missing toolkit/unsupported target is a failed requested gate, never
		// accepted as a skipped compiler qualification.
	}
}

// TestMulticastPrepareQualification produces an exact prebuilt corpus only
// after a separate, pinned compiler-stage approval. It never loads a driver
// or executes a GPU kernel. Source implementation approval does not enable it.
func TestMulticastPrepareQualification(t *testing.T) {
	if os.Getenv("CUDA_TMA_MULTICAST_PREPARE") != "1" {
		t.Skip("compiler preparation requires separate explicit authorization")
	}
	manifestPath := os.Getenv("CUDA_TMA_MULTICAST_PREPARE_MANIFEST")
	pin := os.Getenv("CUDA_TMA_MULTICAST_PREPARE_MANIFEST_SHA256")
	data, err := readQualificationCompilerFile(manifestPath, pin, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		AuthorizationID, CandidateCommit, ImplementationSHA256, NVCCFile, NVCCSHA256, OutputDir string
		CaseIDs                                                                                 []string
		MaxInvocations, OverallSeconds, CommandSeconds                                          int
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err = dec.Decode(&request); err != nil {
		t.Fatal(err)
	}
	if request.AuthorizationID == "" || request.AuthorizationID != os.Getenv("CUDA_TMA_MULTICAST_PREPARE_AUTHORIZATION_ID") || !digestString(request.CandidateCommit, 20) || request.ImplementationSHA256 != qualificationDigest() || !filepath.IsAbs(request.NVCCFile) || !digestString(request.NVCCSHA256, 32) || !filepath.IsAbs(request.OutputDir) || request.MaxInvocations <= 0 || request.OverallSeconds <= 0 || request.OverallSeconds > 86400 || request.CommandSeconds <= 0 || request.CommandSeconds > request.OverallSeconds {
		t.Fatal("explicit compiler identity/source/budgets required")
	}
	if len(request.CaseIDs) == 0 || len(request.CaseIDs)*2+1 > request.MaxInvocations {
		t.Fatal("case list exceeds explicit compiler budget")
	}
	var cases []qualificationCase
	seen := map[string]bool{}
	for _, id := range request.CaseIDs {
		found := false
		for _, c := range qualificationCases() {
			if c.ID == id {
				if seen[id] {
					t.Fatal("duplicate compiler case")
				}
				seen[id] = true
				cases = append(cases, c)
				found = true
				break
			}
		}
		if !found {
			t.Fatal("unknown/unsafe compiler case", id)
		}
	}
	if _, err = readQualificationCompilerFile(request.NVCCFile, request.NVCCSHA256, 128<<20); err != nil {
		t.Fatal(err)
	}
	if err = os.Mkdir(request.OutputDir, 0700); err != nil {
		t.Fatal("new exclusive output directory required:", err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Duration(request.OverallSeconds)*time.Second)
	defer cancel()
	command := func(argv []string) ([]byte, error) {
		sub, c := context.WithTimeout(ctx, time.Duration(request.CommandSeconds)*time.Second)
		defer c()
		return exec.CommandContext(sub, argv[0], argv[1:]...).CombinedOutput()
	}
	version, err := command([]string{request.NVCCFile, "--version"})
	if write := os.WriteFile(filepath.Join(request.OutputDir, "compiler-version.log"), version, 0600); write != nil {
		t.Fatal(write)
	}
	if err != nil {
		t.Fatal("compiler identity query failed:", err)
	}
	var artifacts []qualificationArtifact
	for _, c := range cases {
		source, err := qualificationSource(c)
		if err != nil {
			t.Fatal(err)
		}
		dir := filepath.Join(request.OutputDir, c.ID)
		if err = os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
		artifact := qualificationArtifact{CaseID: c.ID, SourceFile: filepath.Join(dir, "fixture.cu"), PTXFile: filepath.Join(dir, "fixture.ptx"), CubinFile: filepath.Join(dir, "fixture.cubin"), CompilerReceiptFile: filepath.Join(dir, "compiler.json")}
		if err = os.WriteFile(artifact.SourceFile, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		hash := sha256.Sum256([]byte(source))
		artifact.SourceSHA256 = hex.EncodeToString(hash[:])
		receipt := qualificationCompilerReceipt{CaseID: c.ID, Target: TargetSM100a, CandidateCommit: request.CandidateCommit, ImplementationSHA256: request.ImplementationSHA256, NVCCFile: request.NVCCFile, NVCCSHA256: request.NVCCSHA256, NVCCVersion: strings.TrimSpace(string(version)), VersionOutput: string(version), SourceSHA256: artifact.SourceSHA256}
		receipt.PTXCommand = []string{request.NVCCFile, "-std=c++17", "-arch=sm_100a", "-ptx", artifact.SourceFile, "-o", artifact.PTXFile}
		receipt.CubinCommand = []string{request.NVCCFile, "-std=c++17", "-arch=sm_100a", "-cubin", artifact.SourceFile, "-o", artifact.CubinFile}
		ptxOutput, ptxErr := command(receipt.PTXCommand)
		receipt.PTXOutput = string(ptxOutput)
		if ptxErr != nil {
			receipt.PTXExit = qualificationCompilerExit(ptxErr)
		}
		// Retain preparation failures as well; compiler failures are not numerical
		// controls and no retry/fallback target is attempted.
		save := func() {
			b, e := json.MarshalIndent(receipt, "", "  ")
			if e != nil {
				t.Fatal(e)
			}
			if e = os.WriteFile(artifact.CompilerReceiptFile, append(b, '\n'), 0600); e != nil {
				t.Fatal(e)
			}
		}
		if ptxErr != nil {
			save()
			t.Fatalf("PTX compiler attempt failed: %v\n%s", ptxErr, ptxOutput)
		}
		ptx, err := os.ReadFile(artifact.PTXFile)
		if err != nil {
			save()
			t.Fatal(err)
		}
		hash = sha256.Sum256(ptx)
		artifact.PTXSHA256 = hex.EncodeToString(hash[:])
		receipt.PTXSHA256 = artifact.PTXSHA256
		if err = qualificationPTX(string(ptx), c); err != nil {
			save()
			t.Fatal("semantic PTX admission failed:", err)
		}
		cubinOutput, cubinErr := command(receipt.CubinCommand)
		receipt.CubinOutput = string(cubinOutput)
		if cubinErr != nil {
			receipt.CubinExit = qualificationCompilerExit(cubinErr)
			save()
			t.Fatalf("cubin compiler attempt failed: %v\n%s", cubinErr, cubinOutput)
		}
		cubin, err := os.ReadFile(artifact.CubinFile)
		if err != nil {
			save()
			t.Fatal(err)
		}
		if err = qualificationCubin(cubin); err != nil {
			save()
			t.Fatal("native CUDA ELF admission failed:", err)
		}
		hash = sha256.Sum256(cubin)
		artifact.CubinSHA256 = hex.EncodeToString(hash[:])
		receipt.CubinSHA256 = artifact.CubinSHA256
		save()
		body, err := os.ReadFile(artifact.CompilerReceiptFile)
		if err != nil {
			t.Fatal(err)
		}
		hash = sha256.Sum256(body)
		artifact.CompilerReceiptSHA256 = hex.EncodeToString(hash[:])
		artifacts = append(artifacts, artifact)
	}
	b, err := json.MarshalIndent(artifacts, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(request.OutputDir, "artifacts.json"), append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
func readQualificationCompilerFile(path, pin string, limit int64) ([]byte, error) {
	if !filepath.IsAbs(path) || !digestString(pin, 32) {
		return nil, fmt.Errorf("absolute pinned compiler-stage file required")
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		return nil, e
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, fmt.Errorf("compiler-stage file exceeds regular bounded input")
	}
	b, e := io.ReadAll(io.LimitReader(f, limit+1))
	if e != nil {
		return nil, e
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("compiler-stage file grew beyond bound")
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != pin {
		return nil, fmt.Errorf("compiler-stage file digest mismatch")
	}
	return b, nil
}

func qualificationCompilerExit(err error) int {
	if err == nil {
		return 0
	}
	if e, ok := err.(*exec.ExitError); ok {
		return e.ExitCode()
	}
	return -1
}

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestInspectTileIRAndRemarks(t *testing.T) {
	tileIRText := `
cuda_tile.module @gemm {
  cuda_tile.entry @matmul(%arg0: !cuda_tile.tile<ptr<f32>>, %arg1: !cuda_tile.tile<ptr<f32>>) {
    %tok = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %arg0, %tok : (!cuda_tile.tile<ptr<f32>>, !cuda_tile.token) -> !cuda_tile.tile<16x16xf32>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<16x16xf32>
    %acc = cuda_tile.mma_f %vA, %vA, %cst : (!cuda_tile.tile<16x16xf32>, !cuda_tile.tile<16x16xf32>, !cuda_tile.tile<16x16xf32>) -> !cuda_tile.tile<16x16xf32>
    cuda_tile.return
  }
}
`
	remarksYAML := `--- !Passed
Pass: Tensor-core
Name: RemarkTensorCoreInstructionSelected
Function: matmul
Args:
  - String: 'Instruction selected: mma.sync.aligned.m16n8k8.row.col'

--- !Failure
Pass: Memory
Name: RemarkMemoryTmaLoadFailed
Function: matmul
Args:
  - String: 'TMA load instruction selection failed'
  - Reason: 'Shape [16, 16] is below minimum 2D box'
`

	fn := &profile.Function{ID: 1, Name: "matmul", Filename: "gemm.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "gemm.cu",
			PTX:    "gemm.ptx",
			CUBIN:  "gemm.cubin",
			SASS:   "gemm.sass",
			TileIR: "gemm.tileir",
		},
		Files: map[string][]byte{
			"gemm.cu":               []byte("void matmul() {}\n"),
			"gemm.ptx":              []byte(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry matmul()\n{\nret;\n}\n"),
			"gemm.cubin":            []byte("CUBIN_DATA"),
			"gemm.sass":             []byte(".section .text.matmul\n/* 0000 */ RET;\n"),
			"gemm.tileir":           []byte(tileIRText),
			"tileiras-remarks.yaml": []byte(remarksYAML),
		},
	}

	dir := t.TempDir()
	capPath := filepath.Join(dir, "capture.pprof")
	f, err := os.Create(capPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.Write(f, cap); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()

	// 1. Test inspect --tileir
	t.Run("inspect_--tileir", func(t *testing.T) {
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		err := runInspectCommand([]string{"--tileir", capPath})
		w.Close()
		os.Stdout = oldStdout

		if err != nil {
			t.Fatalf("runInspectCommand(--tileir) failed: %v", err)
		}

		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)

		var summary inspectTileIRSummary
		if err := json.Unmarshal(buf.Bytes(), &summary); err != nil {
			t.Fatalf("unmarshal tileir summary: %v\nOutput: %s", err, buf.String())
		}

		if summary.File != "gemm.tileir" {
			t.Errorf("File = %q, want gemm.tileir", summary.File)
		}
		if len(summary.Functions) != 1 || summary.Functions[0] != "matmul" {
			t.Errorf("Functions = %v, want [matmul]", summary.Functions)
		}
		if summary.TMAOps != 1 {
			t.Errorf("TMAOps = %d, want 1", summary.TMAOps)
		}
		if summary.MMAOps != 1 {
			t.Errorf("MMAOps = %d, want 1", summary.MMAOps)
		}
	})

	// 2. Test inspect --remarks
	t.Run("inspect_--remarks", func(t *testing.T) {
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		err := runInspectCommand([]string{"--remarks", capPath})
		w.Close()
		os.Stdout = oldStdout

		if err != nil {
			t.Fatalf("runInspectCommand(--remarks) failed: %v", err)
		}

		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)

		var summary inspectRemarksSummary
		if err := json.Unmarshal(buf.Bytes(), &summary); err != nil {
			t.Fatalf("unmarshal remarks summary: %v\nOutput: %s", err, buf.String())
		}

		if summary.Total != 2 {
			t.Errorf("Total = %d, want 2", summary.Total)
		}
		if summary.TMAFailures != 1 {
			t.Errorf("TMAFailures = %d, want 1", summary.TMAFailures)
		}
		if summary.TensorCoreMMAs != 1 {
			t.Errorf("TensorCoreMMAs = %d, want 1", summary.TensorCoreMMAs)
		}
	})

	// 3. Test inspect full report contains TileIR and Remarks
	t.Run("inspect_full_report", func(t *testing.T) {
		oldStdout := os.Stdout
		r, w, _ := os.Pipe()
		os.Stdout = w

		err := runInspectCommand([]string{"--profiles", capPath})
		w.Close()
		os.Stdout = oldStdout

		if err != nil {
			t.Fatalf("runInspectCommand(--profiles) failed: %v", err)
		}

		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)

		var report inspectReport
		if err := json.Unmarshal(buf.Bytes(), &report); err != nil {
			t.Fatalf("unmarshal report: %v\nOutput: %s", err, buf.String())
		}

		if report.TileIR == nil {
			t.Fatal("report.TileIR is nil")
		}
		if report.TileIR.TMAOps != 1 {
			t.Errorf("report.TileIR.TMAOps = %d, want 1", report.TileIR.TMAOps)
		}
		if report.Remarks == nil {
			t.Fatal("report.Remarks is nil")
		}
		if report.Remarks.TMAFailures != 1 {
			t.Errorf("report.Remarks.TMAFailures = %d, want 1", report.Remarks.TMAFailures)
		}
	})

	// 4. Test fetcher comments include TileIR and compiler remarks
	t.Run("fetcher_comments", func(t *testing.T) {
		f := &fetcher{
			path: capPath,
			dir:  dir,
			cap:  cap,
		}
		prof, _, err := f.Fetch(capPath, 0, 0)
		if err != nil {
			t.Fatalf("fetch failed: %v", err)
		}

		foundTileIRComment := false
		foundRemarksComment := false
		for _, comment := range prof.Comments {
			if strings.Contains(comment, "Tile IR") && strings.Contains(comment, "TMA operations") {
				foundTileIRComment = true
			}
			if strings.Contains(comment, "Compiler remarks") && strings.Contains(comment, "Tensor Core MMA") {
				foundRemarksComment = true
			}
		}

		if !foundTileIRComment {
			t.Errorf("missing Tile IR comment in profile comments: %v", prof.Comments)
		}
		if !foundRemarksComment {
			t.Errorf("missing Compiler remarks comment in profile comments: %v", prof.Comments)
		}
	})
}

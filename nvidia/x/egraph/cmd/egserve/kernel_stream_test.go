package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestKernelsAnnotatedJSONAndTextCrossAndFullLayer(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	for _, tc := range []struct {
		mode          string
		bucket        string
		wantMinStages int
	}{
		{mode: "cross_layer", bucket: "1", wantMinStages: 4},
		{mode: "cross_layer", bucket: "4", wantMinStages: 4},
		{mode: "full_layer", bucket: "1", wantMinStages: 18},
		{mode: "full_layer", bucket: "8", wantMinStages: 18},
	} {
		t.Run(tc.mode+"_M"+tc.bucket, func(t *testing.T) {
			// 1. Test JSON format.
			jsonURL := ts.URL + "/v1/kernels/annotated?mode=" + tc.mode + "&bucket=" + tc.bucket + "&format=json"
			resp, err := ts.Client().Get(jsonURL)
			if err != nil {
				t.Fatalf("GET %s: %v", jsonURL, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET %s status = %d, want 200", jsonURL, resp.StatusCode)
			}
			var annotated AnnotatedKernelResponse
			if err := json.NewDecoder(resp.Body).Decode(&annotated); err != nil {
				t.Fatalf("decode JSON from %s: %v", jsonURL, err)
			}
			if !annotated.BarrierSafetyVerified {
				t.Fatalf("BarrierSafetyVerified = false, want true")
			}
			if len(annotated.CUDALines) == 0 {
				t.Fatalf("expected non-empty CUDALines")
			}
			if len(annotated.Stages) < tc.wantMinStages {
				t.Fatalf("len(Stages) = %d, want >= %d", len(annotated.Stages), tc.wantMinStages)
			}
			if len(annotated.Iterations) != 4 {
				t.Fatalf("len(Iterations) = %d, want 4", len(annotated.Iterations))
			}
			if annotated.HoldoutSpeedup < 6.0 {
				t.Fatalf("HoldoutSpeedup = %.2f, want >= 6.0", annotated.HoldoutSpeedup)
			}
			if annotated.TotalSpillBytes != 0 {
				t.Fatalf("promoted TotalSpillBytes = %d, want 0", annotated.TotalSpillBytes)
			}

			// Verify every stage has valid spans, PTXAS resources, SASS instructions, CUPTI stalls, and EGraphCostNs.
			for _, st := range annotated.Stages {
				if st.StartLine <= 0 || st.EndLine < st.StartLine || st.EndLine > len(annotated.CUDALines) {
					t.Fatalf("stage %s invalid line span [%d, %d] (total lines %d)",
						st.StageName, st.StartLine, st.EndLine, len(annotated.CUDALines))
				}
				if st.RegistersPerThread <= 0 || st.OccupancyCTAsPerSM <= 0 {
					t.Fatalf("stage %s invalid PTXAS metrics: regs=%d occ=%d",
						st.StageName, st.RegistersPerThread, st.OccupancyCTAsPerSM)
				}
				if st.TotalInsts <= 0 || len(st.SASSInsts) == 0 || st.WaitStallWeight <= 0 {
					t.Fatalf("stage %s invalid SASS metrics: total=%d insts=%d wait=%d",
						st.StageName, st.TotalInsts, len(st.SASSInsts), st.WaitStallWeight)
				}
				for _, ins := range st.SASSInsts {
					if ins.PC == 0 || ins.Mnemonic == "" || ins.WaitMask == "" || ins.ScoreboardWeight <= 0 {
						t.Fatalf("stage %s invalid AnnotatedSASSInst: %+v", st.StageName, ins)
					}
				}
				if st.TotalSamples == 0 || st.DominantReason == "" || st.RecommendedRewrite == "" {
					t.Fatalf("stage %s invalid CUPTI metrics: samples=%d reason=%q rewrite=%q",
						st.StageName, st.TotalSamples, st.DominantReason, st.RecommendedRewrite)
				}
				if st.EGraphCostNs <= 0 || st.CalibratedMultiplier <= 0 {
					t.Fatalf("stage %s invalid cost metrics: cost=%.2f mult=%.3f",
						st.StageName, st.EGraphCostNs, st.CalibratedMultiplier)
				}
				if !st.RAWVerified || !st.ArenaNonAliased || !st.NamedBarrierUniform {
					t.Fatalf("stage %s formal barrier proof flags false: RAW=%t Arena=%t Uniform=%t",
						st.StageName, st.RAWVerified, st.ArenaNonAliased, st.NamedBarrierUniform)
				}
				ann, ok := annotated.LineAnnotations[st.StartLine]
				if !ok {
					t.Fatalf("missing LineAnnotation at stage %s StartLine=%d", st.StageName, st.StartLine)
				}
				if ann.StageName != st.StageName || ann.EGraphCostNs <= 0 || ann.CUPTIStallPct <= 0 || len(ann.SASSInsts) == 0 {
					t.Fatalf("invalid LineAnnotation at line %d: %+v", st.StartLine, ann)
				}
			}

			// 2. Test format=text inline-annotated CUDA/PTX/SASS listing.
			textURL := ts.URL + "/v1/kernels/annotated?mode=" + tc.mode + "&bucket=" + tc.bucket + "&format=text"
			textResp, err := ts.Client().Get(textURL)
			if err != nil {
				t.Fatalf("GET %s: %v", textURL, err)
			}
			defer textResp.Body.Close()
			if textResp.StatusCode != http.StatusOK {
				t.Fatalf("GET %s status = %d, want 200", textURL, textResp.StatusCode)
			}
			ct := textResp.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, "text/plain") {
				t.Fatalf("Content-Type = %q, want text/plain", ct)
			}
			bodyBytes, err := io.ReadAll(textResp.Body)
			if err != nil {
				t.Fatalf("read text body: %v", err)
			}
			text := string(bodyBytes)
			for _, wantSub := range []string{
				"/* [STAGE s0",
				"cost=",
				"regs=",
				"spill=0B",
				"SASS",
				"CUPTI",
				"#line 1",
				"PTX:",
			} {
				if !strings.Contains(text, wantSub) {
					t.Fatalf("format=text missing substring %q", wantSub)
				}
			}
		})
	}
}

func TestKernelsStreamSSEProgression(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	streamURL := ts.URL + "/v1/kernels/stream?once=1&mode=cross_layer&bucket=1"
	resp, err := ts.Client().Get(streamURL)
	if err != nil {
		t.Fatalf("GET %s: %v", streamURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s status = %d, want 200", streamURL, resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	var (
		frames       []KernelStreamIteration
		sawComplete  bool
		currentEvent string
	)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if after, ok := strings.CutPrefix(line, "event: "); ok {
			currentEvent = strings.TrimSpace(after)
			continue
		}
		if after, ok := strings.CutPrefix(line, "data: "); ok {
			payload := strings.TrimSpace(after)
			switch currentEvent {
			case "kernel_iteration":
				var iter KernelStreamIteration
				if err := json.Unmarshal([]byte(payload), &iter); err != nil {
					t.Fatalf("unmarshal kernel_iteration SSE data: %v", err)
				}
				frames = append(frames, iter)
			case "stream_complete":
				sawComplete = true
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan SSE stream: %v", err)
	}
	if len(frames) != 4 {
		t.Fatalf("received %d kernel_iteration SSE frames, want 4", len(frames))
	}
	if !sawComplete {
		t.Fatalf("expected final event: stream_complete in SSE stream")
	}

	first := frames[0]
	last := frames[len(frames)-1]

	if first.TotalSpillBytes <= 0 {
		t.Fatalf("iteration 0 TotalSpillBytes = %d, want > 0 on unrefined baseline", first.TotalSpillBytes)
	}
	var firstHasBankConflict bool
	for _, st := range first.Stages {
		if st.SharedBankConflictRisk {
			firstHasBankConflict = true
			break
		}
	}
	if !firstHasBankConflict {
		t.Fatalf("expected iteration 0 to report SharedBankConflictRisk=true on unswizzled baseline")
	}

	if last.TotalSpillBytes != 0 {
		t.Fatalf("final iteration TotalSpillBytes = %d, want 0", last.TotalSpillBytes)
	}
	for _, st := range last.Stages {
		if st.SharedBankConflictRisk {
			t.Fatalf("final iteration stage %s still has SharedBankConflictRisk=true", st.StageName)
		}
	}
	if last.HoldoutSpeedup <= 1.0 || last.HoldoutSpeedup < 6.0 {
		t.Fatalf("final iteration HoldoutSpeedup = %.2f, want >= 6.0 (> 1.0)", last.HoldoutSpeedup)
	}
	if last.MaxRegs >= first.MaxRegs {
		t.Fatalf("expected final MaxRegs (%d) < baseline MaxRegs (%d)", last.MaxRegs, first.MaxRegs)
	}
	for i := 1; i < len(frames); i++ {
		prevLines := frames[i-1].CUDALines
		curLines := frames[i].CUDALines
		if len(prevLines) != len(curLines) {
			t.Fatalf("iteration %d CUDALines len = %d, want %d", i, len(curLines), len(prevLines))
		}
		var diffCount int
		for j := range curLines {
			if prevLines[j] != curLines[j] {
				diffCount++
			}
		}
		if diffCount == 0 {
			t.Fatalf("expected CUDALines to mutate between iteration %d and %d, got 0 line diffs", i-1, i)
		}
	}
}

func TestInspectorUIAndDemoStreamCLIFlag(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:      "sm_121a",
		HiddenDim: 256,
		IntermDim: 512,
		Layers:    4,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	resp, err := ts.Client().Get(ts.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET / status = %d, want 200", resp.StatusCode)
	}
	htmlBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read GET / body: %v", err)
	}
	html := string(htmlBytes)
	for _, want := range []string{
		"EventSource",
		"/v1/kernels/stream",
		"/v1/kernels/annotated",
		"PTXAS",
		"SASS",
		"CUPTI",
		"content-visibility: auto",
	} {
		if !strings.Contains(html, want) {
			t.Fatalf("GET / HTML missing %q", want)
		}
	}

	// Verify -demo-stream CLI flag prints summary and inline-annotated preview cleanly when -addr="".
	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-addr", "",
		"-arch", "sm_121a",
		"-demo-stream",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run(-demo-stream -addr '') exited with %d; stderr:\n%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"/v1/kernels/stream",
		"/v1/kernels/annotated?format=text",
		"baseline_unfused",
		"cupti_holdout_promoted",
		"/* [STAGE s0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("-demo-stream output missing %q:\n%s", want, out)
		}
	}
}

package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

// These are literal CPU report controls, not native readback evidence.
func nativeStateCPUReport(wrong bool) (cudahotswap.State, cudahotswap.State, *cudahotswap.PushResult) {
	prior := cudahotswap.State{Name: "saxpy", Kind: "kernel", Generation: 10, Version: 4, ConfigRevision: 2, SourceDigest: "baseline", CompilerKey: "baseline-key"}
	candidate := prior
	candidate.Generation++
	candidate.Version++
	candidate.Label = "Mac native control"
	candidate.SourceDigest = fmt.Sprintf("%x", sha256.Sum256([]byte("literal candidate")))
	candidate.CompilerKey = "candidate-key"
	current := candidate
	p := &cudahotswap.PushResult{State: candidate, Selected: candidate, Published: true, Report: cudahotswap.Report{Outcome: cudahotswap.OutcomeOK, Recognized: true, Generation: candidate.Generation, ConfigRevision: candidate.ConfigRevision, Compile: &cudahotswap.CompileReport{SourceSHA256: candidate.SourceDigest}}}
	if wrong {
		current = prior
		current.Generation = candidate.Generation + 1
		current.Label = "Mac native control: rollback"
		p.Selected = current
		p.RolledBack = true
		p.Report.Outcome = cudahotswap.OutcomeWrongAnswer
	}
	return prior, current, p
}

func nativeStateCPUResult(t *testing.T, p *cudahotswap.PushResult, wrong bool) *mcp.CallToolResult {
	t.Helper()
	b, err := json.Marshal(httpapi.Response{Push: p})
	if err != nil {
		t.Fatal(err)
	}
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: string(b)}}, IsError: wrong}
}

func TestNativeMCPStateContradictionsCPU(t *testing.T) {
	for _, wrong := range []bool{false, true} {
		prior, current, p := nativeStateCPUReport(wrong)
		if err := nativeMCPPushCheck(prior, current, nativeStateCPUResult(t, p, wrong), "literal candidate", "Mac native control", wrong); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name   string
			change func(*cudahotswap.PushResult)
		}{
			{"outcome", func(p *cudahotswap.PushResult) { p.Report.Outcome = cudahotswap.OutcomeTimeout }},
			{"published", func(p *cudahotswap.PushResult) { p.Published = false }},
			{"rollback", func(p *cudahotswap.PushResult) { p.RolledBack = !wrong }},
			{"selected", func(p *cudahotswap.PushResult) { p.Selected.Version++ }},
			{"source", func(p *cudahotswap.PushResult) { p.State.SourceDigest = "wrong" }},
			{"config", func(p *cudahotswap.PushResult) { p.State.ConfigRevision++ }},
			{"stamp", func(p *cudahotswap.PushResult) { p.Report.Generation++ }},
			{"label", func(p *cudahotswap.PushResult) { p.State.Label = "other" }},
		} {
			t.Run(fmt.Sprintf("%t/%s", wrong, tc.name), func(t *testing.T) {
				copy := *p
				tc.change(&copy)
				if nativeMCPPushCheck(prior, current, nativeStateCPUResult(t, &copy, wrong), "literal candidate", "Mac native control", wrong) == nil {
					t.Fatal("contradiction accepted")
				}
			})
		}
		if wrong {
			bad := current
			bad.SourceDigest = p.State.SourceDigest
			copy := *p
			copy.Selected = bad
			if nativeMCPPushCheck(prior, bad, nativeStateCPUResult(t, &copy, wrong), "literal candidate", "Mac native control", wrong) == nil {
				t.Fatal("wrong source remains selected")
			}
		}
	}
}

func TestNativeMCPStatusPushStatusThroughSDKCPU(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, wrong := range []bool{true, false} {
		prior, current, push := nativeStateCPUReport(wrong)
		var calls []string
		var mu sync.Mutex
		h := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()
			calls = append(calls, r.URL.Path)
			var response httpapi.Response
			if r.URL.Path == "/v1/push" {
				response.Push = push
			} else {
				s := prior
				if len(calls) == 3 {
					s = current
				}
				response.State = &s
			}
			json.NewEncoder(w).Encode(response)
		}))
		server := newServer(&httpapi.Client{URL: h.URL, HTTP: h.Client()})
		left, right := mcp.NewInMemoryTransports()
		ss, err := server.Connect(ctx, left, nil)
		if err != nil {
			t.Fatal(err)
		}
		client := mcp.NewClient(&mcp.Implementation{Name: "state-CPU", Version: "1"}, nil)
		session, err := client.Connect(ctx, right, nil)
		if err != nil {
			t.Fatal(err)
		}
		before, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "kernel_status", Arguments: nameInput{Name: "saxpy"}})
		if err != nil {
			t.Fatal(err)
		}
		result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "push_kernel", Arguments: pushInput{Name: "saxpy", Label: "Mac native control", Source: sourceInput{Code: "literal candidate", Entry: "saxpy"}}})
		if err != nil {
			t.Fatal(err)
		}
		after, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "kernel_status", Arguments: nameInput{Name: "saxpy"}})
		if err != nil {
			t.Fatal(err)
		}
		b, err := nativeMCPResponse(before)
		if err != nil {
			t.Fatal(err)
		}
		a, err := nativeMCPResponse(after)
		if err != nil {
			t.Fatal(err)
		}
		if before.IsError || after.IsError || b.State == nil || a.State == nil {
			t.Fatal("missing actual SDK status")
		}
		if err := nativeMCPPushCheck(*b.State, *a.State, result, "literal candidate", "Mac native control", wrong); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		actualCalls := append([]string(nil), calls...)
		mu.Unlock()
		if !reflect.DeepEqual(actualCalls, []string{"/v1/status", "/v1/push", "/v1/status"}) {
			t.Fatal(actualCalls)
		}
		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
		if err := ss.Close(); err != nil {
			t.Fatal(err)
		}
		h.Close()
	}
}

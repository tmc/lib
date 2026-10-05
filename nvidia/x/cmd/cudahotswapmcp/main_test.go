package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

func TestToolsThroughHTTP(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, outcome := range []cudahotswap.Outcome{cudahotswap.OutcomeWrongAnswer, cudahotswap.OutcomeOK} {
		t.Run(string(outcome), func(t *testing.T) {
			called := make(chan string, 10)
			httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called <- r.URL.Path
				if r.URL.Path == "/v1/events" {
					w.Header().Set("Content-Type", "application/x-ndjson")
					json.NewEncoder(w).Encode(httpapi.Frame{Event: &cudahotswap.Event{Sequence: 3, Name: "kernel", Report: &cudahotswap.Report{Outcome: outcome}}})
					return
				}
				json.NewEncoder(w).Encode(httpapi.Response{Push: &cudahotswap.PushResult{Report: cudahotswap.Report{Outcome: outcome, Reason: "reference reason"}}})
			}))
			defer httpServer.Close()
			p := &httpapi.Client{URL: httpServer.URL, HTTP: httpServer.Client()}
			s := newServer(p)
			left, right := mcp.NewInMemoryTransports()
			ss, err := s.Connect(ctx, left, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer ss.Close()
			client := mcp.NewClient(&mcp.Implementation{Name: "host-control", Version: "1"}, nil)
			session, err := client.Connect(ctx, right, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer session.Close()
			tools, err := session.ListTools(ctx, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(tools.Tools) != 9 {
				t.Fatalf("tools %d", len(tools.Tools))
			}
			geometry := map[string]any{"Grid": map[string]any{"X": 1, "Y": 1, "Z": 1}, "Block": map[string]any{"X": 32, "Y": 1, "Z": 1}, "SharedMem": 0}
			for _, tc := range []struct {
				name, endpoint string
				args           map[string]any
			}{
				{"push_kernel", "push", map[string]any{"name": "kernel", "label": "agent", "source": map[string]any{"code": "cuda", "entry": "entry"}}},
				{"revert_kernel", "revert", map[string]any{"name": "kernel", "label": "agent"}},
				{"list_kernels", "list", map[string]any{}},
				{"kernel_status", "status", map[string]any{"name": "kernel", "label": "agent"}},
				{"set_check", "set_check", map[string]any{"name": "kernel", "label": "agent", "checker": "cpu_reference"}},
				{"set_launch", "set_geometry", map[string]any{"name": "kernel", "label": "agent", "geometry": geometry}},
				{"record_fixture", "record_fixture", map[string]any{"name": "kernel", "label": "agent", "stream": "main", "geometry": geometry, "args": []any{}, "buffers": []any{}, "outputs": []any{}}},
				{"select_version", "select", map[string]any{"name": "kernel", "label": "agent", "version": 1}},
				{"wait_for_report", "events", map[string]any{"name": "kernel", "since": 2, "timeout_seconds": 1}},
			} {
				res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: tc.name, Arguments: tc.args})
				if err != nil {
					t.Fatalf("%s: %v", tc.name, err)
				}
				if res.IsError != (outcome != cudahotswap.OutcomeOK) {
					t.Fatalf("%s outcome %s isError %v", tc.name, outcome, res.IsError)
				}
				select {
				case path := <-called:
					if path != "/v1/"+tc.endpoint {
						t.Fatalf("%s route %s", tc.name, path)
					}
				case <-ctx.Done():
					t.Fatalf("%s HTTP was not called: %+v", tc.name, res)
				}
			}
		})
	}
}

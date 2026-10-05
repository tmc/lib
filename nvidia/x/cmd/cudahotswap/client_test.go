package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

func TestClientCommands(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kernel.cu")
	if err := os.WriteFile(path, []byte("cuda bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, command := range []string{"push", "list", "revert", "status"} {
		t.Run(command, func(t *testing.T) {
			called := false
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				if r.URL.Path != "/v1/"+command {
					t.Error(r.URL.Path)
				}
				var q httpapi.Request
				if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
					t.Error(err)
				}
				if command == "push" && q.Source.Code != "cuda bytes" {
					t.Error(q)
				}
				json.NewEncoder(w).Encode(httpapi.Response{State: &cudahotswap.State{Generation: 3, ConfigRevision: 5}})
			}))
			defer server.Close()
			var out, stderr bytes.Buffer
			args := []string{command, "-addr", "tcp:" + strings.TrimPrefix(server.URL, "http://"), "-name", "kernel", "-label", "human"}
			if command == "push" {
				args = append(args, "-source", path)
			}
			if err := runClient(context.Background(), args, &out, &stderr); err != nil {
				t.Fatal(err)
			}
			if !called || !strings.Contains(out.String(), `"Generation":3`) {
				t.Fatalf("call %v output %s", called, out.String())
			}
		})
	}
}

func TestLocalWatchUsesChangedBytes(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	path := filepath.Join(t.TempDir(), "kernel.cu")
	os.WriteFile(path, []byte("first"), 0600)
	var sources []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var q httpapi.Request
		json.NewDecoder(r.Body).Decode(&q)
		sources = append(sources, q.Source.Code)
		if len(sources) == 1 {
			os.WriteFile(path, []byte("second"), 0600)
		} else {
			cancel()
		}
		json.NewEncoder(w).Encode(httpapi.Response{})
	}))
	defer server.Close()
	err := runClient(ctx, []string{"-addr", "tcp:" + strings.TrimPrefix(server.URL, "http://"), "-watch", path}, new(bytes.Buffer), new(bytes.Buffer))
	if err != context.Canceled {
		t.Fatal(err)
	}
	if len(sources) != 2 || sources[0] != "first" || sources[1] != "second" {
		t.Fatal(sources)
	}
}

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

func nativeMCPCommandClosed(command *exec.Cmd) error {
	if command == nil || command.Process == nil || command.Process.Pid <= 1 || command.ProcessState == nil || !command.ProcessState.Exited() || command.ProcessState.ExitCode() != 0 || command.Process.Pid != command.ProcessState.Pid() {
		return errors.New("MCP control: actual command did not exit successfully")
	}
	return nil
}

// The CPU child serves the actual SDK stdio adapter against a host fake HTTP
// response. It initializes no CUDA and is not native transport evidence.
func TestNativeMCPStdioChildCPU(t *testing.T) {
	if os.Getenv("CUDAHOTSWAP_CPU_MCP_CHILD") != "1" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server := newServer(&httpapi.Client{URL: os.Getenv("CUDAHOTSWAP_CPU_MCP_URL"), HTTP: &http.Client{Timeout: time.Second}})
	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

func TestNativeMCPCommandRetirementCPU(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var requests atomic.Int32
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/list" {
			http.Error(w, "unexpected route", http.StatusBadRequest)
			return
		}
		requests.Add(1)
		if err := json.NewEncoder(w).Encode(httpapi.Response{}); err != nil {
			t.Error(err)
		}
	}))
	defer httpServer.Close()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestNativeMCPStdioChildCPU$")
	command.Env = append(os.Environ(), "CUDAHOTSWAP_CPU_MCP_CHILD=1", "CUDAHOTSWAP_CPU_MCP_URL="+httpServer.URL)
	if nativeMCPCommandClosed(nil) == nil || nativeMCPCommandClosed(command) == nil {
		t.Fatal("unstarted command reported closed")
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "CPU-retirement", Version: "1"}, nil)
	session, err := client.Connect(ctx, &mcp.CommandTransport{Command: command, TerminateDuration: 3 * time.Second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := session.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	if nativeMCPCommandClosed(command) == nil {
		t.Fatal("live SDK command reported closed")
	}
	result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "list_kernels", Arguments: map[string]any{}})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError || requests.Load() != 1 {
		t.Fatalf("list_kernels: error=%v HTTP requests=%d", result.IsError, requests.Load())
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	if err := nativeMCPCommandClosed(command); err != nil {
		t.Fatal(err)
	}
	actualPID := command.Process.Pid
	command.Process.Pid = actualPID + 1
	if nativeMCPCommandClosed(command) == nil {
		t.Error("mismatched exited identity accepted")
	}
	command.Process.Pid = actualPID
	if command.Process.Pid == os.Getpid() {
		t.Fatal("actual SDK child identity missing")
	}
}

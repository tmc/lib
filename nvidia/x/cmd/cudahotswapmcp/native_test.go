//go:build native

package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// This consumer runs on the Mac. The operator starts the demo on DGX and owns
// the SSH forward. A finite server launched under the real GPU-lock holder
// inherits its verified holder PID and directory for nested request leases.
// An unrelated outer holder without that inheritance would block the server.
func TestNativeMacMCPWrongThenCorrect(t *testing.T) {
	if os.Getenv("CUDAHOTSWAP_NATIVE") != "1" {
		t.Skip("set CUDAHOTSWAP_NATIVE=1 after explicit native authorization")
	}
	if runtime.GOOS != "darwin" {
		t.Fatal("Mac MCP consumer is required")
	}
	addr, binary, artifacts := os.Getenv("CUDAHOTSWAP_ADDR"), os.Getenv("CUDAHOTSWAP_MCP_BINARY"), os.Getenv("CUDAHOTSWAP_ARTIFACTS")
	if !strings.HasPrefix(addr, "tcp:127.0.0.1:") || !filepath.IsAbs(binary) || !filepath.IsAbs(artifacts) {
		t.Fatal("loopback SSH forward, absolute MCP binary and artifact directory are required")
	}
	if err := os.MkdirAll(artifacts, 0700); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(body)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "Mac-native-consumer", Version: "1"}, nil)
	session, err := client.Connect(ctx, &mcp.CommandTransport{Command: exec.CommandContext(ctx, binary, "-addr", addr), TerminateDuration: time.Second}, nil)
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
	wrong := `extern "C" __global__ void saxpy(const float*x,const float*y,float*out,float a,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a*x[i]-y[i];}`
	correct := `extern "C" __global__ void saxpy(const float*x,const float*y,float*out,float a,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a*x[i]+y[i];}`
	status := func(filename string) cudahotswap.State {
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "kernel_status", Arguments: nameInput{Name: "saxpy"}})
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(res)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(artifacts, filename), raw, 0600); err != nil {
			t.Fatal(err)
		}
		reply, err := nativeMCPResponse(res)
		if err != nil || res.IsError || reply.Failure != nil || reply.State == nil {
			t.Fatalf("actual MCP status: %+v: %v", res, err)
		}
		return *reply.State
	}
	for i, source := range []string{wrong, correct} {
		prior := status([]string{"wrong-before.json", "correct-before.json"}[i])
		res, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "push_kernel", Arguments: pushInput{Name: "saxpy", Label: "Mac native control", Source: sourceInput{Code: source, Entry: "saxpy", Filename: "native-control.cu"}}})
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(res)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(artifacts, []string{"wrong.json", "correct.json"}[i]), raw, 0600); err != nil {
			t.Fatal(err)
		}
		current := status([]string{"wrong-after.json", "correct-after.json"}[i])
		if err := nativeMCPPushCheck(prior, current, res, source, "Mac native control", i == 0); err != nil {
			t.Fatalf("control %d: %v", i, err)
		}
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	metadata, _ := json.Marshal(map[string]any{"mcp_binary_sha256": hex.EncodeToString(digest[:]), "address": addr, "client_os": runtime.GOOS, "client_arch": runtime.GOARCH, "controls_passed": true, "mcp_process_closed": true})
	if err := os.WriteFile(filepath.Join(artifacts, "client.json"), metadata, 0600); err != nil {
		t.Fatal(err)
	}
}

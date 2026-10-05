// Command cudahotswapmcp proxies MCP tools to the hot-swap HTTP service.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

type proxy interface {
	Call(context.Context, string, httpapi.Request) (httpapi.Response, error)
	Events(context.Context, uint64) (*httpapi.EventStream, error)
}

type nameInput struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}
type sourceInput struct {
	Code     string   `json:"code"`
	Entry    string   `json:"entry"`
	Filename string   `json:"filename,omitempty"`
	Flags    []string `json:"flags,omitempty"`
}
type pushInput struct {
	Name   string      `json:"name"`
	Label  string      `json:"label"`
	Source sourceInput `json:"source"`
}
type selectInput struct {
	Name    string `json:"name"`
	Label   string `json:"label"`
	Version uint64 `json:"version"`
}
type checkInput struct {
	Name          string                    `json:"name"`
	Label         string                    `json:"label"`
	Checker       string                    `json:"checker,omitempty"`
	Fixture       string                    `json:"fixture,omitempty"`
	Policy        *cudahotswap.ReplayPolicy `json:"policy,omitempty"`
	KeepOnFailure bool                      `json:"keep_on_failure,omitempty"`
}
type launchInput struct {
	Name     string                      `json:"name"`
	Label    string                      `json:"label"`
	Geometry *cudahotswap.LaunchGeometry `json:"geometry"`
}
type fixtureInput struct {
	Name     string                      `json:"name"`
	Label    string                      `json:"label"`
	Stream   string                      `json:"stream"`
	Geometry *cudahotswap.LaunchGeometry `json:"geometry"`
	Args     []httpapi.Argument          `json:"args"`
	Buffers  []httpapi.Buffer            `json:"buffers"`
	Outputs  []cudahotswap.FixtureOutput `json:"outputs"`
	Policy   *cudahotswap.ReplayPolicy   `json:"policy,omitempty"`
}
type waitInput struct {
	Name           string `json:"name"`
	Since          uint64 `json:"since"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}
type emptyInput struct{}

func result(value any, failed bool, err error) (*mcp.CallToolResult, any, error) {
	if err != nil {
		value = map[string]any{"response": value, "error": err.Error()}
		failed = true
	}
	b, e := json.Marshal(value)
	if e != nil {
		return nil, nil, e
	}
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: string(b)}}, IsError: failed}, nil, nil
}

func call(ctx context.Context, p proxy, method string, q httpapi.Request) (*mcp.CallToolResult, any, error) {
	r, err := p.Call(ctx, method, q)
	failed := r.Failure != nil || r.Push != nil && r.Push.Report.Outcome != cudahotswap.OutcomeOK
	return result(r, failed, err)
}

func newServer(p proxy) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "cudahotswap", Version: "1"}, nil)
	mcp.AddTool(s, &mcp.Tool{Name: "push_kernel", Description: "Push CUDA source and return the classified check report."}, func(ctx context.Context, _ *mcp.CallToolRequest, q pushInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "push", httpapi.Request{Name: q.Name, Label: q.Label, Source: cudahotswap.Source{Code: q.Source.Code, Entry: q.Source.Entry, Filename: q.Source.Filename, Flags: q.Source.Flags}})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "revert_kernel", Description: "Restore the original retained kernel."}, func(ctx context.Context, _ *mcp.CallToolRequest, q nameInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "revert", httpapi.Request{Name: q.Name, Label: q.Label})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "list_kernels", Description: "List registered service kernels and their revisions."}, func(ctx context.Context, _ *mcp.CallToolRequest, q emptyInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "list", httpapi.Request{})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "kernel_status", Description: "Read one kernel state."}, func(ctx context.Context, _ *mcp.CallToolRequest, q nameInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "status", httpapi.Request{Name: q.Name})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "set_check", Description: "Select a named server checker or retained fixture and replay policy."}, func(ctx context.Context, _ *mcp.CallToolRequest, q checkInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "set_check", httpapi.Request{Name: q.Name, Label: q.Label, Checker: q.Checker, Fixture: q.Fixture, Policy: q.Policy, KeepOnFailure: q.KeepOnFailure})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "set_launch", Description: "Set the kernel launch geometry override."}, func(ctx context.Context, _ *mcp.CallToolRequest, q launchInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "set_geometry", httpapi.Request{Name: q.Name, Label: q.Label, Geometry: q.Geometry})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "record_fixture", Description: "Record a launch using named server buffers and stream; returns a retained fixture handle."}, func(ctx context.Context, _ *mcp.CallToolRequest, q fixtureInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "record_fixture", httpapi.Request{Name: q.Name, Label: q.Label, Stream: q.Stream, Geometry: q.Geometry, Args: q.Args, Buffers: q.Buffers, Outputs: q.Outputs, Policy: q.Policy})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "select_version", Description: "Select an already retained kernel version."}, func(ctx context.Context, _ *mcp.CallToolRequest, q selectInput) (*mcp.CallToolResult, any, error) {
		return call(ctx, p, "select", httpapi.Request{Name: q.Name, Label: q.Label, Version: q.Version})
	})
	mcp.AddTool(s, &mcp.Tool{Name: "wait_for_report", Description: "Wait for a retained report after a cursor; gaps and timeouts are errors."}, func(ctx context.Context, _ *mcp.CallToolRequest, q waitInput) (*mcp.CallToolResult, any, error) {
		if q.TimeoutSeconds < 1 || q.TimeoutSeconds > 120 {
			return result(nil, true, errors.New("timeout_seconds must be in [1,120]"))
		}
		ctx, cancel := context.WithTimeout(ctx, time.Duration(q.TimeoutSeconds)*time.Second)
		defer cancel()
		stream, err := p.Events(ctx, q.Since)
		if err != nil {
			return result(nil, true, err)
		}
		defer stream.Close()
		for {
			frame, err := stream.Read()
			if err != nil {
				return result(frame, true, err)
			}
			if frame.Result != nil {
				return result(frame, true, errors.New("event stream ended with service failure"))
			}
			e := frame.Event
			if e.Gap != nil {
				return result(frame, true, errors.New("requested report range was lost"))
			}
			if q.Name != "" && e.Name != q.Name {
				continue
			}
			if e.Report == nil {
				return result(frame, true, errors.New("event has no report"))
			}
			return result(frame, e.Report.Outcome != cudahotswap.OutcomeOK, nil)
		}
	})
	return s
}

func main() {
	addr := flag.String("addr", "unix:/var/run/cudahotswap.sock", "unix:path or tcp:host:port")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	p, err := httpapi.NewClient(*addr)
	if err == nil {
		err = newServer(p).Run(ctx, &mcp.StdioTransport{})
	}
	if err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, "cudahotswapmcp:", err)
		os.Exit(1)
	}
}

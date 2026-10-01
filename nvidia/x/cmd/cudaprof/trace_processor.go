package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type traceProcessorReceipt struct {
	Path           string     `json:"path"`
	SHA256         string     `json:"sha256"`
	Version        string     `json:"version"`
	Query          string     `json:"query"`
	Rows           [][]string `json:"rows,omitempty"`
	Warnings       string     `json:"warnings,omitempty"`
	TimeoutSeconds int        `json:"timeout_seconds"`
	OutputBytes    int        `json:"output_byte_limit"`
}

type traceBoundedOutput struct {
	bytes.Buffer
	limit int
}

func (b *traceBoundedOutput) Write(p []byte) (int, error) {
	if len(p) > b.limit-b.Len() {
		return 0, errors.New("trace processor output limit exceeded")
	}
	return b.Buffer.Write(p)
}

// queryTraceProcessor only invokes local v58.2, whose query CSV schema is
// explicitly supported. It uses a snapshot of the hashed input, never a URL.
type traceBackendUnavailable struct{ reason string }

func (e *traceBackendUnavailable) Error() string { return e.reason }

type traceProcessorLimits struct {
	Timeout           time.Duration
	OutputBytes, Rows int
}

func defaultTraceProcessorLimits() traceProcessorLimits {
	return traceProcessorLimits{30 * time.Second, 4 << 20, 10000}
}

func queryTraceProcessor(data []byte, configured, sql string) (traceProcessorReceipt, error) {
	return queryTraceProcessorLimits(data, configured, sql, defaultTraceProcessorLimits())
}
func queryTraceProcessorLimits(data []byte, configured, sql string, limits traceProcessorLimits) (traceProcessorReceipt, error) {
	r := traceProcessorReceipt{Query: sql, TimeoutSeconds: int(limits.Timeout / time.Second), OutputBytes: limits.OutputBytes}
	if limits.Timeout <= 0 || limits.OutputBytes <= 0 || limits.Rows <= 0 {
		return r, errors.New("trace processor: resource limits must be positive")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	name := configured
	if name == "" {
		name = os.Getenv("CUDAPROF_TRACE_PROCESSOR")
	}
	if name == "" {
		name = "trace_processor_shell"
	}
	path, err := exec.LookPath(name)
	if err != nil {
		return r, &traceBackendUnavailable{fmt.Sprintf("trace processor unavailable: %v", err)}
	}
	r.Path, err = filepath.Abs(path)
	if err != nil {
		return r, err
	}
	exe, err := os.Open(r.Path)
	if err != nil {
		return r, err
	}
	h := sha256.New()
	_, err = io.Copy(h, exe)
	closeErr := exe.Close()
	if err != nil {
		return r, err
	}
	if closeErr != nil {
		return r, closeErr
	}
	r.SHA256 = hex.EncodeToString(h.Sum(nil))
	run := func(args ...string) (string, string, error) {
		ctx, cancel := context.WithTimeout(ctx, limits.Timeout)
		defer cancel()
		cmd := exec.CommandContext(ctx, r.Path, args...)
		cmd.WaitDelay = time.Second
		out := traceBoundedOutput{limit: r.OutputBytes}
		stderr := traceBoundedOutput{limit: r.OutputBytes}
		cmd.Stdout = &out
		cmd.Stderr = &stderr
		err := cmd.Run()
		if ctx.Err() != nil {
			err = ctx.Err()
		}
		return out.String(), stderr.String(), err
	}
	version, warnings, err := run("--version")
	r.Version = strings.TrimSpace(version)
	r.Warnings = warnings
	if err != nil {
		return r, fmt.Errorf("trace processor version: %w", err)
	}
	if !strings.HasPrefix(r.Version, "Perfetto v58.2") && !strings.HasPrefix(r.Version, "v58.2") {
		return r, &traceBackendUnavailable{fmt.Sprintf("trace processor version %q unsupported; supported version is v58.2", r.Version)}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return r, err
	}
	scratch := filepath.Join(home, "tmp", "cudaprof-trace")
	if err = os.MkdirAll(scratch, 0700); err != nil {
		return r, err
	}
	f, err := os.CreateTemp(scratch, "input-*.pftrace")
	if err != nil {
		return r, err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err != nil {
		f.Close()
		return r, err
	}
	if err = f.Close(); err != nil {
		return r, err
	}
	query, err := os.CreateTemp(scratch, "query-*.sql")
	if err != nil {
		return r, err
	}
	defer os.Remove(query.Name())
	if _, err := io.WriteString(query, sql); err != nil {
		query.Close()
		return r, err
	}
	if err := query.Close(); err != nil {
		return r, err
	}
	output, warnings, err := run("query", "-f", query.Name(), f.Name())
	r.Warnings += warnings
	if err != nil {
		return r, fmt.Errorf("trace processor query: %w: %s", err, strings.TrimSpace(warnings))
	}
	reader := csv.NewReader(strings.NewReader(output))
	reader.FieldsPerRecord = -1
	r.Rows, err = reader.ReadAll()
	if err != nil {
		return r, fmt.Errorf("trace processor CSV: %w", err)
	}
	if len(r.Rows) > limits.Rows {
		return r, errors.New("trace processor row limit exceeded")
	}
	return r, nil
}

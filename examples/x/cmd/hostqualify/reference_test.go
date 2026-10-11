package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReadReference reads a row by header name, so that the Metal
// reference (six columns) and the CUDA reference (nine) both parse.
func TestReadReference(t *testing.T) {
	const metal = `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
2048,2048,343.76,128,26.76,52184460
4096,2048,303.46,128,26.52,80373132
`
	const cuda = `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes
2048,2048,825.76,128,18.05,71.721,127,18.20,52184460
`
	for _, tt := range []struct {
		name        string
		data        string
		ctx         int
		wantPrefill float64
		wantGen     float64
		wantErr     string
	}{
		{name: "metal", data: metal, ctx: 2048, wantPrefill: 343.76, wantGen: 26.76},
		{name: "metal later row", data: metal, ctx: 4096, wantPrefill: 303.46, wantGen: 26.52},
		{name: "cuda", data: cuda, ctx: 2048, wantPrefill: 825.76, wantGen: 18.05},
		{name: "missing ctx", data: metal, ctx: 8192, wantErr: "no row for ctx 8192"},
		{name: "wrong columns", data: "a,b\n1,2\n", ctx: 2048, wantErr: "no ctx_tokens column"},
		{name: "no rows", data: "ctx_tokens,prefill_tps,gen_tps\n", ctx: 2048, wantErr: "has no rows"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "ref.csv")
			if err := os.WriteFile(path, []byte(tt.data), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := readReference(path, tt.ctx)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("readReference = %v, want error containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.prefill != tt.wantPrefill || got.gen != tt.wantGen {
				t.Errorf("readReference = %+v, want prefill %v gen %v", got, tt.wantPrefill, tt.wantGen)
			}
		})
	}
}

// TestRunGate checks the gate's verdict, which a measurement script
// reads as an exit status.
func TestRunGate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ref.csv")
	if err := os.WriteFile(path, []byte("ctx_tokens,prefill_tps,gen_tps\n2048,343.76,26.76\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		args []string
		want bool // degraded
	}{
		{name: "quiet", args: []string{"-max-load", "1000"}},
		{name: "busy", args: []string{"-max-load", "0.0001"}, want: true},
		{name: "matching rates", args: []string{"-max-load", "1000",
			"-reference", path, "-ctx", "2048", "-prefill", "340", "-gen", "26"}},
		{name: "third of reference", args: []string{"-max-load", "1000",
			"-reference", path, "-ctx", "2048", "-prefill", "109.66", "-gen", "10.99"}, want: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := run(tt.args, io.Discard, io.Discard)
			if got := errors.Is(err, errDegraded); got != tt.want {
				t.Errorf("run(%v) = %v, want degraded = %v", tt.args, err, tt.want)
			}
		})
	}
}

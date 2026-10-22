//go:build darwin && arm64

package main

import (
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

func TestMetalStreamingPreloadCount(t *testing.T) {
	for _, tt := range []struct {
		env       string
		requested uint
		want      int
	}{
		{"", 0, 4096}, {"0", 0, 43 * 256}, {"12", 0, 12},
		{"  +12", 0, 12}, {"12x", 0, 4096}, {"12 ", 0, 4096},
		{"-1", 0, 43 * 256}, {"18446744073709551616", 0, 43 * 256},
		{"-18446744073709551616", 0, 43 * 256}, {"+", 0, 4096},
		{"1", 5000, 5000}, {"0", 20000, 43 * 256},
	} {
		if got := metalStreamingPreloadCount(43*256, tt.requested, tt.env); got != tt.want {
			t.Errorf("env=%q requested=%d: got %d, want %d", tt.env, tt.requested, got, tt.want)
		}
		if oracle := os.Getenv("METAL_PRELOAD_ORACLE"); oracle != "" {
			cmd := exec.Command(oracle, strconv.FormatUint(uint64(tt.requested), 10))
			cmd.Env = append(os.Environ(), "DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP="+tt.env)
			out, err := cmd.Output()
			if err != nil {
				t.Fatal(err)
			}
			want := strconv.Itoa(metalStreamingPreloadCount(43*256, tt.requested, tt.env))
			if strings.TrimSpace(string(out)) != want {
				t.Fatalf("C preload %q requested=%d: %s vs %s", tt.env, tt.requested, out, want)
			}
		}
	}
}

func TestMetalStreamingDominant(t *testing.T) {
	makeTriplet := func(n uint64) dsv4expertcache.Triplet {
		return dsv4expertcache.Triplet{
			Gate: dsv4expertcache.Tensor{Stride: n, Experts: 256},
			Up:   dsv4expertcache.Tensor{Stride: n, Experts: 256},
			Down: dsv4expertcache.Tensor{Stride: n, Experts: 256},
		}
	}
	for _, tt := range []struct {
		name  string
		sizes []uint64
		want  uint64
		count int
		fail  bool
	}{
		{"outlier-first", []uint64{2, 1, 1}, 1, 2, false},
		{"tie-earliest", []uint64{2, 1}, 2, 1, false},
		{"overflow", []uint64{math.MaxUint64/2 + 1}, 0, 0, true},
		{"empty", nil, 0, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tr := make(map[int]dsv4expertcache.Triplet)
			for i, n := range tt.sizes {
				tr[i] = makeTriplet(n)
			}
			base, count, err := metalStreamingDominant(tr, len(tt.sizes))
			if (err != nil) != tt.fail || (!tt.fail && (base.Gate.Stride != tt.want || count != tt.count)) {
				t.Fatal("dominant class", base, count, err)
			}
		})
	}
}

func TestMetalStreamingContextAdmission(t *testing.T) {
	for _, name := range []string{"DS4_METAL_PREFILL_CHUNK", "DS4_METAL_GRAPH_RAW_CAP"} {
		t.Setenv(name, "")
	}
	if _, err := metalStreamingContextBytes(0); err == nil {
		t.Fatal("zero context admitted")
	}
	a, err := metalStreamingContextBytes(4096)
	if err != nil || a == 0 {
		t.Fatal(a, err)
	}
	b, err := metalStreamingContextBytes(8192)
	if err != nil || b <= a {
		t.Fatal("context reserve did not grow", a, b, err)
	}
	t.Setenv("DS4_METAL_GRAPH_RAW_CAP", "256")
	c, err := metalStreamingContextBytes(4096)
	if err != nil || a-c != 43*(4096-256)*512*4 {
		t.Fatal("raw geometry override", a, c, err)
	}
}

func TestMetalStreamingContextOracle(t *testing.T) {
	path := os.Getenv("DS4_TEST_METAL_CONTEXT_ORACLE")
	if path == "" {
		t.Skip("pinned C context oracle not supplied")
	}
	for _, tt := range []struct {
		context      int
		prefill, raw string
	}{
		{1, "", ""}, {4096, "", ""}, {8192, "", ""},
		{8192, "1024junk", ""}, {8192, "0suffix", ""}, {8192, "-1", ""},
		{8192, "nonnumeric", ""}, {8192, "+64", ""}, {8192, " 32 ", ""},
		{8192, "4294967296", ""}, {8192, "999999999999999999999999", ""},
		{4096, "", "256suffix"}, {4096, "", "0"}, {4096, "", "-1"},
		{4096, "", "4294967296"}, {4096, "", "4294967552"},
		{8192, "", "999999999999999999999999"},
		{8192, "999999999999999999999999", "999999999999999999999999"},
		{8192, "-999999999999999999999999", "-999999999999999999999999"},
	} {
		t.Setenv("DS4_METAL_PREFILL_CHUNK", tt.prefill)
		t.Setenv("DS4_METAL_GRAPH_RAW_CAP", tt.raw)
		out, err := exec.Command(path, strconv.Itoa(tt.context)).Output()
		if err != nil {
			t.Fatal(err)
		}
		want, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		got, err := metalStreamingContextBytes(tt.context)
		if err != nil || got != want {
			t.Fatalf("ctx=%d prefill=%q raw=%q got=%d want=%d err=%v", tt.context, tt.prefill, tt.raw, got, want, err)
		}
	}
}

func TestMetalStreamingConstructorAdmission(t *testing.T) {
	r, err := newMetalRuntimeStreamingLogged("not-opened", 2, nil, streamingOptions{cold: true})
	if r != nil || err == nil || !strings.Contains(err.Error(), "require ssd-streaming") {
		t.Fatal("orphan options reached model/device construction", r, err)
	}
	r, err = newMetalRuntimeStreamingLogged("not-opened", 2, nil, streamingOptions{enabled: true, budget: "0"})
	if r != nil || err == nil || !strings.Contains(err.Error(), "invalid cache budget") {
		t.Fatal("invalid budget reached model/device construction", r, err)
	}
}

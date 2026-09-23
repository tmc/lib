package cudaprobe

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestDemanglerBatchCacheAndFallback(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cu++filt")
	script := `#!/bin/sh
printf 'call\n' >> "$0.calls"
if [ "$1" = '-v' ]; then printf 'fixture tool 1\n'; exit 0; fi
while IFS= read -r symbol; do
  case "$symbol" in
    _Z1fv) printf 'f()\n' ;;
    _Z1gv) printf 'g()\n' ;;
    *) printf '%s\n' "$symbol" ;;
  esac
done
`
	if err := os.WriteFile(path, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	d, err := NewDemangler(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	names, receipt, err := d.Batch(context.Background(), []string{"_Z1fv", "_Z1fv"})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Path != path || receipt.Version != "fixture tool 1" || len(names) != 2 || names[0].Display != "f()" || !names[1].Demangled {
		t.Fatalf("names=%v receipt=%+v", names, receipt)
	}
	if _, _, err := d.Batch(context.Background(), []string{"_Z1fv"}); err != nil {
		t.Fatal(err)
	}
	calls, err := os.ReadFile(path + ".calls")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(calls), "call") != 2 {
		t.Fatalf("calls=%q", calls)
	}
	input := []cudaflightrecorder.Record{{KernelName: "_Z1fv", Labels: map[string]string{"source": "fixture"}}, {Labels: map[string]string{"kernel_identity": "unavailable"}}}
	records, err := d.Records(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if records[0].KernelName != "f()" || records[0].Labels["kernel.symbol"] != "_Z1fv" || records[0].Labels["demangler.path"] != path || records[0].Labels["demangler.version"] != "fixture tool 1" || records[0].Labels["demangler.status"] != "demangled" {
		t.Fatalf("record=%+v", records[0])
	}
	if input[0].KernelName != "_Z1fv" || len(input[0].Labels) != 1 || records[1].KernelName != "" {
		t.Fatal("mutated input or invented name")
	}
	records[0].Labels["demangler.failure"] = "old failure"
	records, err = d.Records(context.Background(), records)
	if err != nil || records[0].Labels["kernel.symbol"] != "_Z1fv" || records[0].Labels["demangler.failure"] != "" {
		t.Fatalf("stale receipt survived: %+v %v", records, err)
	}
	// One cached and one new input; subsequent churn must remain bounded.
	if _, _, err := d.Batch(context.Background(), []string{"_Z1fv", "_Z1gv"}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.Batch(context.Background(), []string{"plain", "other"}); err != nil {
		t.Fatal(err)
	}
	if len(d.cache) > 2 || len(d.order) > 2 {
		t.Fatal("cache exceeded capacity")
	}
	missing, _ := NewDemangler(filepath.Join(t.TempDir(), "missing"), 2)
	names, receipt, err = missing.Batch(context.Background(), []string{"_Z1fv"})
	if err == nil || receipt.Failure == "" || names[0].Display != "_Z1fv" || names[0].Demangled {
		t.Fatalf("fallback=%v receipt=%+v error=%v", names, receipt, err)
	}
	records, err = missing.Records(context.Background(), input)
	if err == nil || records[0].KernelName != "_Z1fv" || records[0].Labels["demangler.status"] != "unavailable" || records[0].Labels["demangler.failure"] == "" {
		t.Fatalf("record fallback=%+v error=%v", records, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := d.Batch(ctx, []string{"plain"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	if _, _, err := d.Batch(context.Background(), []string{"a\nb"}); err == nil {
		t.Fatal("accepted line separator")
	}
	if _, _, err := d.Batch(context.Background(), []string{"a", "b", "c"}); err == nil {
		t.Fatal("accepted oversized batch")
	}
}

func TestDemanglerRejectsPartialOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tool")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nif [ \"$1\" = '-v' ]; then echo fixture; else echo one; fi\n"), 0700); err != nil {
		t.Fatal(err)
	}
	d, _ := NewDemangler(path, 2)
	names, _, err := d.Batch(context.Background(), []string{"_Z1fv", "_Z1gv"})
	if err == nil || names[0].Demangled || names[1].Demangled || len(d.cache) != 0 {
		t.Fatalf("partial output accepted: %v %v", names, err)
	}
}

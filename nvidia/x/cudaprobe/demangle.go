package cudaprobe

import (
	"context"
	"fmt"
	"maps"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Name retains the original symbol even when the toolkit cannot demangle it.
// Demangled is true only when cu++filt returned a different nonempty name.
type Name struct {
	Symbol, Display string
	Demangled       bool
}

// Records demangles the identified kernel names in a drained batch, returning
// copies with toolkit receipts in their labels. It does not read from Source
// or push to the recorder. Unidentified records remain unidentified. On failure
// names are retained and the returned records carry the failure reason.
func (d *Demangler) Records(ctx context.Context, records []cudaflightrecorder.Record) ([]cudaflightrecorder.Record, error) {
	if d == nil || d.cache == nil || len(records) > d.capacity {
		return nil, fmt.Errorf("cudaprobe: uninitialized demangler or record batch exceeds capacity")
	}
	var symbols []string
	for _, r := range records {
		if r.KernelName != "" {
			symbol := r.Labels["kernel.symbol"]
			if symbol == "" {
				symbol = r.KernelName
			}
			symbols = append(symbols, symbol)
		}
	}
	names, receipt, err := d.Batch(ctx, symbols)
	out := make([]cudaflightrecorder.Record, len(records))
	n := 0
	for i, r := range records {
		r.Labels = maps.Clone(r.Labels)
		if r.Labels == nil {
			r.Labels = make(map[string]string)
		}
		r.Labels["demangler.path"] = receipt.Path
		r.Labels["demangler.version"] = receipt.Version
		r.Labels["demangler.status"] = "unavailable"
		delete(r.Labels, "demangler.failure")
		if err != nil {
			r.Labels["demangler.failure"] = err.Error()
		}
		if r.KernelName != "" {
			r.Labels["kernel.symbol"] = symbols[n]
			if n < len(names) {
				r.KernelName = names[n].Display
				if names[n].Demangled {
					r.Labels["demangler.status"] = "demangled"
				} else if err == nil {
					r.Labels["demangler.status"] = "unchanged"
				}
			}
			n++
		}
		out[i] = r
	}
	return out, err
}

// DemangleReceipt identifies the actual toolkit executable and its version.
// A failure leaves symbols unchanged and records why. Results are not evidence
// that the corresponding kernel was loaded or executed.
type DemangleReceipt struct{ Path, Version, Failure string }

// Demangler batches cu++filt calls through cudasass and retains a bounded cache.
// It is not safe for concurrent use. Call Batch after draining capture records;
// it executes an external process and must not be part of the drain loop.
type Demangler struct {
	tool     cudasass.Demangler
	capacity int
	cache    map[string]string
	order    []string
	next     int
	receipt  DemangleReceipt
}

// NewDemangler creates a cache of at most capacity symbols. path selects
// cu++filt; an empty path uses the CUDA toolkit lookup in cudasass. Each batch
// admits at most capacity names of at most 4096 bytes each.
func NewDemangler(path string, capacity int) (*Demangler, error) {
	if capacity <= 0 {
		return nil, fmt.Errorf("cudaprobe: invalid demangler capacity")
	}
	return &Demangler{tool: cudasass.Demangler{Path: path}, capacity: capacity, cache: make(map[string]string)}, nil
}

// Batch returns one name for each input, preserving order and duplicates.
// On a toolkit error it returns unchanged names for uncached symbols, a failure
// receipt and the error. Failures are not cached. Cancellation is propagated.
func (d *Demangler) Batch(ctx context.Context, symbols []string) ([]Name, DemangleReceipt, error) {
	if d == nil || d.cache == nil || len(symbols) > d.capacity {
		return nil, DemangleReceipt{}, fmt.Errorf("cudaprobe: uninitialized demangler or batch exceeds capacity")
	}
	for _, s := range symbols {
		if s == "" || len(s) > 4096 || strings.ContainsAny(s, "\x00\r\n") {
			return nil, d.receipt, fmt.Errorf("cudaprobe: invalid kernel symbol")
		}
	}
	names := make([]Name, len(symbols))
	var pending []string
	seen := make(map[string]bool)
	for i, s := range symbols {
		v, ok := d.cache[s]
		if !ok {
			v = s
			if !seen[s] {
				pending = append(pending, s)
				seen[s] = true
			}
		}
		names[i] = Name{Symbol: s, Display: v, Demangled: v != s}
	}
	fail := func(err error) ([]Name, DemangleReceipt, error) {
		r := d.receipt
		r.Failure = err.Error()
		return names, r, err
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	if len(pending) == 0 {
		return names, d.receipt, nil
	}
	if d.receipt.Path == "" {
		out, err := d.tool.Version(ctx)
		if err != nil {
			names, r, err := fail(err)
			r.Path = out.Path
			return names, r, err
		}
		d.receipt = DemangleReceipt{Path: out.Path, Version: strings.TrimSpace(string(out.Stdout))}
		if d.receipt.Version == "" {
			d.receipt.Path = ""
			return fail(fmt.Errorf("cudaprobe: demangler version unavailable"))
		}
		d.tool.Path = out.Path
	}
	out, err := d.tool.Demangle(ctx, pending)
	if err != nil {
		return fail(err)
	}
	lines := strings.Split(strings.TrimSuffix(string(out.Stdout), "\n"), "\n")
	if len(lines) != len(pending) {
		return fail(fmt.Errorf("cudaprobe: demangler returned %d names for %d symbols", len(lines), len(pending)))
	}
	for _, s := range lines {
		if s == "" || len(s) > 4096 || strings.ContainsAny(s, "\x00\r") {
			return fail(fmt.Errorf("cudaprobe: invalid demangler output"))
		}
	}
	// Use a batch-local map: inserting a whole batch can evict earlier cached
	// inputs, but must not change this batch's results.
	results := make(map[string]string, len(pending))
	for i, s := range pending {
		results[s] = lines[i]
		if len(d.order) < d.capacity {
			d.order = append(d.order, s)
		} else {
			delete(d.cache, d.order[d.next])
			d.order[d.next] = s
			d.next = (d.next + 1) % d.capacity
		}
		d.cache[s] = lines[i]
	}
	for i := range names {
		if v, ok := results[names[i].Symbol]; ok {
			names[i].Display = v
			names[i].Demangled = v != names[i].Symbol
		}
	}
	return names, d.receipt, nil
}

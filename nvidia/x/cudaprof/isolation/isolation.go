package isolation

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

const MaxBytes = 1 << 20
const Attachment = "gpu-isolation-v1.json"

type Process struct {
	Host  string `json:"host"`
	PID   int    `json:"pid"`
	Start string `json:"start"`
}
type Window struct {
	Domain      string `json:"domain"`
	Start       int64  `json:"start"`
	End         int64  `json:"end"`
	Uncertainty int64  `json:"uncertainty_ns"`
}
type Binding struct {
	Scope           string   `json:"scope,omitempty"`
	Session         string   `json:"session"`
	Process         Process  `json:"process"`
	GPUs            []string `json:"gpu_uuids"`
	Window          Window   `json:"window"`
	CompletionKnown bool     `json:"completion_known"`
}
type Lock struct {
	Owner  Process `json:"owner"`
	Token  string  `json:"token"`
	Window Window  `json:"window"`
}
type Observation struct {
	QueryWindow      *Window `json:"query_window,omitempty"`
	Process          Process `json:"process"`
	GPU              string  `json:"gpu_uuid"`
	First            int64   `json:"first"`
	Last             int64   `json:"last"`
	ResidentBytes    uint64  `json:"resident_bytes"`
	Executable       string  `json:"executable,omitempty"`
	ExecutableSHA256 string  `json:"executable_sha256,omitempty"`
	ActiveKernel     bool    `json:"active_kernel,omitempty"`
}
type DeviceCoverage struct {
	GPU     string `json:"gpu_uuid"`
	Samples int    `json:"samples"`
	First   int64  `json:"first"`
	Last    int64  `json:"last"`
	MaxGap  int64  `json:"max_gap_ns"`
}

type Receipt struct {
	Devices           []DeviceCoverage `json:"devices"`
	Version           int              `json:"version"`
	Observer          string           `json:"observer"`
	ObserverVersion   string           `json:"observer_version"`
	Method            string           `json:"method"`
	Producer          string           `json:"producer"`
	Binding           Binding          `json:"binding"`
	Window            Window           `json:"window"`
	Lock              *Lock            `json:"lock,omitempty"`
	OwnershipMethod   string           `json:"ownership_method"`
	OwnershipResolved bool             `json:"ownership_resolved"`
	RequestedInterval int64            `json:"requested_interval_ns"`
	ObservedInterval  int64            `json:"observed_interval_ns"`
	Samples           int              `json:"samples"`
	BoundarySamples   bool             `json:"boundary_samples"`
	Gaps              []Window         `json:"gaps,omitempty"`
	Outside           []Observation    `json:"outside,omitempty"`
	Errors            []string         `json:"errors,omitempty"`
}
type Result struct {
	Version int      `json:"version"`
	State   string   `json:"state"`
	Reasons []string `json:"reasons"`
	SHA256  string   `json:"source_sha256,omitempty"`
	Receipt *Receipt `json:"receipt,omitempty"`
	Limits  string   `json:"limits"`
}

// Read reads exactly one bounded v1 receipt and hashes the bytes parsed.
func Read(r io.Reader) (*Receipt, string, error) {
	b, err := io.ReadAll(io.LimitReader(r, MaxBytes+1))
	if err != nil {
		return nil, "", fmt.Errorf("read isolation receipt: %w", err)
	}
	if len(b) > MaxBytes {
		return nil, "", fmt.Errorf("isolation receipt exceeds %d bytes", MaxBytes)
	}
	hash := sha256.Sum256(b)
	digest := hex.EncodeToString(hash[:])
	if err := uniqueJSON(b); err != nil {
		return nil, digest, err
	}
	var receipt Receipt
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&receipt); err != nil {
		return nil, digest, fmt.Errorf("decode isolation receipt: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return nil, digest, fmt.Errorf("isolation receipt contains trailing data")
	}
	if receipt.Version != 1 {
		return nil, digest, fmt.Errorf("unsupported isolation schema %d", receipt.Version)
	}
	return &receipt, digest, nil
}

// Evaluate binds a receipt to independently obtained capture identity. Missing
// evidence stays unchecked. Supported outside observations outrank errors.
func Evaluate(r *Receipt, digest string, want Binding) Result {
	out := Result{Version: 1, State: "unchecked", SHA256: digest, Receipt: r, Reasons: []string{}, Limits: "sampled observations may miss brief activity; residency does not quantify timing impact; local observer evidence is not cryptographic attestation"}
	reason := func(s string) { out.Reasons = append(out.Reasons, s) }
	if r == nil {
		reason("no bound observation receipt")
		return out
	}
	bound := r.Binding.Session != "" && r.Binding.Session == want.Session && r.Binding.Process == want.Process && want.Process.PID > 0 && want.Process.Host != "" && want.Process.Start != ""
	if !bound {
		reason("capture session or process generation is unbound")
	}
	devices := sameDevices(r.Binding.GPUs, want.GPUs)
	if !devices {
		reason("physical GPU identities are unavailable or mismatched")
	}
	clock := want.Window.Domain == "CLOCK_MONOTONIC" && want.Window.Start >= 0 && want.Window.Domain == r.Window.Domain && want.Window.Domain == r.Binding.Window.Domain && want.Window.Start <= want.Window.End
	if !clock {
		reason("capture and observer clocks are unbound")
	}
	window := r.Binding.Window == want.Window && r.Binding.Scope == want.Scope
	if !window {
		reason("capture window identity mismatch")
	}
	supportedMethod := r.Method == "nvml-process-residency" || r.Method == "nvidia-smi-compute-residency"
	observed := map[string]bool{}
	if bound && devices && clock && window && supportedMethod {
		for _, o := range r.Outside {
			if o.Process.PID <= 0 || o.Process.Start == "" || o.Process.Host != want.Process.Host {
				reason("outside process generation or host unavailable")
				continue
			}
			if o.QueryWindow != nil {
				q := o.QueryWindow
				if q.Domain != want.Window.Domain || q.Start > q.End {
					reason("outside observation query clock unavailable")
					continue
				}
				if q.Start < want.Window.Start || q.End > want.Window.End {
					if q.End >= want.Window.Start && q.Start <= want.Window.End {
						reason("outside observation overlaps capture boundary with query uncertainty")
					}
					continue
				}
			}
			if contains(want.GPUs, o.GPU) && o.First <= o.Last && o.Last >= want.Window.Start && o.First <= want.Window.End {
				out.State = "contaminated"
				key := fmt.Sprintf("%s:%d:%s:%s", o.Process.Host, o.Process.PID, o.Process.Start, o.GPU)
				if observed[key] {
					continue
				}
				observed[key] = true
				reason(fmt.Sprintf("outside process %d observed on %s; timing impact unknown", o.Process.PID, o.GPU))
			}
		}
	}
	if r.Version != 1 {
		reason("unsupported receipt version")
	}
	if r.Observer == "" || r.ObserverVersion == "" || (r.Method != "nvml-process-residency" && r.Method != "nvidia-smi-compute-residency") {
		reason("observer identity or method unavailable")
	}
	if !r.OwnershipResolved || r.OwnershipMethod != "live-process-generation-ancestry" {
		reason("ownership ancestry unresolved or unsupported")
	}
	if !want.CompletionKnown || !r.Binding.CompletionKnown {
		reason("final relevant GPU completion boundary unknown")
	}
	for _, gpu := range want.GPUs {
		var found *DeviceCoverage
		for i := range r.Devices {
			if r.Devices[i].GPU == gpu {
				if found != nil {
					reason("duplicate device coverage")
					break
				}
				found = &r.Devices[i]
			}
		}
		if found == nil || found.Samples < 2 || found.First > want.Window.Start || found.Last < want.Window.End || found.MaxGap <= 0 || found.MaxGap > r.ObservedInterval {
			reason("physical GPU sampling coverage unavailable: " + gpu)
		}
	}
	if r.Samples < 2 || !r.BoundarySamples || r.RequestedInterval <= 0 || r.ObservedInterval <= 0 {
		reason("continuous sampled window coverage unavailable")
	}
	if r.Window.Uncertainty != 0 || want.Window.Uncertainty != 0 || r.Window.Start > want.Window.Start || r.Window.End < want.Window.End {
		reason("observer does not cover capture boundaries and uncertainty")
	}
	if len(r.Gaps) > 0 {
		reason("observer coverage gaps retained")
	}
	for _, e := range r.Errors {
		reason("observer error: " + e)
	}
	if len(out.Reasons) == 0 {
		out.State = "clear"
		reason("sampled clear")
	}
	return out
}

// Require applies the common qualification policy without changing evidence.
func Require(r Result, clear bool) error {
	if r.State == "contaminated" || clear && r.State != "clear" {
		return fmt.Errorf("GPU isolation: %s", r.State)
	}
	return nil
}
func contains(a []string, s string) bool {
	for _, v := range a {
		if v == s {
			return true
		}
	}
	return false
}
func sameDevices(a, b []string) bool {
	if len(a) == 0 || len(a) != len(b) {
		return false
	}
	a = append([]string(nil), a...)
	b = append([]string(nil), b...)
	sort.Strings(a)
	sort.Strings(b)
	for i := range a {
		if a[i] == "" || a[i] != b[i] || i > 0 && a[i] == a[i-1] {
			return false
		}
	}
	return true
}

func uniqueJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	var value func() error
	value = func() error {
		token, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := map[string]bool{}
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return fmt.Errorf("duplicate isolation JSON key %q", name)
				}
				seen[name] = true
				if err := value(); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("invalid isolation JSON delimiter")
		}
		_, err = d.Token()
		return err
	}
	if err := value(); err != nil {
		return fmt.Errorf("validate isolation JSON: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("isolation receipt contains trailing data")
	}
	return nil
}

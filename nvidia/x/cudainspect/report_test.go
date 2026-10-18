package cudainspect

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"
)

// roundtripCUBIN is real nvcc 13.4 sm_121a output with kernels intmix and
// addk; its provenance is ../cudasass/testdata/roundtrip/README.md.
func roundtripCUBIN(t testing.TB) []byte {
	t.Helper()
	raw, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	b, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestReadReportHistogramCounts(t *testing.T) {
	listing, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.sass")
	if err != nil {
		t.Fatal(err)
	}
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	image := roundtripCUBIN(t)
	for _, tt := range []struct {
		name      string
		histogram string
		wantError bool
	}{
		{"unchanged", "", false},
		{"matching", `{"counted":3,"quarantined":0,"opcodes":{"NOP":1,"EXIT":2}}`, false},
		{"maximum", `{"counted":18446744073709551615,"quarantined":0,"opcodes":{"NOP":18446744073709551615}}`, false},
		{"mismatch", `{"counted":1,"quarantined":0,"opcodes":{"NOP":2}}`, true},
		{"overflow", `{"counted":1,"quarantined":0,"opcodes":{"NOP":18446744073709551615,"EXIT":2}}`, true},
		{"overflow_to_zero", `{"counted":0,"quarantined":0,"opcodes":{"NOP":18446744073709551615,"EXIT":1}}`, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := i.Report(context.Background(), image, Request{
				Symbol: "addk", Required: []string{"opcode_histogram"}, SASS: string(listing),
			})
			if err != nil {
				t.Fatal(err)
			}
			if tt.histogram != "" {
				r.Feature("opcode_histogram").Value = json.RawMessage(tt.histogram)
			}
			var buf bytes.Buffer
			if err := r.WriteJSON(&buf); err != nil {
				t.Fatal(err)
			}
			_, err = ReadReport(&buf)
			if tt.wantError {
				if !errors.Is(err, ErrInvalidReport) {
					t.Fatalf("ReadReport error = %v, want ErrInvalidReport", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}

// renamedSection returns metadataELF(true, 0, EM_CUDA) with each section
// index in names renamed to its name, which must already be in the string
// table. Renaming section 1, the string table, leaves name lookup intact.
func renamedSection(names map[int]string) []byte {
	b := metadataELF(true, 0, elf.EM_CUDA)
	strtab := b[64+4*64:]
	for index, name := range names {
		binary.LittleEndian.PutUint32(b[64+index*64:], uint32(bytes.Index(strtab, []byte(name+"\x00"))))
	}
	return b
}

// elfVersion returns metadataELF(true, 0, EM_CUDA) with EI_VERSION ident and
// e_version version.
func elfVersion(ident byte, version uint32) []byte {
	b := metadataELF(true, 0, elf.EM_CUDA)
	b[elf.EI_VERSION] = ident
	binary.LittleEndian.PutUint32(b[20:], version)
	return b
}

func TestReportRefusals(t *testing.T) {
	cuda := metadataELF(true, 64, elf.EM_CUDA)
	noCode := bytes.Replace(cuda, []byte(".text.kernel"), []byte(".data.kernel"), 1)
	for _, tt := range []struct {
		name  string
		image []byte
		req   Request
		want  error
	}{
		{"empty_symbol", cuda, Request{}, ErrSymbol},
		{"missing_symbol", cuda, Request{Symbol: "other"}, ErrSymbol},
		{"prefix_symbol", cuda, Request{Symbol: "kern"}, ErrSymbol},
		{"ambiguous_symbol", renamedSection(map[int]string{3: ".text.kernel"}), Request{Symbol: "kernel"}, ErrSymbol},
		{"zero_code", noCode, Request{Symbol: "kernel"}, ErrNoCode},
		{"not_elf", []byte("not an ELF image"), Request{Symbol: "kernel"}, ErrNotCUDA},
		{"truncated", cuda[:40], Request{Symbol: "kernel"}, ErrNotCUDA},
		{"empty", nil, Request{Symbol: "kernel"}, ErrNotCUDA},
		{"x86_elf", metadataELF(true, 64, elf.EM_X86_64), Request{Symbol: "kernel"}, ErrNotCUDA},
		{"e_version_2", elfVersion(1, 2), Request{Symbol: "kernel"}, ErrELFVersion},
		{"ei_version_2", elfVersion(2, 2), Request{Symbol: "kernel"}, ErrELFVersion},
		{"input_limit", cuda, Request{Symbol: "kernel", MaxInputBytes: int64(len(cuda) - 1)}, ErrInputTooLarge},
		{"limit_above_default", cuda, Request{Symbol: "kernel", MaxInputBytes: DefaultMaxInputBytes + 1}, ErrRequest},
		{"negative_limit", cuda, Request{Symbol: "kernel", MaxInputBytes: -1}, ErrRequest},
		{"duplicate_feature", cuda, Request{Symbol: "kernel", Required: []string{"target", "target"}}, ErrRequest},
		{"required_and_optional", cuda, Request{Symbol: "kernel", Required: []string{"target"}, Optional: []string{"target"}}, ErrRequest},
		{"empty_feature", cuda, Request{Symbol: "kernel", Optional: []string{""}}, ErrRequest},
		{"too_many_features", cuda, Request{Symbol: "kernel", Optional: strings.Fields(strings.Repeat("x ", maxFeatureNames+1))}, ErrRequest},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(".")
			r, err := i.Report(context.Background(), tt.image, tt.req)
			if !errors.Is(err, tt.want) || r != nil {
				t.Fatalf("Report = %v, %v; want nil, %v", r, err, tt.want)
			}
		})
	}
}

func TestReportRefusalMessages(t *testing.T) {
	for _, tt := range []struct {
		name  string
		image []byte
		want  string
	}{
		{"e_version", elfVersion(1, 2), "unsupported CUDA ELF version: EI_VERSION 1, e_version 2; only version 1 is supported"},
		{"not_elf", []byte("not an ELF image"), "not a CUDA ELF image: ELF parse failed: invalid ELF image"},
		{"truncated", metadataELF(true, 0, elf.EM_CUDA)[:40], "not a CUDA ELF image: ELF parse failed: "},
		{"x86_elf", metadataELF(true, 0, elf.EM_X86_64), `not a CUDA ELF image: machine "EM_X86_64"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(".")
			_, err := i.Report(context.Background(), tt.image, Request{Symbol: "kernel"})
			if err == nil || !strings.Contains(err.Error(), tt.want) || strings.Contains(err.Error(), `machine ""`) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestReportDuplicateSections(t *testing.T) {
	for _, tt := range []struct {
		name    string
		image   []byte
		feature string
		message string
	}{
		{"nv_info", renamedSection(map[int]string{1: ".nv.info.kernel", 3: ".nv.info.kernel"}), "allocated_registers", "duplicate .nv.info.kernel sections"},
		{"nv_shared", renamedSection(map[int]string{1: ".nv.shared.kernel"}), "static_shared_bytes", "duplicate .nv.shared.kernel sections"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(".")
			r, err := i.Report(context.Background(), tt.image, Request{Symbol: "kernel", Optional: []string{tt.feature}})
			if err != nil {
				t.Fatal(err)
			}
			if got := value(t, r, tt.feature); got != StateError {
				t.Fatalf("%s = %s, want error", tt.feature, got)
			}
			if got := diagnostics(r, tt.feature); got != tt.message {
				t.Fatalf("diagnostics %q, want %q", got, tt.message)
			}
		})
	}
}

func TestReportInputLimitBoundary(t *testing.T) {
	cuda := metadataELF(true, 64, elf.EM_CUDA)
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), cuda, Request{Symbol: "kernel", MaxInputBytes: int64(len(cuda))})
	if err != nil {
		t.Fatal(err)
	}
	if r.Limits.InputBytes != int64(len(cuda)) || r.Artifact.SizeBytes != int64(len(cuda)) {
		t.Fatalf("limits %+v artifact %+v", r.Limits, r.Artifact)
	}
}

func TestReportCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	i, _ := NewInspector(".")
	if _, err := i.Report(ctx, metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestReportDeterministic(t *testing.T) {
	i, _ := NewInspector(".")
	image := roundtripCUBIN(t)
	var out [2]bytes.Buffer
	for n := range out {
		r, err := i.Report(context.Background(), image, Request{Symbol: "addk", Optional: []string{"target"}})
		if err != nil {
			t.Fatal(err)
		}
		if err := r.WriteJSON(&out[n]); err != nil {
			t.Fatal(err)
		}
	}
	if !bytes.Equal(out[0].Bytes(), out[1].Bytes()) {
		t.Fatal("nondeterministic JSON")
	}
	r, err := ReadReport(&out[0])
	if err != nil {
		t.Fatal(err)
	}
	if r.Result != ResultComplete {
		t.Fatalf("result %q", r.Result)
	}
}

func TestReportRecordedCUBIN(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), roundtripCUBIN(t), Request{Symbol: "addk"})
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, state, value string }{
		{"target", StateObserved, `{"architecture":"sm_121","major":12,"minor":1}`},
		// The image is sm_121a, but the ELF SM number alone is 121.
		{"feature_suffix", StateUnknown, "null"},
		{"dynamic_shared_bytes", StateUnsupported, "null"},
	} {
		f := r.Feature(tt.name)
		if f == nil || f.State != tt.state || string(f.Value) != tt.value && !(tt.value == "null" && f.Value == nil) {
			t.Errorf("%s = %+v, want %s %s", tt.name, f, tt.state, tt.value)
		}
	}
	if !r.Feature("toolchain_stamp").Known() || !r.Feature("section_inventory").Known() {
		t.Fatalf("ELF facts missing: %+v", r.Features)
	}
}

func TestReportUnknownsAreNull(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(false, 0, elf.EM_CUDA), Request{Symbol: "kernel"})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := r.WriteJSON(&buf); err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Features []map[string]json.RawMessage `json:"features"`
	}
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	unknown := map[string]bool{"allocated_registers": true, "static_shared_bytes": true, "dynamic_shared_bytes": true, "target": false, "feature_suffix": true, "toolchain_stamp": true, "opcode_histogram": true, "memory_access_widths": true, "local_memory_ops": true}
	for _, f := range decoded.Features {
		var name string
		json.Unmarshal(f["name"], &name)
		value, ok := f["value"]
		if !ok {
			t.Errorf("%s: value omitted, want explicit null", name)
		}
		if unknown[name] != (string(value) == "null") {
			t.Errorf("%s: value %s", name, value)
		}
		delete(unknown, name)
	}
	if len(unknown) > 0 {
		t.Errorf("features missing: %v", unknown)
	}
}

func TestReportRequestedOutcomes(t *testing.T) {
	image := metadataELF(true, 64, elf.EM_CUDA)
	for _, tt := range []struct {
		name     string
		req      Request
		result   string
		required bool
	}{
		{"nothing_requested", Request{}, ResultComplete, false},
		{"required_observed", Request{Required: []string{"static_shared_bytes", "section_inventory"}}, ResultComplete, false},
		{"required_unknown", Request{Required: []string{"allocated_registers"}}, ResultFailed, true},
		{"required_unsupported", Request{Required: []string{"dynamic_shared_bytes"}}, ResultFailed, true},
		{"optional_unknown", Request{Required: []string{"static_shared_bytes"}, Optional: []string{"allocated_registers"}}, ResultPartial, false},
		{"unimplemented_optional", Request{Optional: []string{"bogus"}}, ResultPartial, false},
		{"unimplemented_required", Request{Required: []string{"bogus"}}, ResultFailed, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(".")
			tt.req.Symbol = "kernel"
			r, err := i.Report(context.Background(), image, tt.req)
			if errors.Is(err, ErrRequiredFeature) != tt.required || (err != nil && !tt.required) {
				t.Fatalf("err = %v", err)
			}
			if r == nil || r.Result != tt.result {
				t.Fatalf("report %+v, want result %s", r, tt.result)
			}
			if f := r.Feature("bogus"); f != nil && f.State != StateUnsupported {
				t.Fatalf("bogus = %+v", f)
			}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestReportValidateRejects(t *testing.T) {
	i, _ := NewInspector(".")
	good, err := i.Report(context.Background(), metadataELF(true, 64, elf.EM_CUDA), Request{
		Symbol:   "kernel",
		Optional: []string{"static_shared_bytes"},
		SASS:     kernelSASS("sm_121a", "LDG.E.128", "STS"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	clone := func() *Report {
		var buf bytes.Buffer
		good.WriteJSON(&buf)
		var r Report
		json.Unmarshal(buf.Bytes(), &r)
		return &r
	}
	for _, tt := range []struct {
		name   string
		mutate func(*Report)
	}{
		{"schema", func(r *Report) { r.SchemaVersion = 1 }},
		{"digest", func(r *Report) { r.Artifact.SHA256 = "abc" }},
		{"size_over_limit", func(r *Report) { r.Artifact.SizeBytes = r.Limits.InputBytes + 1 }},
		{"tool_limit_claimed", func(r *Report) { r.Limits.ToolOutput = "1MiB" }},
		{"unknown_with_value", func(r *Report) { r.Feature("dynamic_shared_bytes").Value = json.RawMessage("0") }},
		{"unknown_without_diagnostic", func(r *Report) { r.Diagnostics = nil }},
		{"observed_null", func(r *Report) { r.Feature("static_shared_bytes").Value = nil }},
		{"observed_without_evidence", func(r *Report) { r.Feature("static_shared_bytes").Evidence = nil }},
		{"unknown_state", func(r *Report) { r.Feature("target").State = "maybe" }},
		{"duplicate_feature", func(r *Report) { r.Features = append(r.Features, r.Features[0]) }},
		{"required_and_optional", func(r *Report) { r.Request.Required = []string{"static_shared_bytes"} }},
		{"duplicate_request", func(r *Report) { r.Request.Optional = []string{"target", "target"} }},
		{"requested_absent", func(r *Report) { r.Request.Optional = []string{"absent"} }},
		{"result_overstated", func(r *Report) {
			r.Request.Required = []string{"allocated_registers"}
			r.Request.Optional = nil
		}},
		{"result_unknown", func(r *Report) { r.Result = "fine" }},
		{"limit_differs_from_request", func(r *Report) { r.Limits.InputBytes-- }},
		{"limit_forged", func(r *Report) {
			r.Request.MaxInputBytes = 5
			r.Limits.InputBytes = 1 << 20
		}},
		{"suffix_observed", func(r *Report) { r.Feature("feature_suffix").State = StateObserved }},
		{"suffix_value", func(r *Report) { r.Feature("feature_suffix").Value = json.RawMessage(`"z"`) }},
		{"suffix_not_string", func(r *Report) { r.Feature("feature_suffix").Value = json.RawMessage(`1`) }},
		{"target_string", func(r *Report) { r.Feature("target").Value = json.RawMessage(`"anything"`) }},
		{"target_suffix", func(r *Report) {
			r.Feature("target").Value = json.RawMessage(`{"architecture":"sm_121a","major":12,"minor":1}`)
		}},
		{"target_numbers", func(r *Report) {
			r.Feature("target").Value = json.RawMessage(`{"architecture":"sm_121","major":12,"minor":0}`)
		}},
		{"target_extra_field", func(r *Report) {
			r.Feature("target").Value = json.RawMessage(`{"architecture":"sm_121","major":12,"minor":1,"suffix":"a"}`)
		}},
		{"shared_negative", func(r *Report) { r.Feature("static_shared_bytes").Value = json.RawMessage(`-1`) }},
		{"sections_empty", func(r *Report) { r.Feature("section_inventory").Value = json.RawMessage(`[]`) }},
		{"histogram_sum", func(r *Report) {
			r.Feature("opcode_histogram").Value = json.RawMessage(`{"counted":3,"quarantined":0,"opcodes":{"LDG":1,"STS":1}}`)
		}},
		{"widths_class", func(r *Report) {
			r.Feature("memory_access_widths").Value = json.RawMessage(`{"texture_load":{"32":1}}`)
		}},
		{"widths_bits", func(r *Report) {
			r.Feature("memory_access_widths").Value = json.RawMessage(`{"global_load":{"256":1}}`)
		}},
		{"local_shape", func(r *Report) { r.Feature("local_memory_ops").Value = json.RawMessage(`{"spills":1}`) }},
		{"registers_zero", func(r *Report) {
			f := r.Feature("allocated_registers")
			f.State, f.Value, f.Evidence = StateObserved, json.RawMessage(`0`), []string{"claimed"}
		}},
		{"dynamic_shared_observed", func(r *Report) {
			f := r.Feature("dynamic_shared_bytes")
			f.State, f.Value, f.Evidence = StateObserved, json.RawMessage(`0`), []string{"claimed"}
		}},
		{"unimplemented_observed", func(r *Report) {
			r.Features = append(r.Features, Feature{Name: "spill_bytes", State: StateObserved, Value: json.RawMessage(`0`), Evidence: []string{"claimed"}})
		}},
		{"implemented_absent", func(r *Report) {
			r.Features = slices.DeleteFunc(r.Features, func(f Feature) bool { return f.Name == "section_inventory" })
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := clone()
			if err := r.Validate(); err != nil {
				t.Fatalf("clone invalid: %v", err)
			}
			tt.mutate(r)
			if err := r.Validate(); !errors.Is(err, ErrInvalidReport) {
				t.Fatalf("Validate = %v, want ErrInvalidReport", err)
			}
		})
	}
}

func TestReadReportStrict(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 64, elf.EM_CUDA), Request{Symbol: "kernel"})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	r.WriteJSON(&buf)
	good := buf.String()
	for _, tt := range []struct {
		name, input string
		want        error
	}{
		{"unknown_field", strings.Replace(good, `"schema_version"`, `"extra": 1, "schema_version"`, 1), ErrInvalidReport},
		{"trailing_object", good + "{}", ErrInvalidReport},
		{"trailing_brace", good + "}", ErrInvalidReport},
		{"forged_limit", strings.Replace(strings.Replace(good, `"input_bytes": 67108864`, `"input_bytes": 1048576`, 1), `"max_input_bytes": 0`, `"max_input_bytes": 5`, 1), ErrInvalidReport},
		{"oversize", strings.Repeat(" ", MaxReportBytes+1), ErrInputTooLarge},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ReadReport(strings.NewReader(tt.input)); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
	if _, err := ReadReport(strings.NewReader(good)); err != nil {
		t.Fatal(err)
	}
}

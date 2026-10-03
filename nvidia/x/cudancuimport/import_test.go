package cudancuimport

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func knownEnvelope(t *testing.T) Envelope {
	t.Helper()
	b, err := os.ReadFile("testdata/transpose-details.csv")
	if err != nil {
		t.Fatal(err)
	}
	return Envelope{SchemaVersion: 1, ToolVersion: ToolVersion, Format: Format, SourcePath: "NVIDIA bundled sharedBankConflicts/transposeNoBankConflicts.ncu-rep", ReportDigest: "544b42110730bd2adf2325eeea44a69be6455767443fe1d7a1572bbd7183bb29", CSVDigest: fmtDigest(b), CSV: string(b)}
}
func fmtDigest(b []byte) string   { return hex.EncodeToString(sha256Bytes(b)) }
func sha256Bytes(b []byte) []byte { s := sha256.Sum256(b); return s[:] }
func TestImportKnownReport(t *testing.T) {
	e := knownEnvelope(t)
	r, err := Import(e, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Rows) != 65 {
		t.Fatalf("rows=%d", len(r.Rows))
	}
	metrics, rules := 0, 0
	for _, row := range r.Rows {
		if row.Entry != "_Z24transposeNoBankConflictsPfS_ii" || row.Block != [3]uint64{32, 8, 1} || row.Grid != [3]uint64{256, 256, 1} || row.Target != "sm_86" || row.ModuleID != "" {
			t.Fatalf("bad launch %+v", row)
		}
		if row.Metric != nil {
			metrics++
			if row.Metric.Value.Record().Class != cudaevidence.Unread {
				t.Fatal("fixture lacks external launch identity")
			}
		} else {
			rules++
			if len(row.Raw) != 20 || row.Raw[15] == "" {
				t.Fatal("lost rule diagnostic")
			}
		}
	}
	if metrics != 58 || rules != 7 {
		t.Fatalf("metrics=%d rules=%d", metrics, rules)
	}
	if r.Envelope.CSV != e.CSV {
		t.Fatal("raw CSV changed")
	}
}
func TestImportRejectsUnknownSchema(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Envelope)
	}{
		{"schema", func(e *Envelope) { e.SchemaVersion = 2 }},
		{"tool", func(e *Envelope) { e.ToolVersion = "2026.4.0.0" }},
		{"format", func(e *Envelope) { e.Format = "ncu-raw-csv" }},
		{"header", func(e *Envelope) {
			e.CSV = strings.Replace(e.CSV, "Metric Unit", "Units", 1)
			e.CSVDigest = fmtDigest([]byte(e.CSV))
		}},
		{"digest", func(e *Envelope) { e.CSV += "\n" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := knownEnvelope(t)
			tc.change(&e)
			if _, err := Import(e, nil); err == nil {
				t.Fatal("unsupported input accepted")
			}
		})
	}
}
func TestImportPreservesMetricUnits(t *testing.T) {
	r, err := Import(knownEnvelope(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range r.Rows {
		if row.Metric == nil {
			continue
		}
		m := row.Metric
		if m.Unit != row.Raw[13] || m.RawValue != row.Raw[14] || m.Name != row.Raw[12] || m.Value.Record().Unit != m.Unit {
			t.Fatal("raw metric changed")
		}
	}
}
func TestImportRecordsToolAndReportDigest(t *testing.T) {
	e := knownEnvelope(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := Read(bytes.NewReader(b), MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	r, err := Import(decoded, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Envelope.ToolVersion != ToolVersion || r.Envelope.ReportDigest != e.ReportDigest || r.Envelope.SourcePath != e.SourcePath || r.Envelope.CSVDigest != "57ba1eddd8918134381f4db76672ffd16bc4b1e3520a432f42853f16d0e05411" {
		t.Fatal("provenance changed")
	}
}

// Synthetic launch bindings below are project producer receipts, not metadata
// claimed to exist in NVIDIA's bundled transpose report. The module bytes and
// source/disassembly fixtures are retained real cudacapture artifacts.
func joinedFixture(t *testing.T) (Envelope, Candidate) {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("..", "cudacapture", "testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	raw, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "fill", schema, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	w := cudacache.WorkloadIdentity{ProgramID: "synthetic fill launch receipt", ShapeDims: []int{64}, NumericalMode: "fp32"}
	key := cudacache.CompilerCacheKey(a.Digest, cudacache.CompilerIdentity{TargetArch: a.Target, OutputKind: nvrtcx.OutputCUBIN, Flags: []string{"retained fixture"}})
	id, err := cudaprof.BindEvidence(a, w, key)
	if err != nil {
		t.Fatal(err)
	}
	code := cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"}
	c := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}}, Files: map[string][]byte{"kernels.cu": read("kernels.cu"), "kernels.ptx": read("kernels.ptx"), "kernels.cubin": raw, "kernels.sass": read("kernels.sass")}, Code: code, Modules: []cudacapture.Module{{ID: "module", Generation: 1, Code: code}}}
	e := knownEnvelope(t)
	e.SourcePath = "synthetic fill launch exporter control"
	e.ReportDigest = fmtDigest([]byte(e.SourcePath))
	var b bytes.Buffer
	writer := csv.NewWriter(&b)
	if err := writer.Write(detailsHeader); err != nil {
		t.Fatal(err)
	}
	fields := []string{"0", "1", "synthetic", "host", "fill", "1", "7", "(64, 1, 1)", "(1, 1, 1)", "0", "9.0", "synthetic section", "gpu__time_duration.sum", "ns", "42", ""}
	if err := writer.Write(fields); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	e.CSV = b.String()
	e.CSVDigest = fmtDigest(b.Bytes())
	e.Bindings = []Binding{{Launch: LaunchKey{"0", "1", "1", "7"}, Identity: id}}
	return e, Candidate{Capture: c, Artifact: a, Workload: w, CompilerKey: key}
}
func importJoined(t *testing.T, e Envelope, c []Candidate) *Report {
	t.Helper()
	r, err := Import(e, c)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func TestImportedMetricIsExternalMeasured(t *testing.T) {
	e, c := joinedFixture(t)
	row := importJoined(t, e, []Candidate{c}).Rows[0]
	r := row.Metric.Value.Record()
	if row.ModuleID != "module" || row.Generation != 1 || row.JoinReason != "" || r.Class != cudaevidence.Measured || r.Number == nil || *r.Number != 42 || r.Basis != "external:ncu" || r.Source.Producer != "external:ncu" || r.Source.Identity != e.Bindings[0].Identity {
		t.Fatalf("not external measured: %+v %+v", row, r)
	}
}
func TestImportDoesNotClaimLocalCUPTI(t *testing.T) {
	e, c := joinedFixture(t)
	r := importJoined(t, e, []Candidate{c})
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte("local:cupti")) || !bytes.Contains(b, []byte("external:ncu")) {
		t.Fatal("wrong collector provenance")
	}
	before := len(c.Capture.Observations)
	Import(e, []Candidate{c})
	if len(c.Capture.Observations) != before {
		t.Fatal("modified local observations")
	}
}
func TestImportRejectsStaleImageJoin(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Envelope, *Candidate)
	}{
		{"image", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.ArtifactDigest = strings.Repeat("1", 64) }},
		{"entry", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.Entry = "scale" }},
		{"target", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.Target = "sm_80" }},
		{"compiler", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.CompilerKey = strings.Repeat("2", 64) }},
		{"workload", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.WorkloadKey = strings.Repeat("3", 64) }},
		{"retained bytes", func(e *Envelope, c *Candidate) { c.Capture.Files["kernels.cubin"][0] ^= 1 }},
		{"artifact bytes", func(e *Envelope, c *Candidate) { c.Artifact.Raw[0] ^= 1 }},
		{"capture workload", func(e *Envelope, c *Candidate) {
			c.Capture.Identity = &cudacapture.ComparisonIdentity{WorkloadHash: strings.Repeat("4", 64)}
		}},
		{"missing identity", func(e *Envelope, c *Candidate) { e.Bindings = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, c := joinedFixture(t)
			tc.change(&e, &c)
			row := importJoined(t, e, []Candidate{c}).Rows[0]
			if row.ModuleID != "" || row.JoinReason == "" || row.Metric.Value.Record().Class != cudaevidence.Unread {
				t.Fatal("stale join became measured")
			}
		})
	}
}
func TestImportRefusesAmbiguousKernelJoin(t *testing.T) {
	e, c := joinedFixture(t)
	m := c.Capture.Modules[0]
	m.ID = "second"
	m.Generation = 2
	c.Capture.Modules = append(c.Capture.Modules, m)
	r := importJoined(t, e, []Candidate{c})
	if r.Rows[0].JoinReason != "ambiguous retained module generation" || r.Rows[0].Metric.Value.Record().Class != cudaevidence.Unread {
		t.Fatal("ambiguous generation accepted")
	}
	e, c = joinedFixture(t)
	if importJoined(t, e, []Candidate{c, c}).Rows[0].ModuleID != "" {
		t.Fatal("candidate order broke ambiguity")
	}
}

func TestImportRejectsMislabeledNativeTarget(t *testing.T) {
	e, c := joinedFixture(t)
	c.Artifact.Target = "sm_80"
	e.Bindings[0].Identity.Target = "sm_80"
	e.CSV = strings.Replace(e.CSV, ",9.0,", ",8.0,", 1)
	e.CSVDigest = fmtDigest([]byte(e.CSV))
	row := importJoined(t, e, []Candidate{c}).Rows[0]
	if row.ModuleID != "" || row.Metric.Value.Record().Class != cudaevidence.Unread {
		t.Fatal("descriptor target overrode actual retained architecture")
	}
}

func TestImportBindingAndInventoryRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*Envelope, *Candidate)
	}{
		{"duplicate binding", func(e *Envelope, c *Candidate) { e.Bindings = append(e.Bindings, e.Bindings[0]) }},
		{"missing exported launch", func(e *Envelope, c *Candidate) { e.Bindings[0].Launch.ID = "99" }},
		{"partial binding", func(e *Envelope, c *Candidate) { e.Bindings[0].Identity.CompilerKey = "" }},
		{"descriptor budget", func(e *Envelope, c *Candidate) { c.Artifact.CompilerOpts = make([]string, 129) }},
		{"workload budget", func(e *Envelope, c *Candidate) { c.Workload.ShapeDims = make([]int, 1025) }},
		{"module budget", func(e *Envelope, c *Candidate) { c.Capture.Modules = make([]cudacapture.Module, 65) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, c := joinedFixture(t)
			tc.change(&e, &c)
			if _, err := Import(e, []Candidate{c}); err == nil {
				t.Fatal("invalid binding/inventory admitted")
			}
		})
	}
}
func changeMetric(t *testing.T, e *Envelope, name, unit, value string) {
	t.Helper()
	r := csv.NewReader(strings.NewReader(e.CSV))
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	rows[1][12], rows[1][13], rows[1][14] = name, unit, value
	var b bytes.Buffer
	w := csv.NewWriter(&b)
	if err := w.WriteAll(rows); err != nil {
		t.Fatal(err)
	}
	e.CSV = b.String()
	e.CSVDigest = fmtDigest(b.Bytes())
}
func TestImportMetricRefusals(t *testing.T) {
	for _, tc := range []struct{ name, metric, unit, value string }{
		{"unit", "gpu__time_duration.sum", "ms", "42"},
		{"denominator", "gpu__time_duration.sum.per_second", "ns", "42"},
		{"nan", "gpu__time_duration.sum", "ns", "NaN"},
		{"inf", "gpu__time_duration.sum", "ns", "Inf"},
		{"decimal", "gpu__time_duration.sum", "ns", "1.2"},
		{"unsigned exact", "smsp__inst_executed.sum", "inst", "18446744073709551615"},
		{"unknown", "vendor__unknown.sum", "ns", "42"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, c := joinedFixture(t)
			changeMetric(t, &e, tc.metric, tc.unit, tc.value)
			m := importJoined(t, e, []Candidate{c}).Rows[0].Metric
			if m.Value.Record().Class != cudaevidence.Unread || m.Value.Record().Number != nil || m.RawValue != tc.value || m.Unit != tc.unit {
				t.Fatal("unsupported interpretation lost raw or became numeric")
			}
			if tc.name == "unsigned exact" && (m.Unsigned == nil || *m.Unsigned != ^uint64(0)) {
				t.Fatal("integer rounded")
			}
		})
	}
	t.Run("zero measured", func(t *testing.T) {
		e, c := joinedFixture(t)
		changeMetric(t, &e, "gpu__time_duration.sum", "ns", "0")
		m := importJoined(t, e, []Candidate{c}).Rows[0].Metric
		if m.Value.Record().Number == nil || *m.Value.Record().Number != 0 || m.Value.Record().Class != cudaevidence.Measured {
			t.Fatal("zero confused with missing")
		}
	})
}

func TestImportGeometryBounds(t *testing.T) {
	for _, tc := range []struct {
		name, value string
		valid       bool
	}{
		{"normal", "(64, 1, 1)", true},
		{"uint32 axes within uint64 product", "(4294967295, 4294967295, 1)", true},
		{"uint64 product overflow", "(4294967295, 4294967295, 4294967295)", false},
		{"zero axis", "(64, 0, 1)", false},
		{"negative axis", "(-1, 1, 1)", false},
		{"uint32 axis overflow", "(4294967296, 1, 1)", false},
		{"uint64 axis overflow", "(18446744073709551616, 1, 1)", false},
		{"two axes", "(64, 1)", false},
		{"four axes", "(64, 1, 1, 1)", false},
		{"missing parentheses", "64, 1, 1", false},
	} {
		for _, index := range []int{7, 8} {
			t.Run(tc.name+strconv.Itoa(index), func(t *testing.T) {
				e, c := joinedFixture(t)
				changeLaunchField(t, &e, index, tc.value)
				r, err := Import(e, []Candidate{c})
				if tc.valid {
					if err != nil || r.Rows[0].Metric.Value.Record().Class != cudaevidence.Measured {
						t.Fatalf("valid dimensions rejected: %v", err)
					}
				} else if err == nil {
					t.Fatal("invalid geometry admitted")
				}
			})
		}
	}
}

func changeLaunchField(t *testing.T, e *Envelope, index int, value string) {
	t.Helper()
	reader := csv.NewReader(strings.NewReader(e.CSV))
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	rows[1][index] = value
	var b bytes.Buffer
	writer := csv.NewWriter(&b)
	if err := writer.WriteAll(rows); err != nil {
		t.Fatal(err)
	}
	e.CSV = b.String()
	e.CSVDigest = fmtDigest(b.Bytes())
}

func TestImportScalarDenominators(t *testing.T) {
	for _, tc := range []struct{ name, unit, denominator, value string }{
		{"launch__registers_per_thread", "register/thread", "thread", "24"},
		{"launch__shared_mem_per_block_static", "byte/block", "block", "4224"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, c := joinedFixture(t)
			changeMetric(t, &e, tc.name, tc.unit, tc.value)
			m := importJoined(t, e, []Candidate{c}).Rows[0].Metric
			if m.Value.Record().Class != cudaevidence.Measured || m.Denominator != tc.denominator || m.Unit != tc.unit || m.Value.Record().Unit != tc.unit {
				t.Fatalf("denominator lost: %+v", m)
			}
			malformed := strings.Replace(tc.unit, tc.denominator, "warp", 1)
			changeMetric(t, &e, tc.name, malformed, tc.value)
			m = importJoined(t, e, []Candidate{c}).Rows[0].Metric
			if m.Value.Record().Class != cudaevidence.Unread || m.Value.Record().Number != nil || m.Unit != malformed {
				t.Fatal("malformed denominator admitted or changed")
			}
		})
	}
}
func TestReadStrictAndBounded(t *testing.T) {
	e := knownEnvelope(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{
		append(append([]byte(nil), b...), []byte(" {}")...),
		bytes.Replace(b, []byte(`"schema_version":1`), []byte(`"schema_version":1,"schema_version":1`), 1),
		bytes.Replace(b, []byte(`"schema_version"`), []byte(`"Schema_Version"`), 1),
		bytes.Replace(b, []byte(`"schema_version"`), []byte(`"ſchema_version"`), 1),
		bytes.Replace(b, []byte(`"schema_version"`), []byte(`"\u017fchema_version"`), 1),
		bytes.Replace(b, []byte(`"schema_version"`), []byte(`"\u0053chema_version"`), 1),
		bytes.Replace(b, []byte(`"schema_version":1`), []byte(`"unknown":1,"schema_version":1`), 1),
	} {
		if _, err := Read(bytes.NewReader(bad), MaxBytes); err == nil {
			t.Fatal("ambiguous JSON accepted")
		}
	}
	if _, err := Read(bytes.NewReader(b), len(b)-1); err == nil {
		t.Fatal("byte budget ignored")
	}
	if _, err := Read(bytes.NewReader(b), 0); err == nil {
		t.Fatal("invalid limit accepted")
	}
}

func TestReadRejectsNestedUnicodeAliases(t *testing.T) {
	e, _ := joinedFixture(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ key, alias string }{
		{"stream", `ſtream`}, {"stream", `\u017ftream`},
		{"artifact_digest", `artifact_digeſt`}, {"artifact_digest", `artifact_dige\u017ft`},
		{"workload_key", `workload_Key`}, {"workload_key", `workload_\u212aey`},
	} {
		bad := bytes.Replace(b, []byte(`"`+tc.key+`"`), []byte(`"`+tc.alias+`"`), 1)
		if _, err := Read(bytes.NewReader(bad), MaxBytes); err == nil {
			t.Fatalf("nested alias accepted: %s", tc.alias)
		}
	}
}
func FuzzRead(f *testing.F) {
	f.Add([]byte(`{"schema_version":2}`))
	f.Add([]byte(`{}`))
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		e, err := Read(bytes.NewReader(b), 1<<20)
		if err == nil {
			Import(e, nil)
		}
	})
}

func FuzzImportCSV(f *testing.F) {
	b, err := os.ReadFile("testdata/transpose-details.csv")
	if err != nil {
		f.Fatal(err)
	}
	f.Add(string(b))
	lines := strings.SplitN(string(b), "\n", 3)
	if len(lines) >= 2 {
		f.Add(lines[0] + "\n" + lines[1] + "\n")
	}
	f.Fuzz(func(t *testing.T, text string) {
		if len(text) > 1<<20 {
			return
		}
		e := Envelope{SchemaVersion: 1, ToolVersion: ToolVersion, Format: Format, SourcePath: "synthetic parser control", ReportDigest: fmtDigest([]byte("synthetic parser control")), CSV: text, CSVDigest: fmtDigest([]byte(text))}
		Import(e, nil)
	})
}

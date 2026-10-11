package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func mutateCompletion(t *testing.T, raw []byte, edit func(*nvtelemetry.Manifest)) []byte {
	t.Helper()
	var records []nvtelemetry.StreamRecord
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var rec nvtelemetry.StreamRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatal(err)
		}
		records = append(records, rec)
	}
	edit(records[len(records)-1].Manifest)
	var out bytes.Buffer
	for _, rec := range records {
		if err := json.NewEncoder(&out).Encode(rec); err != nil {
			t.Fatal(err)
		}
	}
	return out.Bytes()
}

func TestTelemetryRetainedKernelCannotHaveZeroTotal(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	raw = mutateCompletion(t, raw, func(m *nvtelemetry.Manifest) { m.LostRecordsByFamily["activity"] = tracecontract.Loss{} })
	if _, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt); !errors.Is(err, nvtelemetry.ErrMalformedStream) {
		t.Fatalf("zero count with retained kernel: %v", err)
	}
}

func TestTelemetryIndependentLossUnits(t *testing.T) {
	for _, tc := range []struct {
		name     string
		loss     tracecontract.Loss
		admitted bool
		absent   bool
	}{
		{"different producer population", tracecontract.Loss{Total: 900}, true, false},
		{"raw dropped", tracecontract.Loss{Total: 900, Dropped: 10}, false, false},
		{"raw unknown", tracecontract.Loss{Unknown: true}, false, false},
		{"raw missing", tracecontract.Loss{}, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, opt := mixedProtocol()
			opt.Receipt.LostRecords["activity"] = tc.loss
			if tc.absent {
				delete(opt.Receipt.LostRecords, "activity")
			}
			before := opt.Receipt.LostRecords["activity"]
			raw := collectMixed(t, r, opt)
			got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
			if err != nil {
				t.Fatal(err)
			}
			if got.Diagnosis.Admitted != tc.admitted {
				t.Fatalf("raw loss ignored: %+v", got.Diagnosis)
			}
			if tc.absent {
				finding := got.Observations[1].Finding
				if finding.Admission.Status != "admitted" || !finding.Admission.Admitted || !strings.Contains(finding.Admission.Reason, "no producer population completeness claim") {
					t.Fatalf("retained-only scope missing: %+v", finding.Admission)
				}
			}
			if !reflect.DeepEqual(opt.Receipt.LostRecords["activity"], before) {
				t.Fatal("independent receipt mutated")
			}
		})
	}
}

func TestTelemetryMissingFamilyCannotEstablishZeroLoss(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	raw = mutateCompletion(t, raw, func(m *nvtelemetry.Manifest) { delete(m.LostRecordsByFamily, "activity") })
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Diagnosis.Admitted || got.Observations[1].Finding.Admission.Status != "withheld" {
		t.Fatal("missing family became known zero")
	}
}

func TestTelemetryAdmissionFieldsAgree(t *testing.T) {
	r, opt := mixedProtocol()
	raw := collectMixed(t, r, opt)
	got, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range got.Observations {
		if o.Finding.Admission.Status != "admitted" || !o.Finding.Admission.Admitted {
			t.Fatalf("positive admission inconsistent: %+v", o.Finding.Admission)
		}
	}
	raw = mutateCompletion(t, raw, func(m *nvtelemetry.Manifest) {
		loss := m.LostRecordsByFamily["activity"]
		loss.Unknown = true
		m.LostRecordsByFamily["activity"] = loss
	})
	got, err = cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(raw), opt)
	if err != nil {
		t.Fatal(err)
	}
	a := got.Observations[1].Finding.Admission
	if a.Status != "withheld" || a.Admitted {
		t.Fatalf("negative admission inconsistent: %+v", a)
	}
}

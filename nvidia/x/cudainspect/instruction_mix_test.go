package cudainspect

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestPrecisionAttributionRoundtripObserved(t *testing.T) {
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r, err := i.Report(context.Background(), roundtripCUBIN(t), Request{
		Symbol:   "addk",
		Required: []string{"precision_attribution", "opcode_histogram"},
		SASS:     roundtripSASS(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := r.WriteJSON(&buf); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadReport(&buf)
	if err != nil {
		t.Fatal(err)
	}

	mix, state, err := decoded.InstructionMix()
	if err != nil {
		t.Fatalf("InstructionMix: %v", err)
	}
	if state != StateObserved {
		t.Fatalf("state = %q, want %q", state, StateObserved)
	}
	if got := mix.Counts[cudaroofline.FP32]; got != 1 || len(mix.Counts) != 1 {
		t.Fatalf("mix.Counts = %+v, want {fp32: 1}", mix.Counts)
	}

	attr, err := cudacensus.AttributeFromMix(mix)
	if err != nil {
		t.Fatalf("AttributeFromMix: %v", err)
	}
	if !attr.Known || attr.Precision != cudaroofline.FP32 || attr.Method != cudacensus.FromInstructionMix || attr.Share != 1 {
		t.Fatalf("AttributeFromMix = %+v", attr)
	}

	pa, err := decoded.PrecisionAttribution()
	if err != nil {
		t.Fatalf("PrecisionAttribution: %v", err)
	}
	if pa.Symbol != "addk" || pa.State != StateObserved || pa.Counted != 1 || pa.Excluded != 31 || pa.Opcodes["FADD"] != 1 {
		t.Fatalf("PrecisionAttribution = %+v", pa)
	}
	if pa.Artifact != decoded.Artifact || decoded.SASS == nil || pa.SASS != *decoded.SASS {
		t.Fatalf("digest mismatch: pa=%+v report=%+v", pa, decoded)
	}

	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatal(err)
	}
	est := cudacensus.NewEstimate(m, cudacensus.Inputs{
		Readings: []cudacensus.Reading{
			{Counter: cudacensus.FP32Active, Value: 0.5, Available: true},
			{Counter: cudacensus.DRAMBandwidthUtil, Value: 0.25, Available: true},
		},
		Attribution: attr,
		Interval:    10 * time.Millisecond,
	})
	if !est.Available || est.Confidence != cudacensus.ConfidenceStaticMix || est.AchievedTFLOPS != 20.0 {
		t.Fatalf("NewEstimate = %+v", est)
	}
}

func TestPrecisionAttributionMixedFormatsAndEstimate(t *testing.T) {
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	image := metadataELF(true, 64, elf.EM_CUDA)

	for _, tt := range []struct {
		name      string
		opcodes   []string
		wantPrec  cudaroofline.Precision
		wantShare float64
		wantCount uint64
		wantExcl  uint64
		wantPred  uint64
	}{
		{
			name:      "fp16_tensor_dominant_over_fp32",
			opcodes:   []string{"LDG.E.128", "HMMA.16816.F32.F16", "HMMA.16816.F32.F16", "HMMA.16816.F16", "FADD.FTZ", "STG.E.128", "EXIT"},
			wantPrec:  cudaroofline.FP16Tensor,
			wantShare: 0.75,
			wantCount: 4,
			wantExcl:  3,
		},
		{
			name:      "bf16_tensor_distinct_from_fp16_tensor",
			opcodes:   []string{"HMMA.16816.F32.BF16", "HMMA.16816.F32.BF16", "HMMA.16816.F32.F16", "EXIT"},
			wantPrec:  cudaroofline.BF16Tensor,
			wantShare: 2.0 / 3.0,
			wantCount: 3,
			wantExcl:  1,
		},
		{
			name:      "tf32_tensor_distinct_from_fp32_scalar",
			opcodes:   []string{"HMMA.1688.F32.TF32", "HMMA.16816.F32.TF32", "FFMA", "EXIT"},
			wantPrec:  cudaroofline.TF32Tensor,
			wantShare: 2.0 / 3.0,
			wantCount: 3,
			wantExcl:  1,
		},
		{
			name:      "fp8_tensor_e4m3_and_e5m2",
			opcodes:   []string{"QMMA.16832.F32.E4M3.E5M2", "QMMA.16832.F16.E4M3.E4M3", "HMMA.16832.F32.E5M2", "FADD", "EXIT"},
			wantPrec:  cudaroofline.FP8Tensor,
			wantShare: 0.75,
			wantCount: 4,
			wantExcl:  1,
		},
		{
			name:      "int8_tensor_imma_with_s32_accumulator",
			opcodes:   []string{"IMMA.16832.S32.S8.S8", "IMMA.8816.U8.S8", "FADD", "IMAD.WIDE.U32", "EXIT"},
			wantPrec:  cudaroofline.INT8Tensor,
			wantShare: 2.0 / 3.0,
			wantCount: 3,
			wantExcl:  2,
		},
		{
			name:      "fp64_scalar_and_dmma",
			opcodes:   []string{"DFMA", "DADD", "DMMA.1688.F64", "FADD", "EXIT"},
			wantPrec:  cudaroofline.FP64,
			wantShare: 0.75,
			wantCount: 4,
			wantExcl:  1,
		},
		{
			name:      "fp16_non_tensor_hfma2",
			opcodes:   []string{"HFMA2", "HADD2.F16_V2", "HMUL2", "FFMA", "EXIT"},
			wantPrec:  cudaroofline.FP16,
			wantShare: 0.75,
			wantCount: 4,
			wantExcl:  1,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := i.Report(context.Background(), image, Request{
				Symbol:   "kernel",
				Required: []string{"precision_attribution"},
				SASS:     kernelSASS("sm_121a", tt.opcodes...),
			})
			if err != nil {
				t.Fatalf("Report: %v (diagnostics: %+v)", err, r.Diagnostics)
			}
			var buf bytes.Buffer
			if err := r.WriteJSON(&buf); err != nil {
				t.Fatal(err)
			}
			decoded, err := ReadReport(&buf)
			if err != nil {
				t.Fatalf("ReadReport: %v", err)
			}
			mix, state, err := decoded.InstructionMix()
			if err != nil {
				t.Fatalf("InstructionMix: %v", err)
			}
			if state != StateInferred {
				t.Fatalf("unbound SASS state = %q, want %q", state, StateInferred)
			}
			attr, err := cudacensus.AttributeFromMix(mix)
			if err != nil {
				t.Fatalf("AttributeFromMix: %v", err)
			}
			if attr.Precision != tt.wantPrec || attr.Method != cudacensus.FromInstructionMix || attr.Share != tt.wantShare || !attr.Known {
				t.Fatalf("AttributeFromMix = %+v, want %q %q %v", attr, tt.wantPrec, cudacensus.FromInstructionMix, tt.wantShare)
			}
			pa, err := decoded.PrecisionAttribution()
			if err != nil {
				t.Fatal(err)
			}
			if pa.Counted != tt.wantCount || pa.Excluded != tt.wantExcl || pa.Predicated != tt.wantPred {
				t.Fatalf("counts = %+v", pa)
			}
			if !strings.Contains(diagnostics(decoded, "precision_attribution"), "not bound to image bytes") {
				t.Fatalf("missing unbound SASS diagnostic: %+v", decoded.Diagnostics)
			}

			// Non-FP32 attribution succeeds statically while NewEstimate reports
			// the absent non-FP32 compute peak without fabricating a number.
			m, err := cudaroofline.NewModel("sm_121a", cudaroofline.DeviceResident)
			if err != nil {
				t.Fatal(err)
			}
			est := cudacensus.NewEstimate(m, cudacensus.Inputs{
				Readings: []cudacensus.Reading{
					{Counter: cudacensus.HMMATensorActive, Value: 0.75, Available: true},
					{Counter: cudacensus.FP32Active, Value: 0.75, Available: true},
				},
				Attribution: attr,
				Interval:    10 * time.Millisecond,
			})
			if est.Available || est.Confidence != cudacensus.ConfidenceUnavailable || !strings.Contains(est.Derivation.UnavailableReason, "FP32 non-tensor peak only") {
				t.Fatalf("NewEstimate for %s = %+v, want unavailable non-FP32 peak", tt.wantPrec, est)
			}
		})
	}
}

func TestPrecisionAttributionRefusalsAndUnavailable(t *testing.T) {
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	image := metadataELF(true, 64, elf.EM_CUDA)

	for _, tt := range []struct {
		name     string
		sass     string
		wantDiag string
	}{
		{
			name:     "missing_listing",
			sass:     "",
			wantDiag: "no SASS supplied",
		},
		{
			name:     "empty_arithmetic_mix",
			sass:     kernelSASS("sm_121a", "LDG.E", "IMAD.WIDE.U32", "STG.E", "EXIT"),
			wantDiag: "empty instruction mix",
		},
		{
			name:     "tie_between_formats",
			sass:     kernelSASS("sm_121a", "FFMA", "FFMA", "HMMA.16816.F32.F16", "HMMA.16816.F32.F16", "EXIT"),
			wantDiag: "no dominant precision",
		},
		{
			name:     "hmma_accumulator_f32_without_input_format",
			sass:     kernelSASS("sm_121a", "HMMA.16816.F32", "FFMA", "FFMA", "EXIT"),
			wantDiag: "names accumulator F32 without an explicit tensor input format",
		},
		{
			name:     "bare_hmma_without_input_format",
			sass:     kernelSASS("sm_121a", "HMMA.16816", "FFMA", "EXIT"),
			wantDiag: "states no tensor input format",
		},
		{
			name:     "unknown_fp4_tensor_input_does_not_disappear",
			sass:     kernelSASS("sm_121a", "FFMA", "FFMA", "FFMA", "HMMA.16816.F32.E2M1", "EXIT"),
			wantDiag: "no cudaroofline.Precision representation",
		},
		{
			name:     "omma_tensor_rejected",
			sass:     kernelSASS("sm_121a", "FFMA", "OMMA.16864.F32.E2M1", "EXIT"),
			wantDiag: "no cudaroofline.Precision representation",
		},
		{
			name:     "imma_subbyte_s4_rejected",
			sass:     kernelSASS("sm_121a", "IMMA.16832.S32.S8.S8", "IMMA.16832.S32.S8.S8", "IMMA.8832.S32.S4.S4", "EXIT"),
			wantDiag: "unsupported or ambiguous IMMA modifier",
		},
		{
			name:     "imma_accumulator_only_rejected",
			sass:     kernelSASS("sm_121a", "IMMA.16832.S32", "EXIT"),
			wantDiag: "does not state an 8-bit integer tensor input format",
		},
		{
			name:     "non_tensor_bf16_rejected",
			sass:     kernelSASS("sm_121a", "FFMA", "FFMA", "HADD2.BF16_V2", "EXIT"),
			wantDiag: "non-tensor BF16 modifier",
		},
		{
			name:     "non_tensor_idp4a_rejected",
			sass:     kernelSASS("sm_121a", "FFMA", "IDP4A.S8.S8", "EXIT"),
			wantDiag: "non-tensor integer dot-product",
		},
		{
			name:     "quarantined_instruction_blocks_attribution",
			sass:     kernelSASS("sm_121a", "FFMA", "LDG.E.ENL2.256", "EXIT"),
			wantDiag: "quarantined instructions may be arithmetic operations",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := i.Report(context.Background(), image, Request{
				Symbol:   "kernel",
				Required: []string{"precision_attribution"},
				SASS:     tt.sass,
			})
			if !errors.Is(err, ErrRequiredFeature) || r == nil || r.Result != ResultFailed {
				t.Fatalf("Report = %+v, %v; want ResultFailed with ErrRequiredFeature", r, err)
			}
			if f := r.Feature("precision_attribution"); f == nil || f.State != StateUnknown || f.Known() {
				t.Fatalf("precision_attribution feature = %+v, want unknown", f)
			}
			if _, _, mixErr := r.InstructionMix(); !errors.Is(mixErr, ErrRequiredFeature) {
				t.Fatalf("InstructionMix error = %v, want ErrRequiredFeature", mixErr)
			}
			diag := diagnostics(r, "precision_attribution")
			if !strings.Contains(diag, tt.wantDiag) {
				t.Fatalf("diagnostics = %q, want substring %q", diag, tt.wantDiag)
			}
			if err := r.Validate(); err != nil {
				t.Fatalf("Validate failed on report with unavailable precision_attribution: %v", err)
			}
		})
	}

	// A SASS listing for another symbol is refused before any attribution can be claimed.
	wrongSym := strings.Replace(kernelSASS("sm_121a", "FFMA"), ".text.kernel", ".text.other", 1)
	if r, err := i.Report(context.Background(), image, Request{
		Symbol:   "kernel",
		Required: []string{"precision_attribution"},
		SASS:     wrongSym,
	}); !errors.Is(err, ErrSASS) || r != nil {
		t.Fatalf("wrong symbol SASS = %+v, %v; want nil, ErrSASS", r, err)
	}
}

func TestPrecisionAttributionValidateRejects(t *testing.T) {
	i, err := NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	good, err := i.Report(context.Background(), roundtripCUBIN(t), Request{
		Symbol:   "addk",
		Required: []string{"precision_attribution"},
		SASS:     roundtripSASS(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	clone := func() (*Report, PrecisionAttribution) {
		var buf bytes.Buffer
		if err := good.WriteJSON(&buf); err != nil {
			t.Fatal(err)
		}
		var r Report
		if err := json.Unmarshal(buf.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		pa, err := r.PrecisionAttribution()
		if err != nil {
			t.Fatal(err)
		}
		return &r, *pa
	}

	for _, tt := range []struct {
		name   string
		mutate func(*Report, *PrecisionAttribution)
	}{
		{"wrong_symbol", func(_ *Report, pa *PrecisionAttribution) { pa.Symbol = "intmix" }},
		{"forged_state", func(_ *Report, pa *PrecisionAttribution) { pa.State = StateInferred }},
		{"wrong_artifact_digest", func(_ *Report, pa *PrecisionAttribution) { pa.Artifact.SizeBytes++ }},
		{"wrong_sass_digest", func(_ *Report, pa *PrecisionAttribution) { pa.SASS.SizeBytes++ }},
		{"counter_ratio_method", func(_ *Report, pa *PrecisionAttribution) { pa.Method = cudacensus.FromCounterRatio }},
		{"unattributed_method", func(_ *Report, pa *PrecisionAttribution) { pa.Method = cudacensus.Unattributed }},
		{"known_false", func(_ *Report, pa *PrecisionAttribution) { pa.Known = false }},
		{"wrong_share", func(_ *Report, pa *PrecisionAttribution) { pa.Share = 0.5 }},
		{"wrong_dominant_precision", func(_ *Report, pa *PrecisionAttribution) { pa.Precision = cudaroofline.FP16Tensor }},
		{"counted_mismatch", func(_ *Report, pa *PrecisionAttribution) { pa.Counted = 2 }},
		{"opcode_mismatch", func(_ *Report, pa *PrecisionAttribution) { pa.Opcodes = map[string]uint64{"HMMA.16816.F32.F16": 1} }},
		{"non_arithmetic_in_opcodes", func(_ *Report, pa *PrecisionAttribution) { pa.Opcodes["LDG.E"] = 1 }},
		{"empty_limitations", func(_ *Report, pa *PrecisionAttribution) { pa.Limitations = nil }},
		{"predicated_over_counted", func(_ *Report, pa *PrecisionAttribution) { pa.Predicated = pa.Counted + 1 }},
		{"absent_feature_when_sass_present", func(r *Report, _ *PrecisionAttribution) {
			var filtered []Feature
			for _, f := range r.Features {
				if f.Name != "precision_attribution" {
					filtered = append(filtered, f)
				}
			}
			r.Features = filtered
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, pa := clone()
			tt.mutate(r, &pa)
			if f := r.Feature("precision_attribution"); f != nil {
				b, err := json.Marshal(pa)
				if err != nil {
					t.Fatal(err)
				}
				f.Value = b
			}
			if err := r.Validate(); !errors.Is(err, ErrInvalidReport) {
				t.Fatalf("Validate = %v, want ErrInvalidReport", err)
			}
			if _, _, err := r.InstructionMix(); !errors.Is(err, ErrInvalidReport) {
				t.Fatalf("InstructionMix = %v, want ErrInvalidReport", err)
			}
			var buf bytes.Buffer
			if err := r.WriteJSON(&buf); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadReport(&buf); !errors.Is(err, ErrInvalidReport) {
				t.Fatalf("ReadReport = %v, want ErrInvalidReport", err)
			}
		})
	}
}

func TestNilReportInstructionMixPreservesError(t *testing.T) {
	var r *Report
	_, paErr := r.PrecisionAttribution()
	if !errors.Is(paErr, ErrInvalidReport) {
		t.Fatalf("nil Report PrecisionAttribution error = %v, want ErrInvalidReport", paErr)
	}
	mix, state, mixErr := r.InstructionMix()
	if !errors.Is(mixErr, ErrInvalidReport) {
		t.Fatalf("nil Report InstructionMix error = %v, want ErrInvalidReport", mixErr)
	}
	if state != StateUnknown {
		t.Fatalf("nil Report InstructionMix state = %q, want %q", state, StateUnknown)
	}
	if len(mix.Counts) != 0 {
		t.Fatalf("nil Report InstructionMix counts = %+v, want empty", mix.Counts)
	}
	if mixErr.Error() != paErr.Error() {
		t.Fatalf("nil Report InstructionMix error = %q, want %q", mixErr.Error(), paErr.Error())
	}
}

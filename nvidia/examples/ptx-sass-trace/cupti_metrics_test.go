package main

import (
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestSassMetricsCensus(t *testing.T) {
	dis := &cudasass.Disassembly{
		Functions: []*cudasass.Function{
			{
				Name: "ds4_q8_0_matvec",
				Instructions: []cudasass.Instruction{
					{Address: 0x00, Opcode: "MOV"},
					{Address: 0x10, Opcode: "LDG.E"},
					{Address: 0x20, Opcode: "SHFL.BFLY"},
					{Address: 0x30, Opcode: "RET"},
				},
			},
			{
				Name: "ds4_iq2_simd",
				Instructions: []cudasass.Instruction{
					{Address: 0x00, Opcode: "IDP4A"},
					{Address: 0x10, Opcode: "RET"},
				},
			},
		},
	}

	res, data, err := collectSassMetrics(nil, dis)
	if err != nil {
		t.Fatalf("collectSassMetrics: %v", err)
	}
	if res.Census["ds4_q8_0_matvec"] != 4 {
		t.Errorf("expected census 4 for ds4_q8_0_matvec, got %d", res.Census["ds4_q8_0_matvec"])
	}
	if res.Census["ds4_iq2_simd"] != 2 {
		t.Errorf("expected census 2 for ds4_iq2_simd, got %d", res.Census["ds4_iq2_simd"])
	}

	var parsed sassMetricsResult
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal json data: %v", err)
	}
	if parsed.Status == "" {
		t.Error("expected non-empty status in json output")
	}
}

func TestPMSamplingDiagnostics(t *testing.T) {
	res, data, err := collectPMSampling(nil)
	if err != nil {
		t.Fatalf("collectPMSampling: %v", err)
	}
	if res == nil || len(data) == 0 {
		t.Fatal("expected non-nil result and non-empty json")
	}
}

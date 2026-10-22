package dsv4tools

import (
	"strconv"
	"strings"
	"testing"
)

func dispatchFixture(gate, up, down string) []ExpertLayoutRecord {
	records := make([]ExpertLayoutRecord, 43)
	for i := range records {
		records[i] = ExpertLayoutRecord{
			Layer: i, GateType: gate, UpType: up, DownType: down,
			GateDims: []uint64{4096, 2048, 256},
			UpDims:   []uint64{4096, 2048, 256},
			DownDims: []uint64{2048, 4096, 256},
		}
	}
	return records
}

func TestPlanExpertDispatchQ2(t *testing.T) {
	for _, backend := range []ExpertBackend{ExpertBackendCUDA, ExpertBackendMetal} {
		plan, err := PlanExpertDispatch(dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K"), backend)
		if err != nil {
			t.Fatalf("%s: PlanExpertDispatch: %v", backend, err)
		}
		if len(plan) != 43 {
			t.Fatalf("%s: plan has %d layers, want 43", backend, len(plan))
		}
		for layer, got := range plan {
			if got.Layer != layer || got.Format != "IQ2_XXS/Q2_K" || got.GateUpInput != "Q8_1" || got.DownInput != "Q8_1" {
				t.Errorf("%s: plan[%d] = %#v, want layer %d IQ2_XXS/Q2_K Q8_1/Q8_1", backend, layer, got, layer)
			}
		}
	}
}

func TestPlanExpertDispatchRejectsQ2Q4(t *testing.T) {
	for layer := 0; layer < 43; layer++ {
		t.Run(strconv.Itoa(layer), func(t *testing.T) {
			records := dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
			records[layer].GateType = "Q4_K"
			records[layer].UpType = "Q4_K"
			records[layer].DownType = "Q4_K"
			for _, backend := range []ExpertBackend{ExpertBackendCUDA, ExpertBackendMetal} {
				_, err := PlanExpertDispatch(records, backend)
				if err == nil {
					t.Errorf("%s: accepted an arbitrary single-Q4-layer layout", backend)
				}
			}
		})
	}
}

func TestPlanExpertDispatchScopeASplice(t *testing.T) {
	records := dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
	for layer := 37; layer < 43; layer++ {
		records[layer].GateType, records[layer].UpType, records[layer].DownType = "Q4_K", "Q4_K", "Q4_K"
	}
	plan, err := PlanExpertDispatch(records, ExpertBackendCUDA)
	if err != nil {
		t.Fatal(err)
	}
	for layer, got := range plan {
		want := "Q8_1"
		if layer >= 37 {
			want = "Q8_K"
		}
		if got.GateUpInput != want || got.DownInput != want {
			t.Fatalf("layer %d: activation contract %+v, want %s", layer, got, want)
		}
	}
	for _, backend := range []ExpertBackend{ExpertBackendCUDA, ExpertBackendMetal} {
		if _, err := PlanExpertDispatch(dispatchFixture("Q4_K", "Q4_K", "Q4_K"), backend); err == nil {
			t.Fatalf("%s admitted scope-B all-Q4", backend)
		}
	}
	metal, err := PlanExpertDispatch(records, ExpertBackendMetal)
	if err != nil {
		t.Fatal(err)
	}
	for layer, got := range metal {
		want := "Q8_1"
		if layer >= 37 {
			want = "F32"
		}
		if got.GateUpInput != want || got.DownInput != want {
			t.Fatalf("Metal layer %d: %+v", layer, got)
		}
	}
}

func TestPlanExpertDispatchRejectsCPUAndUnknown(t *testing.T) {
	records := dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
	if _, err := PlanExpertDispatch(records, ExpertBackendCPU); err == nil || !strings.Contains(err.Error(), "no DSV4 expert kernel") {
		t.Fatalf("CPU error = %v", err)
	}
	if _, err := PlanExpertDispatch(records, ExpertBackend("vulkan")); err == nil || !strings.Contains(err.Error(), "backend") {
		t.Fatalf("unknown backend error = %v", err)
	}
}

func TestPlanExpertDispatchRejectsMalformedLayout(t *testing.T) {
	records := dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
	records[12].Layer = 13
	if _, err := PlanExpertDispatch(records, ExpertBackendCUDA); err == nil || !strings.Contains(err.Error(), "expert dispatch layout") {
		t.Fatalf("malformed layout error = %v", err)
	}
}

func TestClassifyExpertLayoutVariants(t *testing.T) {
	tests := []struct {
		name string
		make func(int) (string, string, string)
		want ExpertVariant
	}{
		{"iq2-q2k", func(int) (string, string, string) { return "IQ2_XXS", "IQ2_XXS", "Q2_K" }, ExpertVariantIQ2Q2K},
		{"iq2-q4k", func(layer int) (string, string, string) {
			if layer >= 37 {
				return "Q4_K", "Q4_K", "Q4_K"
			}
			return "IQ2_XXS", "IQ2_XXS", "Q2_K"
		}, ExpertVariantIQ2Q4K},
		{"q4k", func(int) (string, string, string) { return "Q4_K", "Q4_K", "Q4_K" }, ExpertVariantQ4K},
		{"mxfp4", func(int) (string, string, string) { return "MXFP4", "MXFP4", "MXFP4" }, ExpertVariantMXFP4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			records := make([]ExpertLayoutRecord, 43)
			for layer := range records {
				gate, up, down := tt.make(layer)
				records[layer] = ExpertLayoutRecord{
					Layer: layer, GateType: gate, UpType: up, DownType: down,
					GateDims: []uint64{4096, 2048, 256}, UpDims: []uint64{4096, 2048, 256}, DownDims: []uint64{2048, 4096, 256},
				}
			}
			got, err := ClassifyExpertLayout(records)
			if err != nil || got != tt.want {
				t.Fatalf("ClassifyExpertLayout = %v, %v; want %v", got, err, tt.want)
			}
		})
	}
}

func TestClassifyExpertLayoutRejectsSpliceMutations(t *testing.T) {
	records := dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
	records[36].GateType, records[36].UpType, records[36].DownType = "Q4_K", "Q4_K", "Q4_K"
	if _, err := ClassifyExpertLayout(records); err == nil {
		t.Fatal("accepted Q4 layer before the q2-q4 splice boundary")
	}
	records = dispatchFixture("IQ2_XXS", "IQ2_XXS", "Q2_K")
	records[37].GateType, records[37].UpType, records[37].DownType = "Q4_K", "Q4_K", "Q2_K"
	if _, err := ClassifyExpertLayout(records); err == nil {
		t.Fatal("accepted mixed Q4 gate/up and Q2 down")
	}
}

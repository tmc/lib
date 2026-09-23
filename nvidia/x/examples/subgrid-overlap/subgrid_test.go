package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestPTXEmissionAndAudit(t *testing.T) {
	tests := []struct {
		name    string
		cfg     PTXConfig
		wantErr bool
	}{
		{
			name:    "default sm_90 PTX 7.8",
			cfg:     DefaultPTXConfig(),
			wantErr: false,
		},
		{
			name: "sm_90a PTX 8.0",
			cfg: PTXConfig{
				Version:     "8.0",
				Target:      "sm_90a",
				AddressSize: 64,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ptxText := ModulePTX(tt.cfg)
			m, err := AuditModule(ptxText)
			if (err != nil) != tt.wantErr {
				t.Fatalf("AuditModule() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}

			if m.Target != tt.cfg.Target {
				t.Errorf("module target = %s, want %s", m.Target, tt.cfg.Target)
			}

			// Verify producer opcodes
			if err := m.AssertOpcodes("producer_subgrid", []string{
				"griddepcontrol.launch_dependents",
				"fence.sc.gpu",
			}, nil); err != nil {
				t.Errorf("producer missing required opcode: %v", err)
			}

			// Verify consumer opcodes
			if err := m.AssertOpcodes("consumer_subgrid", []string{
				"griddepcontrol.wait",
			}, nil); err != nil {
				t.Errorf("consumer missing required opcode: %v", err)
			}

			// Verify architecture audit
			if err := m.AuditArchitecture(""); err != nil {
				t.Errorf("AuditArchitecture failed: %v", err)
			}
		})
	}
}

func TestPTXOpcodeAuditNegative(t *testing.T) {
	// Asserting a missing opcode should fail
	ptxText := ModulePTX(DefaultPTXConfig())
	m, err := cudaptx.Parse(strings.NewReader(ptxText))
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	err = m.AssertOpcodes("producer_subgrid", []string{"nonexistent_opcode"}, nil)
	if err == nil {
		t.Errorf("expected error asserting nonexistent opcode, got nil")
	}

	err = m.AssertOpcodes("producer_subgrid", nil, []string{"griddepcontrol.launch_dependents"})
	if err == nil {
		t.Errorf("expected error when forbidding griddepcontrol.launch_dependents, got nil")
	}
}

func TestSimulationInvariants(t *testing.T) {
	hw := DefaultHardware()

	tests := []struct {
		name       string
		gridDim    int
		numTiles   int
		tileCycles int
		triggerPct float64
	}{
		{"quarter trigger 4 tiles", 64, 4, 10000, 0.25},
		{"half trigger 4 tiles", 132, 4, 20000, 0.50},
		{"three-quarter trigger 8 tiles", 132, 8, 15000, 0.75},
		{"oversubscribed half trigger", 264, 8, 10000, 0.50},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wl := WorkloadConfig{
				GridDimX:           tt.gridDim,
				BlockDimX:          256,
				NumTiles:           tt.numTiles,
				TileCycles:         tt.tileCycles,
				TriggerPercent:     tt.triggerPct,
				HostLaunchLatency:  10000,
				GraphLaunchLatency: 1000,
			}

			// 1. Sequential unpipelined
			resSeq, err := Simulate(SimConfig{Hardware: hw, Workload: wl, Mode: ModeUnpipelined})
			if err != nil {
				t.Fatalf("Simulate(Unpipelined) error: %v", err)
			}
			if resSeq.OverlapCycles != 0 {
				t.Errorf("sequential unpipelined overlap = %d, want 0", resSeq.OverlapCycles)
			}

			// 2. Sequential CUDA graph
			resGraph, err := Simulate(SimConfig{Hardware: hw, Workload: wl, Mode: ModeSequentialGraph})
			if err != nil {
				t.Fatalf("Simulate(Graph) error: %v", err)
			}
			if resGraph.OverlapCycles != 0 {
				t.Errorf("sequential graph overlap = %d, want 0", resGraph.OverlapCycles)
			}
			if resSeq.TotalCycles <= resGraph.TotalCycles {
				t.Errorf("expected unpipelined cycles (%d) > graph cycles (%d)", resSeq.TotalCycles, resGraph.TotalCycles)
			}

			// 3. PDL Sub-Grid Overlap
			resPDL, err := Simulate(SimConfig{Hardware: hw, Workload: wl, Mode: ModePDLOverlap})
			if err != nil {
				t.Fatalf("Simulate(PDL) error: %v", err)
			}

			// Invariant: PDL must exhibit intra-kernel pipeline overlap
			if resPDL.OverlapCycles <= 0 {
				t.Errorf("PDL overlap cycles = %d, want > 0", resPDL.OverlapCycles)
			}
			if resPDL.OverlapPercent <= 0.0 {
				t.Errorf("PDL overlap percent = %.2f%%, want > 0", resPDL.OverlapPercent)
			}

			// Invariant: Total PDL latency must be strictly less than sequential graph latency
			if resPDL.TotalCycles >= resGraph.TotalCycles {
				t.Errorf("PDL total cycles (%d) >= Graph total cycles (%d)", resPDL.TotalCycles, resGraph.TotalCycles)
			}

			// Invariant: Consumer start cycle must match or follow PDL trigger cycle
			if resPDL.ConsumerStartCycle < resPDL.PDLTriggerCycle {
				t.Errorf("consumer start (%d) < trigger cycle (%d)", resPDL.ConsumerStartCycle, resPDL.PDLTriggerCycle)
			}

			// Invariant: Consumer end cycle must be >= producer end cycle (due to dependency wait)
			if resPDL.ConsumerEndCycle < resPDL.ProducerEndCycle {
				t.Errorf("consumer end (%d) < producer end (%d)", resPDL.ConsumerEndCycle, resPDL.ProducerEndCycle)
			}

			// Invariant: Mean SM occupancy with PDL must be positive and within bounds
			if resPDL.AverageOccupancy <= 0 || resPDL.AverageOccupancy > 1.0 {
				t.Errorf("invalid PDL average occupancy: %f", resPDL.AverageOccupancy)
			}
		})
	}
}

func TestBenchmarkSuiteExecution(t *testing.T) {
	suite := DefaultBenchmarkSuite()
	report, err := suite.Run()
	if err != nil {
		t.Fatalf("BenchmarkSuite.Run() error = %v", err)
	}

	if len(report.Scenarios) != 3 {
		t.Fatalf("expected 3 scenarios, got %d", len(report.Scenarios))
	}

	// Verify Table output is non-empty and well-formed
	table := report.Table()
	if !strings.Contains(table, "PDL Overlap (50%)") {
		t.Errorf("table missing PDL 50%% row")
	}

	// Verify Markdown output contains sections and chart
	md := report.Markdown()
	if !strings.Contains(md, "# Sub-Grid Early Release") {
		t.Errorf("markdown missing title")
	}
	if !strings.Contains(md, "SM Occupancy Timeline") {
		t.Errorf("markdown missing timeline chart")
	}

	// Verify JSON output parses correctly
	data, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON() error: %v", err)
	}
	var unmarshaled SuiteReport
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal generated JSON: %v", err)
	}
	if len(unmarshaled.Scenarios) != 3 {
		t.Errorf("unmarshaled scenarios = %d, want 3", len(unmarshaled.Scenarios))
	}
}

func BenchmarkSimulator(b *testing.B) {
	cfg := SimConfig{
		Hardware: DefaultHardware(),
		Workload: DefaultWorkload(),
		Mode:     ModePDLOverlap,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Simulate(cfg)
	}
}

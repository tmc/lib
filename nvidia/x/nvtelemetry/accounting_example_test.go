package nvtelemetry_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"io"
	"os"
)

func ExampleReadAccounting() {
	f, err := os.Open("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	report, err := nvtelemetry.ReadAccounting(f, 1<<20)
	fmt.Println(report.PhysicalEnergyMillijoules, report.ResidualEnergyMillijoules, err)
	// Output: 200000 75000 <nil>
}
func ExampleAccountEnergy() {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	var input nvtelemetry.AccountingInput
	json.Unmarshal(raw, &input)
	report, err := nvtelemetry.AccountEnergy(input)
	fmt.Println(report.Shares[0].CostMajor, report.Shares[0].Currency, err)
	// Output: 3/2 USD <nil>
}
func ExampleWriteAccounting() {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	var input nvtelemetry.AccountingInput
	json.Unmarshal(raw, &input)
	var wire bytes.Buffer
	err = nvtelemetry.WriteAccounting(&wire, input)
	report, readErr := nvtelemetry.ReadAccounting(&wire, 1<<20)
	fmt.Println(report.Shares[0].RoundedMinor, err, readErr)
	// Output: 150 <nil> <nil>
}
func ExampleWriteAccountingReport() {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	report, err := nvtelemetry.ReadAccounting(bytes.NewReader(raw), 1<<20)
	fmt.Println(err, nvtelemetry.WriteAccountingReport(io.Discard, report))
	// Output: <nil> <nil>
}

func ExampleAccountTelemetry() {
	// Both files are synthetic known-answer contracts, not native receipts.
	source, err := os.Open("testdata/accounting_v2.jsonl")
	if err != nil {
		panic(err)
	}
	defer source.Close()
	policies, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	var input nvtelemetry.AccountingInput
	json.Unmarshal(policies, &input)
	input.Energy = nil
	report, err := nvtelemetry.AccountTelemetry(source, input, "measurement", 3600000000000, 10800000000000, 1<<20)
	fmt.Println(report.PhysicalEnergyMillijoules, report.Shares[0].CostMajor, err)
	// Output: 200000 3/2 <nil>
}

func ExampleAccountEnergy_costOnly() {
	raw, err := os.ReadFile("testdata/accounting_v1.json")
	if err != nil {
		panic(err)
	}
	var input nvtelemetry.AccountingInput
	json.Unmarshal(raw, &input)
	input.Energy = nil
	report, err := nvtelemetry.AccountEnergy(input)
	fmt.Println(report.EnergyAvailable, report.Shares[0].CostMajor, report.Shares[0].EfficiencyUnavailable, err)
	// Output: false 3/2 physical_energy_unavailable <nil>
}

func ExampleAccountingNode() {
	raw, _ := os.ReadFile("testdata/accounting_v1.json")
	var in nvtelemetry.AccountingInput
	json.Unmarshal(raw, &in)
	gpu := in.Energy[0]
	node := gpu
	node.Scope, node.Lineage, node.Summary.DeviceUUID = "whole_node", "node-counter", "node-1"
	node.Summary.EnergyConsumedMJ = 300000
	node.Summary.EndEnergyMJ = node.Summary.StartEnergyMJ + 300000
	node.Summary.EnergyConsumedJoules = 300
	in.Nodes = []nvtelemetry.AccountingNode{{NodeID: "node-1", Energy: node, DeviceUUIDs: []string{gpu.Summary.DeviceUUID}, MembershipEvidence: "synthetic inventory receipt"}}
	report, err := nvtelemetry.AccountEnergy(in)
	fmt.Println(report.Nodes[0].NodeEnergyMillijoules, report.Nodes[0].GPUComponentMillijoules, report.Nodes[0].OtherEnergyMillijoules, err)
	// Output: 300000 200000 100000 <nil>
}

func ExampleAccountingIdle() {
	raw, _ := os.ReadFile("testdata/accounting_v1.json")
	var in nvtelemetry.AccountingInput
	json.Unmarshal(raw, &in)
	in.Allocations, in.Rates, in.Work = nil, nil, nil
	e := in.Energy[0].Summary
	in.Idle = []nvtelemetry.AccountingIdle{{DeviceUUID: e.DeviceUUID, ClockDomain: in.Clock.Domain, ClockEpoch: in.Clock.Epoch, StartNanos: e.StartMonoNanos, EndNanos: e.EndMonoNanos, CapacityPermille: 250, Authoritative: true, Evidence: "synthetic idle-capacity receipt", Policy: "capacity_time_v1"}}
	report, err := nvtelemetry.AccountEnergy(in)
	fmt.Println(report.IdleEnergyMillijoules, report.UnclassifiedResidualMillijoules, err)
	// Output: 50000 150000 <nil>
}

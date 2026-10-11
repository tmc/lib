package nvtelemetry_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleNewNVLinkReader() {
	d, c, cfg := linkFixture()
	reader, _ := nvtelemetry.NewNVLinkReader(d, c, cfg)
	report, _ := reader.Read(context.Background())
	fmt.Println(report.Links[0].PeerUUID)
	// Output: GPU-1
}
func ExampleNVLinkReader_Read() {
	d, c, cfg := linkFixture()
	reader, _ := nvtelemetry.NewNVLinkReader(d, c, cfg)
	report, _ := reader.Read(context.Background())
	fmt.Println(report.Links[0].RemotePCI, report.Links[0].LinkVersion)
	// Output: 00000000:03:00.0 4
}
func ExampleNVLinkSnapshot_Validate() {
	d, c, cfg := linkFixture()
	reader, _ := nvtelemetry.NewNVLinkReader(d, c, cfg)
	report, _ := reader.Read(context.Background())
	fmt.Println(report.Validate())
	// Output: <nil>
}
func ExampleReadNVLinkSnapshot() {
	d, c, cfg := linkFixture()
	reader, _ := nvtelemetry.NewNVLinkReader(d, c, cfg)
	report, _ := reader.Read(context.Background())
	raw, _ := json.Marshal(report)
	parsed, _ := nvtelemetry.ReadNVLinkSnapshot(strings.NewReader(string(raw)), 8192)
	fmt.Println(parsed.Version, len(parsed.Links))
	// Output: 1 1
}
func ExampleJoinNVLinkHealth() {
	d, c, cfg := linkFixture()
	reader, _ := nvtelemetry.NewNVLinkReader(d, c, cfg)
	report, _ := reader.Read(context.Background())
	s := nvtelemetry.HealthSnapshot{Version: 1, NodeID: "n", TopologyEpoch: "t", ClockDomain: report.ClockDomain, ClockEpoch: report.ClockEpoch, AtNanos: 10, DeviceStates: map[string]string{"GPU-0": "present", "GPU-1": "present"}}
	limits := nvtelemetry.HealthLimits{MaxBytes: 8192, MaxDevices: 2, MaxSources: 2, MaxObservations: 4, MaxEdges: 4}
	out, _ := nvtelemetry.JoinNVLinkHealth(s, *report, limits)
	fmt.Println(out.Edges[0].From, out.Edges[0].To, out.Edges[0].Capability)
	// Output: GPU-0 GPU-1 nominal_version=4
}

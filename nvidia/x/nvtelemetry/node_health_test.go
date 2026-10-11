package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/ebpfhost/hostmetrics"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func nodeFixture() (nvtelemetry.HealthSnapshot, hostmetrics.Snapshot, nvtelemetry.HealthLimits) {
	l := nvtelemetry.HealthLimits{MaxBytes: 4096, MaxDevices: 1, MaxSources: 2, MaxObservations: 8, MaxEdges: 1}
	s := nvtelemetry.HealthSnapshot{Version: 1, NodeID: "n", TopologyEpoch: "t", ClockDomain: "mono", ClockEpoch: "b", AtNanos: 20, DeviceStates: map[string]string{"GPU": "present"}}
	n, _ := hostmetrics.ReadCounter(strings.NewReader("9007199254740993"), hostmetrics.Context{NodeID: "n", SourceID: "net", Version: "linux6.8", ClockDomain: "mono", ClockEpoch: "b", QueryStartNanos: 10, QueryEndNanos: 20}, "node.net.rx_bytes", "eth0", 128)
	return s, n, l
}
func ExampleJoinNodeHealth() {
	s, n, l := nodeFixture()
	joined, _ := nvtelemetry.JoinNodeHealth(s, n, "GPU", l)
	var b bytes.Buffer
	_ = nvtelemetry.WriteHealthSnapshot(&b, *joined, l)
	got, _ := nvtelemetry.ReadHealthSnapshot(&b, l)
	fmt.Println(got.Observations[0].Reading.IntValue, got.Observations[0].Scope)
	// Output: 9007199254740993 node/n/eth0
}
func TestNodeRefusal(t *testing.T) {
	s, n, l := nodeFixture()
	for _, mutate := range []func(*hostmetrics.Snapshot){func(n *hostmetrics.Snapshot) { n.Context.NodeID = "other" }, func(n *hostmetrics.Snapshot) { n.Context.ClockEpoch = "other" }, func(n *hostmetrics.Snapshot) { n.Context.QueryEndNanos = 21 }, func(n *hostmetrics.Snapshot) { n.Metrics[0].Unit = "packets" }} {
		x := n
		x.Metrics = append([]hostmetrics.Metric(nil), n.Metrics...)
		mutate(&x)
		if _, e := nvtelemetry.JoinNodeHealth(s, x, "GPU", l); e == nil {
			t.Fatal("accepted bad join")
		}
	}
	if _, e := nvtelemetry.JoinNodeHealth(s, n, "unknown", l); e == nil {
		t.Fatal("accepted unknown device")
	}
}

func ExampleJoinRedfishHealth() {
	s, _, l := nodeFixture()
	raw := `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/p","Id":"p","ReadingType":"Power","ReadingUnits":"W","Implementation":"PhysicalSensor","Status":{"State":"Enabled"},"Reading":null}`
	sensor, _ := hostmetrics.ReadRedfishSensor(strings.NewReader(raw), hostmetrics.Context{NodeID: "n", SourceID: "bmc", Version: "Sensor.v1_9_0", ClockDomain: "mono", ClockEpoch: "b", QueryEndNanos: 20}, "/redfish/v1/Chassis/1/Sensors/p", "p", 1024)
	out, _ := nvtelemetry.JoinRedfishHealth(s, sensor, "GPU", l)
	fmt.Println(out.Observations[0].Reading.Evidence, out.Observations[0].Reading.HasSourceTimestamp)
	// Output: unavailable false
}

func TestNodeCompositionOwnership(t *testing.T) {
	s, n, l := nodeFixture()
	out, e := nvtelemetry.JoinNodeHealth(s, n, "GPU", l)
	if e != nil {
		t.Fatal(e)
	}
	out.DeviceStates["GPU"] = "lost"
	if s.DeviceStates["GPU"] != "present" {
		t.Fatal("aliased device map")
	}
	n.Metrics[0].Value = 0
	if out.Observations[0].Reading.Value == 0 {
		t.Fatal("aliased node input")
	}
}
func TestRedfishComposition(t *testing.T) {
	s, _, l := nodeFixture()
	base := hostmetrics.RedfishSensor{Context: hostmetrics.Context{NodeID: "n", SourceID: "bmc", Version: "Sensor.v1_9_0", ClockDomain: "mono", ClockEpoch: "b", QueryEndNanos: 20}, ID: "sensor", URI: "/redfish/v1/Chassis/1/Sensors/sensor", ReadingType: "Temperature", Unit: "Cel", ImplementationType: "Synthesized", State: "Enabled"}
	v := -5.0
	base.Reading = &v
	out, e := nvtelemetry.JoinRedfishHealth(s, base, "GPU", l)
	if e != nil {
		t.Fatal(e)
	}
	o := out.Observations[0]
	if o.Reading.Evidence != nvtelemetry.EvidenceDerived || o.Class != "inference" || o.Reading.HasSourceTimestamp {
		t.Fatal(o)
	}
	out.Observations[0].Scope = "GPU"
	if e = out.Validate(l); e == nil {
		t.Fatal("accepted lost node scope")
	}
}

func TestRedfishInactiveComposition(t *testing.T) {
	for _, state := range []string{"Disabled", "Absent", "UnavailableOffline", "StandbyOffline", "StandbySpare", "Starting", "Updating", ""} {
		t.Run(state, func(t *testing.T) {
			s, _, l := nodeFixture()
			raw := `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/p","Id":"p","ReadingType":"Power","ReadingUnits":"W","Implementation":"Synthesized","Status":{"State":"` + state + `","Health":"OK"},"Reading":125.5}`
			if state == "" {
				raw = strings.Replace(raw, `"State":""`, `"State":null`, 1)
			}
			sensor, e := hostmetrics.ReadRedfishSensor(strings.NewReader(raw), hostmetrics.Context{NodeID: "n", SourceID: "bmc", Version: "Sensor.v1_9_0", ClockDomain: "mono", ClockEpoch: "b", QueryEndNanos: 20}, "/redfish/v1/Chassis/1/Sensors/p", "p", 1024)
			if e != nil {
				t.Fatal(e)
			}
			out, e := nvtelemetry.JoinRedfishHealth(s, sensor, "GPU", l)
			if e != nil {
				t.Fatal(e)
			}
			var b bytes.Buffer
			if e = nvtelemetry.WriteHealthSnapshot(&b, *out, l); e != nil {
				t.Fatal(e)
			}
			got, e := nvtelemetry.ReadHealthSnapshot(&b, l)
			if e != nil {
				t.Fatal(e)
			}
			o := got.Observations[0]
			if o.Reading.Evidence != nvtelemetry.EvidenceUnavailable || o.Reading.Value != 0 || o.Reading.UnavailableReason == "" || o.Reading.HasSourceTimestamp || o.Judgment != "" || o.Class != "raw" {
				t.Fatal(o)
			}
			if sensor.Reading == nil || *sensor.Reading != 125.5 || sensor.Health != "OK" {
				t.Fatal("lost original source evidence", sensor)
			}
		})
	}
}

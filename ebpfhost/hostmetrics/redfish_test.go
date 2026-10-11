package hostmetrics

import (
	"fmt"
	"strings"
	"testing"
)

const sensor = `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/power","Id":"power","ReadingType":"Power","ReadingUnits":"W","Implementation":"PhysicalSensor","Status":{"State":"Enabled","Health":"Warning"},"Reading":125.5}`

func ExampleReadRedfishSensor() {
	s, _ := ReadRedfishSensor(strings.NewReader(sensor), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024)
	fmt.Println(*s.Reading, s.Health)
	// Output: 125.5 Warning
}
func ExampleRedfishSensor_Validate() {
	s, _ := ReadRedfishSensor(strings.NewReader(sensor), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024)
	fmt.Println(s.Validate())
	// Output: <nil>
}
func TestRedfishNegative(t *testing.T) {
	for _, x := range []string{strings.Replace(sensor, `"ReadingUnits":"W"`, `"ReadingUnits":"kW"`, 1), strings.Replace(sensor, `"Reading":125.5`, `"Reading":-1`, 1), strings.Replace(sensor, `"Reading":125.5`, `"Reading":125.5,"Reading":0`, 1), strings.Replace(sensor, `"Reading":125.5`, `"Reading":"1"`, 1), strings.Replace(sensor, `v1_9_0`, `v1_8_0`, 1), strings.Replace(sensor, `"Id":"power"`, `"Id":"other"`, 1)} {
		if _, e := ReadRedfishSensor(strings.NewReader(x), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024); e == nil {
			t.Fatal("accepted invalid Redfish", x)
		}
	}
	s, e := ReadRedfishSensor(strings.NewReader(strings.Replace(sensor, `"Reading":125.5`, `"Reading":null`, 1)), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024)
	if e != nil || s.Reading != nil {
		t.Fatal(s, e)
	}
}

func TestRedfishInactiveReadingRetained(t *testing.T) {
	for _, state := range []string{"Disabled", "Absent", "UnavailableOffline", "StandbyOffline", "StandbySpare", "Starting", "Updating", ""} {
		t.Run(state, func(t *testing.T) {
			raw := strings.Replace(sensor, `"State":"Enabled"`, `"State":"`+state+`"`, 1)
			if state == "" {
				raw = strings.Replace(sensor, `"State":"Enabled"`, `"State":null`, 1)
			}
			s, e := ReadRedfishSensor(strings.NewReader(raw), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024)
			if e != nil || s.Reading == nil || *s.Reading != 125.5 || s.State != state || s.Health != "Warning" {
				t.Fatalf("%+v %v", s, e)
			}
		})
	}
	raw := strings.Replace(sensor, `"State":"Enabled",`, "", 1)
	s, e := ReadRedfishSensor(strings.NewReader(raw), fixture(), "/redfish/v1/Chassis/1/Sensors/power", "power", 1024)
	if e != nil || s.State != "" || s.Reading == nil {
		t.Fatal(s, e)
	}
}

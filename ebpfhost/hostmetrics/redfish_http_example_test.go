package hostmetrics_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"time"

	"github.com/tmc/lib/ebpfhost/hostmetrics"
)

func ExampleRedfishHTTPClient_GetSensor() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"@odata.type":"#Sensor.v1_9_0.Sensor","@odata.id":"/redfish/v1/Chassis/1/Sensors/power","Id":"power","ReadingType":"Power","ReadingUnits":"W","Implementation":"PhysicalSensor","Reading":250}`)
	}))
	defer server.Close()
	client, err := hostmetrics.NewRedfishHTTPClient(hostmetrics.RedfishHTTPConfig{BaseURL: server.URL, Client: server.Client(), MaxResponseBytes: 4096, RequestTimeout: time.Second, MaxCollectionMembers: 16, MaxPages: 2})
	if err != nil {
		fmt.Println(err)
		return
	}
	source := hostmetrics.Context{NodeID: "node-1", SourceID: "bmc-1", Version: "fixture", ClockDomain: "process", ClockEpoch: "epoch-1", QueryStartNanos: 0, QueryEndNanos: 1}
	sensor, err := client.GetSensor(context.Background(), source, "/redfish/v1/Chassis/1/Sensors/power", "power")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(*sensor.Reading, sensor.Unit)
	// Output: 250 W
}
func ExampleRedfishHTTPClient_GetCollection() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"@odata.id":"/redfish/v1/Chassis","Members@odata.count":1,"Members":[{"@odata.id":"/redfish/v1/Chassis/1"}]}`)
	}))
	defer server.Close()
	client, err := hostmetrics.NewRedfishHTTPClient(hostmetrics.RedfishHTTPConfig{BaseURL: server.URL, Client: server.Client(), MaxResponseBytes: 4096, RequestTimeout: time.Second, MaxCollectionMembers: 16, MaxPages: 2})
	if err != nil {
		fmt.Println(err)
		return
	}
	collection, err := client.GetCollection(context.Background(), "/redfish/v1/Chassis")
	fmt.Println(collection.Count, collection.HasAdvertisedCount, err)
	// Output: 1 true <nil>
}
func ExampleReadRedfishCollection() {
	raw := `{"@odata.id":"/redfish/v1/Chassis","Members":[{"@odata.id":"/redfish/v1/Chassis/1"}]}`
	page, err := hostmetrics.ReadRedfishCollection(strings.NewReader(raw), "/redfish/v1/Chassis", 4096)
	fmt.Println(len(page.Members), page.HasAdvertisedCount, err)
	// Output: 1 false <nil>
}
func ExampleRedfishHTTPConfig() {
	config := hostmetrics.RedfishHTTPConfig{BaseURL: "https://bmc.example", Client: http.DefaultClient, MaxResponseBytes: 4096, RequestTimeout: time.Second, MaxCollectionMembers: 16, MaxPages: 2}
	_, err := hostmetrics.NewRedfishHTTPClient(config)
	fmt.Println(err)
	// Output: <nil>
}
func ExampleRedfishCollectionPage() {
	page := hostmetrics.RedfishCollectionPage{URI: "/redfish/v1/Chassis", Members: []string{"/redfish/v1/Chassis/1"}}
	fmt.Println(len(page.Members), page.HasAdvertisedCount)
	// Output: 1 false
}
func ExampleRedfishCollection() {
	collection := hostmetrics.RedfishCollection{URI: "/redfish/v1/Chassis", Count: 1, Members: []string{"/redfish/v1/Chassis/1"}}
	fmt.Println(collection.Count, collection.HasAdvertisedCount)
	// Output: 1 false
}
func ExampleRedfishHTTPClient() {
	var client hostmetrics.RedfishHTTPClient
	_, err := client.GetCollection(context.Background(), "/redfish/v1/Chassis")
	fmt.Println(err != nil)
	// Output: true
}

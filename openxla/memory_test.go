package openxla

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/ebitengine/purego"
)

func TestMemoryStats(t *testing.T) {
	for _, mode := range []string{"absent", "zero", "values", "error"} {
		t.Run(mode, func(t *testing.T) {
			device := &Device{}
			calls, destroys := 0, 0
			api := &Api{}
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { destroys++ })
			api.Device_MemoryStats = purego.NewCallback(func(a *Device_MemoryStats_Args) *ErrorHandle {
				calls++
				if a.Device != device || a.Struct_size != PJRT_Device_MemoryStats_Args_STRUCT_SIZE || a.Extension_start != nil {
					t.Errorf("invalid arguments: %+v", a)
				}
				a.Bytes_in_use = 1024
				// Distinct nonzero values in absent fields must be ignored.
				v := reflect.ValueOf(a).Elem()
				for i := 4; i < v.NumField(); i += 2 {
					n := int64(i)
					if mode == "zero" {
						n = 0
					}
					v.Field(i).SetInt(n)
					v.Field(i + 1).SetBool(mode == "zero" || mode == "values")
				}
				if mode == "error" {
					return &ErrorHandle{}
				}
				return nil
			})
			c := &ClientWrapper{api: api, client: &Client{}}
			got, err := c.MemoryStats(device)
			if calls != 1 {
				t.Fatalf("calls=%d", calls)
			}
			if mode == "error" {
				if err == nil || destroys != 1 || !reflect.DeepEqual(got, DeviceMemoryStats{}) {
					t.Fatalf("got %+v, %v; destroys=%d", got, err, destroys)
				}
				return
			}
			if err != nil || got.BytesInUse != 1024 || destroys != 0 {
				t.Fatalf("got %+v, %v; destroys=%d", got, err, destroys)
			}
			v := reflect.ValueOf(got)
			for i := 1; i < v.NumField(); i++ {
				p := v.Field(i)
				if mode == "absent" {
					if !p.IsNil() {
						t.Errorf("%s should be absent", v.Type().Field(i).Name)
					}
					continue
				}
				want := int64(2*i + 2)
				if mode == "zero" {
					want = 0
				}
				if p.IsNil() || p.Elem().Int() != want {
					t.Errorf("%s: got %v, want %d", v.Type().Field(i).Name, p, want)
				}
			}
		})
	}
}

func TestMemoryStatsUnavailable(t *testing.T) {
	for _, c := range []*ClientWrapper{nil, {}, {api: &Api{}}, {api: &Api{}, client: &Client{}}} {
		if _, err := c.MemoryStats(&Device{}); err == nil {
			t.Fatal("accepted unavailable memory statistics")
		}
	}
	c := &ClientWrapper{api: &Api{Device_MemoryStats: 1}, client: &Client{}}
	if _, err := c.MemoryStats(nil); err == nil {
		t.Fatal("accepted nil device")
	}
}

func ExampleDeviceMemoryStats() {
	zero := int64(0)
	stats := DeviceMemoryStats{BytesInUse: 1024, NumAllocs: &zero}
	fmt.Println(stats.BytesInUse, *stats.NumAllocs, stats.BytesLimit == nil)
	// Output: 1024 0 true
}

func ExampleClientWrapper_MemoryStats() {
	var client *ClientWrapper
	_, err := client.MemoryStats(nil)
	fmt.Println(err)
	// Output: openxla: client is closed
}

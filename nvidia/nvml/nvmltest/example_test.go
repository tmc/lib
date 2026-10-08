package nvmltest_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/nvml/nvmltest"
)

func Example() {
	script := nvmltest.Recording{Version: nvmltest.Version, Provenance: nvmltest.Provenance{Source: "synthetic example"}, Calls: []nvmltest.Call{{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)}}}
	device, err := nvmltest.NewReplay(script)
	if err != nil {
		panic(err)
	}
	supported, err := device.GpmSupported()
	fmt.Println(supported, err, device.Done())
	// Output: true <nil> true
}

func ExampleRead() {
	rec, err := nvmltest.Read(strings.NewReader(`{"version":1,"provenance":{},"calls":[]}`))
	fmt.Println(rec.Version, len(rec.Calls), err)
	// Output: 1 0 <nil>
}

func ExampleWrite() {
	var output bytes.Buffer
	err := nvmltest.Write(&output, nvmltest.Recording{Version: nvmltest.Version})
	rec, readErr := nvmltest.Read(&output)
	fmt.Println(rec.Version, err, readErr)
	// Output: 1 <nil> <nil>
}

func exampleQuery(method, args, result string) *nvmltest.Replay {
	rec := nvmltest.Recording{Version: nvmltest.Version, Provenance: nvmltest.Provenance{Source: "synthetic example"}, Calls: []nvmltest.Call{{Method: method, Args: json.RawMessage(args), Result: json.RawMessage(result)}}}
	q, err := nvmltest.NewReplay(rec)
	if err != nil {
		panic(err)
	}
	return q
}

func exampleRecorder(method, args, result string) *nvmltest.Recorder {
	r, err := nvmltest.NewRecorder(exampleQuery(method, args, result), nvmltest.Provenance{Source: "synthetic example"})
	if err != nil {
		panic(err)
	}
	return r
}

func ExampleNewReplay() {
	q, err := nvmltest.NewReplay(nvmltest.Recording{Version: nvmltest.Version})
	fmt.Println(q.Done(), err)
	// Output: true <nil>
}

func ExampleNewRecorder() {
	q := exampleQuery("GpmSupported", `{}`, `true`)
	r, err := nvmltest.NewRecorder(q, nvmltest.Provenance{Source: "synthetic example"})
	if err != nil {
		panic(err)
	}
	supported, err := r.GpmSupported()
	fmt.Println(supported, err)
	// Output: true <nil>
}

func ExampleRecorder_Recording() {
	r := exampleRecorder("GpmSupported", `{}`, `true`)
	r.GpmSupported()
	rec, err := r.Recording()
	fmt.Println(rec.Version, len(rec.Calls), rec.Provenance.Measured, err)
	// Output: 1 1 false <nil>
}

func ExampleReplay_Done() {
	q := exampleQuery("GpmSupported", `{}`, `true`)
	fmt.Println(q.Done())
	q.GpmSupported()
	fmt.Println(q.Done())
	// Output:
	// false
	// true
}

func ExampleFailure_Error() {
	f := &nvmltest.Failure{Kind: "other", Message: "query failed"}
	fmt.Println(f.Error())
	// Output: query failed
}

func ExampleFailure_Unwrap() {
	f := &nvmltest.Failure{Kind: "result", Message: "not supported", Code: nvml.ErrNotSupported}
	fmt.Println(errors.Is(f, nvml.ErrNotSupported))
	// Output: true
}

func ExampleReplay_GpmSupported() {
	q := exampleQuery("GpmSupported", `{}`, `true`)
	v, err := q.GpmSupported()
	fmt.Println(v, err)
	// Output: true <nil>
}

func ExampleReplay_ProcessesUtilization() {
	q := exampleQuery("ProcessesUtilization", `{"LastSeen":7}`, `[{"Pid":42,"SmUtil":19}]`)
	v, err := q.ProcessesUtilization(7)
	fmt.Println(v[0].Pid, v[0].SmUtil, err)
	// Output: 42 19 <nil>
}

func ExampleReplay_FabricInfo() {
	q := exampleQuery("FabricInfo", `{}`, `{"CliqueId":11}`)
	v, err := q.FabricInfo()
	fmt.Println(v.CliqueId, err)
	// Output: 11 <nil>
}

func ExampleReplay_FieldValuesFor() {
	q := exampleQuery("FieldValuesFor", `{"IDs":[71]}`, `[{"FieldId":71,"NvmlReturn":4}]`)
	v, err := q.FieldValuesFor(71)
	fmt.Println(v[0].FieldId, int32(v[0].NvmlReturn), err)
	// Output: 71 4 <nil>
}

func ExampleReplay_GpuInstanceList() {
	q := exampleQuery("GpuInstanceList", `{"ProfileID":3}`, `[37]`)
	v, err := q.GpuInstanceList(3)
	fmt.Println(uintptr(v[0]), err)
	// Output: 37 <nil>
}

func ExampleRecorder_GpmSupported() {
	q := exampleRecorder("GpmSupported", `{}`, `true`)
	v, err := q.GpmSupported()
	fmt.Println(v, err)
	// Output: true <nil>
}

func ExampleRecorder_ProcessesUtilization() {
	q := exampleRecorder("ProcessesUtilization", `{"LastSeen":7}`, `[{"Pid":42,"SmUtil":19}]`)
	v, err := q.ProcessesUtilization(7)
	fmt.Println(v[0].Pid, v[0].SmUtil, err)
	// Output: 42 19 <nil>
}

func ExampleRecorder_FabricInfo() {
	q := exampleRecorder("FabricInfo", `{}`, `{"CliqueId":11}`)
	v, err := q.FabricInfo()
	fmt.Println(v.CliqueId, err)
	// Output: 11 <nil>
}

func ExampleRecorder_FieldValuesFor() {
	q := exampleRecorder("FieldValuesFor", `{"IDs":[71]}`, `[{"FieldId":71,"NvmlReturn":4}]`)
	v, err := q.FieldValuesFor(71)
	fmt.Println(v[0].FieldId, int32(v[0].NvmlReturn), err)
	// Output: 71 4 <nil>
}

func ExampleRecorder_GpuInstanceList() {
	q := exampleRecorder("GpuInstanceList", `{"ProfileID":3}`, `[37]`)
	v, err := q.GpuInstanceList(3)
	fmt.Println(uintptr(v[0]), err)
	// Output: 37 <nil>
}

// inspectDevice summarizes telemetry from any Queries implementation.
func inspectDevice(q nvmltest.Queries) (bool, int, nvml.Result, error) {
	supported, err := q.GpmSupported()
	if err != nil && !nvml.IsUnsupported(err) {
		return false, 0, 0, err
	}
	procs, err := q.ProcessesUtilization(0)
	if err != nil && !nvml.IsUnsupported(err) {
		return false, 0, 0, err
	}
	fabric, err := q.FabricInfo()
	if err != nil && !nvml.IsUnsupported(err) {
		return false, 0, 0, err
	}
	return supported, len(procs), fabric.Status, nil
}

func ExampleQueries() {
	rec := nvmltest.Recording{
		Version:    nvmltest.Version,
		Provenance: nvmltest.Provenance{Source: "synthetic example"},
		Calls: []nvmltest.Call{
			{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)},
			{Method: "ProcessesUtilization", Args: json.RawMessage(`{"LastSeen":0}`), Result: json.RawMessage(`[{"Pid":101,"SmUtil":12},{"Pid":102,"SmUtil":45}]`)},
			{Method: "FabricInfo", Args: json.RawMessage(`{}`), Result: json.RawMessage(`{"CliqueId":1,"Status":0}`)},
		},
	}
	device, err := nvmltest.NewReplay(rec)
	if err != nil {
		panic(err)
	}

	supported, activePids, status, err := inspectDevice(device)
	if err != nil {
		panic(err)
	}
	fmt.Printf("gpm=%v procs=%d fabric=%d done=%v\n", supported, activePids, int32(status), device.Done())
	// Output: gpm=true procs=2 fabric=0 done=true
}

func Example_recordedReplay() {
	synthetic := nvmltest.Recording{
		Version:    nvmltest.Version,
		Provenance: nvmltest.Provenance{Source: "synthetic mock"},
		Calls: []nvmltest.Call{
			{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)},
		},
	}
	device, err := nvmltest.NewReplay(synthetic)
	if err != nil {
		panic(err)
	}

	recorder, err := nvmltest.NewRecorder(device, nvmltest.Provenance{Source: "recorded session"})
	if err != nil {
		panic(err)
	}
	supported, err := recorder.GpmSupported()
	if err != nil {
		panic(err)
	}

	rec, err := recorder.Recording()
	if err != nil {
		panic(err)
	}
	var buf bytes.Buffer
	if err := nvmltest.Write(&buf, rec); err != nil {
		panic(err)
	}

	loaded, err := nvmltest.Read(&buf)
	if err != nil {
		panic(err)
	}
	replay, err := nvmltest.NewReplay(loaded)
	if err != nil {
		panic(err)
	}
	replayedSupported, err := replay.GpmSupported()
	fmt.Printf("recorded=%v replayed=%v done=%v\n", supported, replayedSupported, replay.Done())
	// Output: recorded=true replayed=true done=true
}

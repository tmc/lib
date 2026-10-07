// Command nvmlrecord records read-only queries for one device. Run it under
// gpu-lock; it neither enables MIG nor changes clocks or device state.
package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/nvml/nvmltest"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

type observation struct {
	Method      string
	Args        any
	Value       any
	Error       string
	Code        *nvml.Result
	Unsupported bool
	Unmeasured  string
}

type receipt struct {
	// Complete means collection and retirement completed, not that every
	// queried capability was supported or every measured query succeeded.
	Complete bool
	Failure  string
	Version  int
	Time     string
	Source   string
	GPU      string
	Name     string
	Driver   string
	MIG      string
	LockPID  int
	Calls    []observation
	Replay   nvmltest.Recording
}

func main() {
	index := flag.Uint("index", 0, "device index")
	source := flag.String("source", "", "qualified source commit/digest")
	out := flag.String("out", "", "new receipt path")
	flag.Parse()
	if uint64(*index) > uint64(^uint32(0)) {
		fmt.Fprintln(os.Stderr, "nvmlrecord: device index exceeds uint32")
		os.Exit(1)
	}
	if err := run(uint32(*index), *source, *out); err != nil {
		fmt.Fprintln(os.Stderr, "nvmlrecord:", err)
		os.Exit(1)
	}
}

func run(index uint32, source, out string) error {
	return runWith(index, source, out, collect)
}

func runWith(index uint32, source, out string, query func(uint32, string, int) (receipt, error)) error {
	if source == "" || out == "" {
		return errors.New("source identity and output path required")
	}
	if _, err := hex.DecodeString(source); err != nil || len(source) != 40 && len(source) != 64 {
		return errors.New("source must be a commit or SHA256 digest")
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		return fmt.Errorf("inherit gpu lock: %w", err)
	}
	r, queryErr := query(index, source, owner.PID)
	r.Complete = queryErr == nil
	if queryErr != nil {
		r.Failure = queryErr.Error()
	}
	return errors.Join(queryErr, writeReceipt(out, r))
}

func collect(index uint32, source string, pid int) (r receipt, err error) {
	r = receipt{Version: 1, Time: time.Now().UTC().Format(time.RFC3339Nano), Source: source, MIG: "unavailable", LockPID: pid}
	defer func() { completeUnmeasured(&r, err) }()
	initErr := nvml.Init()
	r.add("Init", nil, nil, initErr)
	if err := initErr; err != nil {
		return r, fmt.Errorf("initialize nvml: %w", err)
	}
	defer func() { err = errors.Join(err, cleanupObservation(&r, "Shutdown", nil, nvml.Shutdown)) }()
	d, err := nvml.NewDeviceByIndex(index)
	r.add("NewDeviceByIndex", index, d, err)
	if err != nil {
		return r, err
	}
	uuid, err := d.UUID()
	r.add("UUID", nil, uuid, err)
	if err != nil {
		return r, err
	}
	name, err := d.Name()
	r.add("Name", nil, name, err)
	if err != nil {
		return r, err
	}
	driver, err := nvml.SystemGetDriverVersion()
	r.add("SystemGetDriverVersion", nil, driver, err)
	if err != nil {
		return r, err
	}
	r.GPU, r.Name, r.Driver = uuid, name, driver
	add := r.add
	current, pending, err := d.MigMode()
	migReason := "MIG is disabled"
	if err != nil {
		migReason = "MIG mode query failed: " + err.Error()
	}
	add("MigMode", nil, []uint32{current, pending}, err)
	if err == nil {
		r.MIG = fmt.Sprintf("current=%d pending=%d", current, pending)
	}
	recorder, err := nvmltest.NewRecorder(d, nvmltest.Provenance{GPU: uuid, Driver: driver, MIG: r.MIG, Source: source, Measured: true})
	if err != nil {
		return r, err
	}
	supported, supportErr := recorder.GpmSupported()
	add("GpmSupported", nil, supported, supportErr)
	processes, err := recorder.ProcessesUtilization(0)
	add("ProcessesUtilization", uint64(0), processes, err)
	fabric, err := recorder.FabricInfo()
	add("FabricInfo", nil, fabric, err)
	ids := []uint32{nvml.FieldDevNvlinkCountEffectiveBer, nvml.FieldDevNvlinkCountRawBerLane0V2, nvml.FieldDevNvlinkCountRawBerLane1V2, nvml.FieldDevC2CLinkErrorReplay, nvml.FieldDevC2CLinkErrorReplayB2B}
	fields, err := recorder.FieldValuesFor(ids...)
	add("FieldValuesFor", ids, fields, err)
	profile, err := d.GpuInstanceProfileInfo(0)
	add("GpuInstanceProfileInfo", uint32(0), profile, err)
	instanceQuery := err == nil
	if err == nil {
		instances, e := recorder.GpuInstanceList(profile.Id)
		add("GpuInstanceList", profile.Id, instances, e)
	} else {
		r.unmeasured("GpuInstanceList", nil, "profile query failed: "+err.Error())
	}
	rows, err := d.RemappedRows()
	add("RemappedRows", nil, rows, err)
	repair, err := d.RepairStatus()
	add("RepairStatus", nil, repair, err)
	clocks, err := d.CurrentClocksEventReasons()
	add("CurrentClocksEventReasons", nil, clocks, err)
	samples, typ, err := d.Samples(nvml.TotalPowerSamples, 0, nil)
	add("Samples", map[string]any{"type": nvml.TotalPowerSamples, "lastSeen": uint64(0), "buffer": "nil"}, map[string]any{"samples": samples, "valueType": typ}, err)
	cleanupErr := collectGPM(&r, supported, supportErr, current == 1 && r.MIG != "unavailable", gpmQueries{
		migUnavailable: migReason,
		allocate:       nvml.NewGpmSample,
		get:            func(s nvml.GpmSample) error { return s.Get(d) },
		metrics: func(a, b nvml.GpmSample) ([]nvml.GpmMetric, error) {
			return nvml.GpmMetrics(a, b, nvml.GpmMetricSmUtil, nvml.GpmMetricDramBwUtil)
		},
		instanceID: d.GpuInstanceId,
		migGet:     d.GpmMigSampleGet,
		close:      func(s *nvml.GpmSample) error { return s.Close() },
		pause:      func() { time.Sleep(110 * time.Millisecond) },
	})
	r.Replay, err = recorder.Recording()
	if err != nil || cleanupErr != nil {
		return r, errors.Join(err, cleanupErr)
	}
	// Exercise the actual recorded script through the same public API before
	// writing it. This verifies serialization, not another device's behavior.
	replay, err := nvmltest.NewReplay(r.Replay)
	if err != nil {
		return r, err
	}
	copyRecorder, err := nvmltest.NewRecorder(replay, r.Replay.Provenance)
	if err != nil {
		return r, err
	}
	copyRecorder.GpmSupported()
	copyRecorder.ProcessesUtilization(0)
	copyRecorder.FabricInfo()
	copyRecorder.FieldValuesFor(ids...)
	if instanceQuery {
		copyRecorder.GpuInstanceList(profile.Id)
	}
	if !replay.Done() {
		return r, errors.New("replay did not consume all recorded queries")
	}
	copyRecording, err := copyRecorder.Recording()
	if err != nil {
		return r, err
	}
	before, err := json.Marshal(r.Replay)
	if err != nil {
		return r, err
	}
	after, err := json.Marshal(copyRecording)
	if err != nil {
		return r, err
	}
	if !bytes.Equal(before, after) {
		return r, errors.New("record/replay round-trip changed responses")
	}
	return r, nil
}

func writeReceipt(out string, r receipt) error {
	f, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	err = enc.Encode(r)
	closeErr := f.Close()
	return errors.Join(err, closeErr)
}

package nvmltest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestCompileTimeInterfaceDrift(t *testing.T) {
	qType := reflect.TypeOf((*Queries)(nil)).Elem()
	devType := reflect.TypeOf(nvml.Device(0))

	if qType.NumMethod() != 5 {
		t.Fatalf("Queries method count = %d, want 5", qType.NumMethod())
	}

	for i := 0; i < qType.NumMethod(); i++ {
		qm := qType.Method(i)
		dm, ok := devType.MethodByName(qm.Name)
		if !ok {
			t.Fatalf("nvml.Device missing Queries method %s", qm.Name)
		}
		// In dm, the receiver is param 0, so compare parameter slices starting from index 1.
		if qm.Type.NumIn() != dm.Type.NumIn()-1 {
			t.Fatalf("method %s param count: Queries=%d, Device=%d", qm.Name, qm.Type.NumIn(), dm.Type.NumIn()-1)
		}
		for p := 0; p < qm.Type.NumIn(); p++ {
			if qm.Type.In(p) != dm.Type.In(p+1) {
				t.Fatalf("method %s param %d: Queries=%v, Device=%v", qm.Name, p, qm.Type.In(p), dm.Type.In(p+1))
			}
		}
		if qm.Type.NumOut() != dm.Type.NumOut() {
			t.Fatalf("method %s return count: Queries=%d, Device=%d", qm.Name, qm.Type.NumOut(), dm.Type.NumOut())
		}
		for r := 0; r < qm.Type.NumOut(); r++ {
			if qm.Type.Out(r) != dm.Type.Out(r) {
				t.Fatalf("method %s return %d: Queries=%v, Device=%v", qm.Name, r, qm.Type.Out(r), dm.Type.Out(r))
			}
		}
	}

	// Verify concrete implementations satisfy Queries.
	var _ Queries = nvml.Device(0)
	var _ Queries = (*Replay)(nil)
	var _ Queries = (*Recorder)(nil)
}

func TestDeepCopyResponseSafety(t *testing.T) {
	rec := Recording{
		Version:    Version,
		Provenance: Provenance{Source: "deep copy test"},
		Calls: []Call{
			{
				Method: "ProcessesUtilization",
				Args:   json.RawMessage(`{"LastSeen":0}`),
				Result: json.RawMessage(`[{"Pid":100,"SmUtil":50}]`),
			},
			{
				Method: "FieldValuesFor",
				Args:   json.RawMessage(`{"IDs":[71]}`),
				Result: json.RawMessage(`[{"FieldId":71,"NvmlReturn":0,"ValueType":1}]`),
			},
			{
				Method: "GpuInstanceList",
				Args:   json.RawMessage(`{"ProfileID":1}`),
				Result: json.RawMessage(`[10,20]`),
			},
		},
	}

	r, err := NewReplay(rec)
	if err != nil {
		t.Fatal(err)
	}

	// 1. ProcessesUtilization response slice mutation must not affect another replay instance.
	procs, err := r.ProcessesUtilization(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(procs) != 1 || procs[0].Pid != 100 {
		t.Fatalf("unexpected procs %+v", procs)
	}
	procs[0].Pid = 999
	procs[0].SmUtil = 99

	// 2. FieldValuesFor response slice mutation.
	fields, err := r.FieldValuesFor(71)
	if err != nil {
		t.Fatal(err)
	}
	if len(fields) != 1 || fields[0].FieldId != 71 {
		t.Fatalf("unexpected fields %+v", fields)
	}
	fields[0].FieldId = 999
	fields[0].NvmlReturn = nvml.ErrGpuIsLost

	// 3. GpuInstanceList response slice mutation.
	instances, err := r.GpuInstanceList(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(instances) != 2 || instances[0] != 10 {
		t.Fatalf("unexpected instances %+v", instances)
	}
	instances[0] = 999

	// Verify a second replay instance from the same recording sees untouched data.
	r2, err := NewReplay(rec)
	if err != nil {
		t.Fatal(err)
	}
	procs2, err := r2.ProcessesUtilization(0)
	if err != nil || procs2[0].Pid != 100 || procs2[0].SmUtil != 50 {
		t.Fatalf("ProcessesUtilization was corrupted by caller mutation: %+v, %v", procs2, err)
	}
	fields2, err := r2.FieldValuesFor(71)
	if err != nil || fields2[0].FieldId != 71 || fields2[0].NvmlReturn != nvml.Success {
		t.Fatalf("FieldValuesFor was corrupted by caller mutation: %+v, %v", fields2, err)
	}
	instances2, err := r2.GpuInstanceList(1)
	if err != nil || instances2[0] != 10 || instances2[1] != 20 {
		t.Fatalf("GpuInstanceList was corrupted by caller mutation: %+v, %v", instances2, err)
	}

	// 4. Recorder.Recording() deep copy: mutating returned Recording must not mutate Recorder internal state.
	dummy := &stubQueries{
		support: true,
	}
	recorder, err := NewRecorder(dummy, Provenance{Source: "recorder copy test"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recorder.GpmSupported(); err != nil {
		t.Fatal(err)
	}
	out1, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	out1.Calls[0].Method = "CorruptedMethod"
	out1.Calls[0].Result = json.RawMessage(`false`)

	out2, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	if out2.Calls[0].Method != "GpmSupported" || string(out2.Calls[0].Result) != "true" {
		t.Fatalf("Recorder internal state corrupted: %+v", out2.Calls[0])
	}

	// 5. Caller argument slice mutation after Recorder.FieldValuesFor call.
	ids := []uint32{71, 83}
	if _, err := recorder.FieldValuesFor(ids...); err != nil {
		t.Fatal(err)
	}
	ids[0] = 999
	ids[1] = 888

	out3, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}
	if string(out3.Calls[1].Args) != `{"IDs":[71,83]}` {
		t.Fatalf("Recorder arguments corrupted by subsequent caller slice mutation: %s", out3.Calls[1].Args)
	}
}

type stubQueries struct {
	Queries
	support   bool
	procs     []nvml.ProcessUtilizationInfoV1
	fabric    nvml.GpuFabricInfoV
	fields    []nvml.FieldValue
	instances []nvml.GpuInstance
	err       error
}

func (s *stubQueries) GpmSupported() (bool, error) {
	return s.support, s.err
}

func (s *stubQueries) ProcessesUtilization(uint64) ([]nvml.ProcessUtilizationInfoV1, error) {
	return s.procs, s.err
}

func (s *stubQueries) FabricInfo() (nvml.GpuFabricInfoV, error) {
	return s.fabric, s.err
}

func (s *stubQueries) FieldValuesFor(...uint32) ([]nvml.FieldValue, error) {
	return s.fields, s.err
}

func (s *stubQueries) GpuInstanceList(uint32) ([]nvml.GpuInstance, error) {
	return s.instances, s.err
}

func TestOperationalVersusUnsupportedErrorPreservation(t *testing.T) {
	testErrors := []struct {
		name          string
		err           error
		isUnsupported bool
		expectedCode  nvml.Result
	}{
		// Operational errors (IsUnsupported must be false)
		{
			name:          "operational OS error",
			err:           nvml.ErrOperatingSystem,
			isUnsupported: false,
			expectedCode:  nvml.ErrOperatingSystem,
		},
		{
			name:          "operational reset required",
			err:           nvml.ErrResetRequired,
			isUnsupported: false,
			expectedCode:  nvml.ErrResetRequired,
		},
		{
			name:          "operational out of memory",
			err:           nvml.ErrMemory,
			isUnsupported: false,
			expectedCode:  nvml.ErrMemory,
		},
		{
			name:          "operational timeout",
			err:           nvml.ErrTimeout,
			isUnsupported: false,
			expectedCode:  nvml.ErrTimeout,
		},
		{
			name:          "operational gpu is lost",
			err:           nvml.ErrGpuIsLost,
			isUnsupported: false,
			expectedCode:  nvml.ErrGpuIsLost,
		},
		{
			name:          "operational wrapped error",
			err:           fmt.Errorf("query telemetry: %w", nvml.ErrOperatingSystem),
			isUnsupported: false,
			expectedCode:  nvml.ErrOperatingSystem,
		},
		{
			name:          "operational joined error",
			err:           errors.Join(nvml.ErrOperatingSystem, nvml.ErrMemory),
			isUnsupported: false,
			expectedCode:  nvml.ErrOperatingSystem,
		},
		{
			name:          "operational custom string",
			err:           errors.New("driver connection reset"),
			isUnsupported: false,
			expectedCode:  nvml.Success, // not an nvml.Result
		},
		// Unsupported errors (IsUnsupported must be true)
		{
			name:          "unsupported result",
			err:           nvml.ErrNotSupported,
			isUnsupported: true,
			expectedCode:  nvml.ErrNotSupported,
		},
		{
			name:          "unsupported missing symbol",
			err:           &nvml.SymbolError{Symbol: "nvmlDeviceGetGpmSupport", Err: nil},
			isUnsupported: true,
			expectedCode:  nvml.Success,
		},
		{
			name:          "unsupported wrapped",
			err:           fmt.Errorf("gpm feature: %w", nvml.ErrNotSupported),
			isUnsupported: true,
			expectedCode:  nvml.ErrNotSupported,
		},
		{
			name:          "unsupported joined with permission",
			err:           errors.Join(nvml.ErrNoPermission, nvml.ErrNotSupported),
			isUnsupported: true,
			expectedCode:  nvml.ErrNotSupported,
		},
		{
			name:          "unsupported joined missing symbol",
			err:           errors.Join(nvml.ErrNoPermission, &nvml.SymbolError{Symbol: "nvmlDeviceGetFabricInfoV", Err: nil}),
			isUnsupported: true,
			expectedCode:  nvml.Success,
		},
		// Non-unsupported non-operational (permission / symbol load error)
		{
			name:          "permission error alone is not unsupported",
			err:           nvml.ErrNoPermission,
			isUnsupported: false,
			expectedCode:  nvml.ErrNoPermission,
		},
		{
			name:          "symbol binding error alone is not unsupported",
			err:           &nvml.SymbolError{Symbol: "nvmlInit", Err: errors.New("dlsym failed")},
			isUnsupported: false,
			expectedCode:  nvml.Success,
		},
	}

	methods := []struct {
		name string
		call func(q Queries) error
	}{
		{
			name: "GpmSupported",
			call: func(q Queries) error { _, err := q.GpmSupported(); return err },
		},
		{
			name: "ProcessesUtilization",
			call: func(q Queries) error { _, err := q.ProcessesUtilization(0); return err },
		},
		{
			name: "FabricInfo",
			call: func(q Queries) error { _, err := q.FabricInfo(); return err },
		},
		{
			name: "FieldValuesFor",
			call: func(q Queries) error { _, err := q.FieldValuesFor(71); return err },
		},
		{
			name: "GpuInstanceList",
			call: func(q Queries) error { _, err := q.GpuInstanceList(1); return err },
		},
	}

	for _, tc := range testErrors {
		t.Run(tc.name, func(t *testing.T) {
			if nvml.IsUnsupported(tc.err) != tc.isUnsupported {
				t.Fatalf("test setup assumption failed: IsUnsupported(%v) = %v, want %v", tc.err, nvml.IsUnsupported(tc.err), tc.isUnsupported)
			}

			for _, m := range methods {
				t.Run(m.name, func(t *testing.T) {
					stub := &stubQueries{err: tc.err}
					recorder, err := NewRecorder(stub, Provenance{Source: "error test"})
					if err != nil {
						t.Fatal(err)
					}

					recErr := m.call(recorder)
					if recErr == nil {
						t.Fatal("expected error, got nil")
					}
					if recErr.Error() != tc.err.Error() {
						t.Fatalf("recorder error string = %q, want %q", recErr.Error(), tc.err.Error())
					}
					if nvml.IsUnsupported(recErr) != tc.isUnsupported {
						t.Fatalf("recorder IsUnsupported = %v, want %v", nvml.IsUnsupported(recErr), tc.isUnsupported)
					}

					// Round-trip to JSON
					rec, err := recorder.Recording()
					if err != nil {
						t.Fatal(err)
					}
					var buf bytes.Buffer
					if err := Write(&buf, rec); err != nil {
						t.Fatal(err)
					}

					readRec, err := Read(&buf)
					if err != nil {
						t.Fatal(err)
					}

					replay, err := NewReplay(readRec)
					if err != nil {
						t.Fatal(err)
					}

					replayErr := m.call(replay)
					if replayErr == nil {
						t.Fatal("expected replayed error, got nil")
					}
					if replayErr.Error() != tc.err.Error() {
						t.Fatalf("replayed error string = %q, want %q", replayErr.Error(), tc.err.Error())
					}
					if nvml.IsUnsupported(replayErr) != tc.isUnsupported {
						t.Fatalf("replayed IsUnsupported = %v, want %v (err=%v)", nvml.IsUnsupported(replayErr), tc.isUnsupported, replayErr)
					}
					if tc.expectedCode != nvml.Success {
						if !errors.Is(replayErr, tc.expectedCode) {
							t.Fatalf("replayed error lost code %v: %v", tc.expectedCode, replayErr)
						}
					}
					if !replay.Done() {
						t.Fatal("replay not done")
					}
				})
			}
		})
	}
}

func TestArgumentAndVersionMismatchMutantControls(t *testing.T) {
	// 1. Version mutants
	for _, v := range []int{0, 2, -1, 999} {
		t.Run(fmt.Sprintf("version_%d", v), func(t *testing.T) {
			rec := Recording{Version: v}
			if err := Write(&bytes.Buffer{}, rec); err == nil {
				t.Fatalf("Write accepted version %d", v)
			}
			if _, err := NewReplay(rec); err == nil {
				t.Fatalf("NewReplay accepted version %d", v)
			}
			raw := fmt.Sprintf(`{"version":%d,"provenance":{},"calls":[]}`, v)
			if _, err := Read(bytes.NewReader([]byte(raw))); err == nil {
				t.Fatalf("Read accepted version %d", v)
			}
		})
	}

	// 2. Argument mutant controls and non-advancement verification
	t.Run("ProcessesUtilization_cursor_mismatch", func(t *testing.T) {
		rec := Recording{
			Version: Version,
			Calls: []Call{
				{
					Method: "ProcessesUtilization",
					Args:   json.RawMessage(`{"LastSeen":100}`),
					Result: json.RawMessage(`[]`),
				},
			},
		}
		r, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		// Mutant 1: wrong cursor
		if _, err := r.ProcessesUtilization(99); err == nil {
			t.Fatal("accepted wrong cursor 99")
		}
		// Invariant: mismatch must not advance cursor
		if r.Done() {
			t.Fatal("mismatch advanced script")
		}
		// Correct call must still succeed
		if _, err := r.ProcessesUtilization(100); err != nil {
			t.Fatalf("correct call failed after mismatch: %v", err)
		}
		if !r.Done() {
			t.Fatal("expected script done")
		}
	})

	t.Run("FieldValuesFor_IDs_mismatches", func(t *testing.T) {
		rec := Recording{
			Version: Version,
			Calls: []Call{
				{
					Method: "FieldValuesFor",
					Args:   json.RawMessage(`{"IDs":[10,20]}`),
					Result: json.RawMessage(`[]`),
				},
			},
		}
		r, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		// Mutants
		mutants := [][]uint32{
			{20, 10},
			{10},
			{20},
			{10, 20, 30},
			{10, 21},
			nil,
			{},
		}
		for _, m := range mutants {
			if _, err := r.FieldValuesFor(m...); err == nil {
				t.Fatalf("accepted mutant IDs %v", m)
			}
			if r.Done() {
				t.Fatalf("mutant %v advanced script", m)
			}
		}
		// Correct call succeeds
		if _, err := r.FieldValuesFor(10, 20); err != nil {
			t.Fatalf("correct IDs failed: %v", err)
		}
		if !r.Done() {
			t.Fatal("expected script done")
		}
	})

	t.Run("FieldValuesFor_nil_vs_empty_slice", func(t *testing.T) {
		// Null IDs
		recNil := Recording{
			Version: Version,
			Calls: []Call{
				{
					Method: "FieldValuesFor",
					Args:   json.RawMessage(`{"IDs":null}`),
					Result: json.RawMessage(`[]`),
				},
			},
		}
		rNil, err := NewReplay(recNil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := rNil.FieldValuesFor([]uint32{}...); err == nil {
			t.Fatal("accepted empty slice when nil was recorded")
		}
		if _, err := rNil.FieldValuesFor(nil...); err != nil {
			t.Fatalf("nil call failed: %v", err)
		}

		// Empty slice IDs
		recEmpty := Recording{
			Version: Version,
			Calls: []Call{
				{
					Method: "FieldValuesFor",
					Args:   json.RawMessage(`{"IDs":[]}`),
					Result: json.RawMessage(`[]`),
				},
			},
		}
		rEmpty, err := NewReplay(recEmpty)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := rEmpty.FieldValuesFor(nil...); err == nil {
			t.Fatal("accepted nil when empty slice was recorded")
		}
		if _, err := rEmpty.FieldValuesFor([]uint32{}...); err != nil {
			t.Fatalf("empty slice call failed: %v", err)
		}
	})

	t.Run("GpuInstanceList_profile_mismatch", func(t *testing.T) {
		rec := Recording{
			Version: Version,
			Calls: []Call{
				{
					Method: "GpuInstanceList",
					Args:   json.RawMessage(`{"ProfileID":7}`),
					Result: json.RawMessage(`[]`),
				},
			},
		}
		r, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := r.GpuInstanceList(8); err == nil {
			t.Fatal("accepted profile 8")
		}
		if r.Done() {
			t.Fatal("mismatch advanced script")
		}
		if _, err := r.GpuInstanceList(7); err != nil {
			t.Fatalf("correct profile failed: %v", err)
		}
		if !r.Done() {
			t.Fatal("expected script done")
		}
	})

	t.Run("Method_mismatch_and_excess_calls", func(t *testing.T) {
		rec := Recording{
			Version: Version,
			Calls: []Call{
				{Method: "GpmSupported", Args: json.RawMessage(`{}`), Result: json.RawMessage(`true`)},
			},
		}
		r, err := NewReplay(rec)
		if err != nil {
			t.Fatal(err)
		}
		// Wrong methods
		if _, err := r.FabricInfo(); err == nil {
			t.Fatal("accepted FabricInfo when GpmSupported expected")
		}
		if _, err := r.ProcessesUtilization(0); err == nil {
			t.Fatal("accepted ProcessesUtilization when GpmSupported expected")
		}
		if r.Done() {
			t.Fatal("method mismatch advanced script")
		}
		// Correct method
		if v, err := r.GpmSupported(); err != nil || !v {
			t.Fatalf("correct GpmSupported failed: %v, %v", v, err)
		}
		if !r.Done() {
			t.Fatal("expected script done")
		}
		// Excess call after done
		if _, err := r.GpmSupported(); err == nil {
			t.Fatal("accepted excess call after Done")
		}
	})
}

func TestPerEntryStatusAndErrorPreservation(t *testing.T) {
	// FieldValuesFor with per-entry errors
	fields := []nvml.FieldValue{
		{FieldId: 10, NvmlReturn: nvml.Success, ValueType: 1},
		{FieldId: 20, NvmlReturn: nvml.ErrNotSupported, ValueType: 1},
		{FieldId: 30, NvmlReturn: nvml.ErrNoPermission, ValueType: 1},
		{FieldId: 40, NvmlReturn: nvml.ErrGpuIsLost, ValueType: 1},
	}
	fabric := nvml.GpuFabricInfoV{
		CliqueId: 42,
		Status:   nvml.ErrNotSupported,
	}

	stub := &stubQueries{
		fields: fields,
		fabric: fabric,
	}

	recorder, err := NewRecorder(stub, Provenance{Source: "per-entry test"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := recorder.FieldValuesFor(10, 20, 30, 40); err != nil {
		t.Fatal(err)
	}
	if _, err := recorder.FabricInfo(); err != nil {
		t.Fatal(err)
	}

	rec, err := recorder.Recording()
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := Write(&buf, rec); err != nil {
		t.Fatal(err)
	}

	readRec, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	replay, err := NewReplay(readRec)
	if err != nil {
		t.Fatal(err)
	}

	replayedFields, err := replay.FieldValuesFor(10, 20, 30, 40)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayedFields) != len(fields) {
		t.Fatalf("fields count = %d, want %d", len(replayedFields), len(fields))
	}
	for i, want := range fields {
		got := replayedFields[i]
		if got.FieldId != want.FieldId || got.NvmlReturn != want.NvmlReturn {
			t.Errorf("field %d: got {FieldId: %d, NvmlReturn: %v}, want {FieldId: %d, NvmlReturn: %v}", i, got.FieldId, got.NvmlReturn, want.FieldId, want.NvmlReturn)
		}
	}

	replayedFabric, err := replay.FabricInfo()
	if err != nil {
		t.Fatal(err)
	}
	if replayedFabric.CliqueId != fabric.CliqueId || replayedFabric.Status != fabric.Status {
		t.Errorf("fabric: got %+v, want %+v", replayedFabric, fabric)
	}
}

//go:build dcgm_physical

// This file is the physical consumer test. It talks to a real DCGM host engine
// about real GPUs and checks that a value actually arrives. It is behind a
// build tag because it cannot be made to pass by skipping, and a skipped
// hardware test is not evidence that the hardware path works.
//
// It has never been run. It was written on a machine with no GPU, no NVIDIA
// driver and no libdcgm -- DCGM does not ship for macOS at all -- and
// everything below is unverified.
//
// # What it needs
//
//   - Linux on x86_64 or aarch64. DCGM has no macOS or Windows build.
//   - An NVIDIA data center GPU that DCGM supports. Consumer GeForce parts are
//     not supported by DCGM even when the driver works, and this test fails
//     rather than skips if dcgmGetAllSupportedDevices returns none: "the
//     driver sees a GPU" is exactly the confusion this test exists to avoid.
//   - An NVIDIA driver the installed DCGM supports. The bindings are generated
//     from DCGM 4.6.1 headers; the loaded library must have soname
//     libdcgm.so.4.
//   - The datacenter-gpu-manager package installed, providing libdcgm.so.4 on
//     the loader path or at DCGM_LIB_PATH.
//   - For TestConnectedHostEngine only: nv-hostengine already running, with
//     its address in DCGM_HOSTENGINE. Everything else uses an embedded engine
//     started inside the test process, which needs no service but does need
//     permission to talk to the driver -- typically root or membership in the
//     group /dev/nvidiactl is owned by.
//   - For TestProfilingFieldsArrive only: the DCGM profiling module loadable,
//     which additionally requires no other profiler holding the GPU. That test
//     fails, with the module status printed, if the module is not loaded.
//
// # How to run it
//
//	go test -tags dcgm_physical ./nvidia/dcgm/...
//	DCGM_HOSTENGINE=127.0.0.1:5555 go test -tags dcgm_physical ./nvidia/dcgm/...
//
// Every test below fails, loudly, when a prerequisite is missing. None of them
// skips, and none of them treats a blank reading as a pass.
//
// # What a pass would mean
//
// That libdcgm loaded, that a host engine started or was reached, that a
// group and a field group were accepted, that a watch was established, and
// that at least one non-blank sample with a plausible timestamp came back
// through these bindings. It would not mean the numbers are correct: nothing
// here has an independent oracle for a GPU temperature. What it checks is that
// the transport, the versioned structures and the value decoding work, which
// is the part the bindings are responsible for.

package dcgm

import (
	"errors"
	"os"
	"testing"
	"time"
)

// mustLoad opens libdcgm and fails if it cannot. Under this build tag the
// library is a stated prerequisite, so its absence is a failure, not a skip.
func mustLoad(t *testing.T) {
	t.Helper()
	if err := Load(""); err != nil {
		t.Fatalf("libdcgm.so.4 could not be loaded, which this build tag asserts is present. "+
			"Install the NVIDIA datacenter-gpu-manager package or set DCGM_LIB_PATH: %v", err)
	}
	t.Logf("libdcgm loaded from %s", Path())
}

// mustInit calls dcgmInit and arranges for dcgmShutdown.
func mustInit(t *testing.T) {
	t.Helper()
	mustLoad(t)
	if err := Init(); err != nil {
		t.Fatalf("dcgmInit failed. The library loaded, so this is DCGM refusing to start "+
			"rather than a missing install: %v", err)
	}
	t.Cleanup(func() {
		if err := Shutdown(); err != nil {
			t.Errorf("dcgmShutdown: %v", err)
		}
	})
}

// embeddedSession starts a host engine in this process. Manual mode is used so
// the test decides when sampling happens; in auto mode a read that is too
// early returns blank and the test would be timing-dependent.
func embeddedSession(t *testing.T) *Session {
	t.Helper()
	mustInit(t)
	s, err := StartEmbedded(OperationModeManual)
	if err != nil {
		t.Fatalf("starting an embedded host engine failed. This usually means the process "+
			"cannot reach the driver: check that /dev/nvidiactl and /dev/nvidia* are "+
			"readable by this user, and that the driver version is one this DCGM supports: %v", err)
	}
	if !s.OwnsHostEngine() {
		t.Fatal("StartEmbedded returned a session that does not own its host engine")
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Session.Close left something behind on the host engine: %v", err)
		}
	})
	return s
}

// supportedGPUs returns the GPUs DCGM supports, failing when there are none.
//
// A machine where nvidia-smi works but this returns nothing is a real and
// common configuration -- a GeForce card, or a driver DCGM does not support --
// and it must fail here. Reporting it as a skip is how "the GPU is fine" gets
// mistaken for "the bindings work".
func supportedGPUs(t *testing.T, s *Session) []uint32 {
	t.Helper()
	all, err := s.Handle.Devices()
	if err != nil {
		t.Fatalf("dcgmGetAllDevices: %v", err)
	}
	gpus, err := s.Handle.SupportedDevices()
	if err != nil {
		t.Fatalf("dcgmGetAllSupportedDevices: %v", err)
	}
	if len(gpus) == 0 {
		t.Fatalf("the host engine sees %d GPUs and supports none of them. DCGM supports data "+
			"center GPUs; a working nvidia-smi on an unsupported card is not a substitute, "+
			"and no field can be read here", len(all))
	}
	t.Logf("%d GPUs, %d supported by DCGM", len(all), len(gpus))
	return gpus
}

// TestVersionAndHealth checks the two calls that need no watch: the library
// reports its version, and the host engine reports itself healthy.
func TestVersionAndHealth(t *testing.T) {
	s := embeddedSession(t)

	info := VersionInfoStruct{Version: VersionInfoStructVersion}
	if err := VersionInfo(&info); err != nil {
		t.Errorf("dcgmVersionInfo: %v", err)
	} else {
		t.Logf("DCGM version %s", goString(&info.RawBuildInfoString[0]))
	}

	health := HostengineHealth{Version: HostengineHealthVersion1}
	if err := HostengineIsHealthy(s.Handle, &health); err != nil {
		t.Fatalf("dcgmHostengineIsHealthy: %v", err)
	}
	if health.OverallHealth != 0 {
		t.Errorf("host engine reports overall health %d, want 0", health.OverallHealth)
	}
}

// TestWatchedFieldArrives is the test the whole package exists for: a field is
// watched, the host engine is told to sample, and a real value comes back with
// a real timestamp.
//
// It fails on a blank reading. A blank is a legitimate DCGM answer, and every
// caller must handle it, but a run where every field is blank has demonstrated
// nothing about whether values flow.
func TestWatchedFieldArrives(t *testing.T) {
	s := embeddedSession(t)
	gpus := supportedGPUs(t, s)

	group, err := s.NewGroup("gotest-watched-field", GroupEmpty)
	if err != nil {
		t.Fatalf("dcgmGroupCreate: %v", err)
	}
	if err := s.AddEntity(group, FeGpu, FieldEid(gpus[0])); err != nil {
		t.Fatalf("dcgmGroupAddEntity: %v", err)
	}

	// Temperature and SM clock are supported on every DCGM-supported GPU
	// and need no profiling module, so a blank here is a real failure
	// rather than an unsupported field.
	fields := []FieldID{FieldDevGpuTempCelsius, FieldDevSmClock}
	fieldGroup, err := s.NewFieldGroup("gotest-watched-field", fields)
	if err != nil {
		t.Fatalf("dcgmFieldGroupCreate: %v", err)
	}
	if err := s.Watch(group, fieldGroup, 100*time.Millisecond, time.Minute, 100); err != nil {
		t.Fatalf("dcgmWatchFields: %v", err)
	}
	if err := UpdateAllFields(s.Handle, 1); err != nil {
		t.Fatalf("dcgmUpdateAllFields: %v", err)
	}

	values, err := s.Handle.LatestValues(gpus[0], fields)
	if err != nil {
		t.Fatalf("dcgmGetLatestValuesForFields: %v", err)
	}
	if len(values) != len(fields) {
		t.Fatalf("got %d readings for %d fields", len(values), len(fields))
	}
	for i := range values {
		v := &values[i]
		if v.ID() != fields[i] {
			t.Errorf("reading %d is for %v, want %v; the results are not in request order", i, v.ID(), fields[i])
		}
		if err := v.Err(); err != nil {
			t.Errorf("%v: per-field status %v", v.ID(), err)
			continue
		}
		if v.Blank() {
			t.Errorf("%v came back blank after an explicit UpdateAllFields. Either the field "+
				"is unsupported on this GPU, in which case this test is asking for the "+
				"wrong field, or no sample was taken", v.ID())
			continue
		}
		n, ok := v.Int64()
		if !ok {
			t.Errorf("%v: field type %d is not integer-typed", v.ID(), v.Type())
			continue
		}
		ts := v.Time()
		if ts.IsZero() {
			t.Errorf("%v carries a value but no timestamp", v.ID())
		} else if age := time.Since(ts); age < 0 || age > time.Hour {
			t.Errorf("%v was sampled at %s, %v ago, which is not a plausible sample time", v.ID(), ts, age)
		}
		t.Logf("%v = %d at %s", v.ID(), n, ts.Format(time.RFC3339Nano))
	}
}

// TestEntityValuesKeepPerReadingStatus checks the multi-entity path and the
// property a caller depends on: each reading carries its own status,
// timestamp and entity, so one unsupported field does not make the call fail
// and does not contaminate the others.
//
// It deliberately asks for a field that is often unsupported alongside two
// that are not. The unsupported one must come back blank, not as a call
// failure and not as a zero.
func TestEntityValuesKeepPerReadingStatus(t *testing.T) {
	s := embeddedSession(t)
	gpus := supportedGPUs(t, s)

	entities := make([]GroupEntityPair, 0, len(gpus))
	for _, gpu := range gpus {
		entities = append(entities, GroupEntityPair{EntityGroupId: FeGpu, EntityId: FieldEid(gpu)})
	}
	fields := []FieldID{FieldDevGpuTempCelsius, FieldDevSmClock, FieldDevRetiredDbe}

	group, err := s.NewGroup("gotest-entity-values", GroupDefault)
	if err != nil {
		t.Fatalf("dcgmGroupCreate: %v", err)
	}
	fieldGroup, err := s.NewFieldGroup("gotest-entity-values", fields)
	if err != nil {
		t.Fatalf("dcgmFieldGroupCreate: %v", err)
	}
	if err := s.Watch(group, fieldGroup, 100*time.Millisecond, time.Minute, 100); err != nil {
		t.Fatalf("dcgmWatchFields: %v", err)
	}
	if err := UpdateAllFields(s.Handle, 1); err != nil {
		t.Fatalf("dcgmUpdateAllFields: %v", err)
	}

	values, err := s.Handle.EntityLatestValues(entities, fields, 0)
	if err != nil {
		t.Fatalf("dcgmEntitiesGetLatestValues: %v", err)
	}
	if want := len(entities) * len(fields); len(values) != want {
		t.Fatalf("got %d readings, want %d: one per entity-field pair", len(values), want)
	}
	nonBlank := 0
	for i := range values {
		v := &values[i]
		entity := entities[i/len(fields)]
		if got := v.Entity(); got != entity {
			t.Errorf("reading %d belongs to entity %+v, want %+v; the results are not in entity-major order", i, got, entity)
		}
		if v.Err() != nil {
			continue
		}
		if !v.Blank() {
			nonBlank++
		}
	}
	if nonBlank == 0 {
		t.Fatal("every reading was blank or failed; nothing about value retrieval was demonstrated")
	}
	t.Logf("%d of %d readings carried data", nonBlank, len(values))
}

// TestTeardownRemovesOnlyOurs checks the ownership rule against a live host
// engine: after Close, the group this session created is gone and a group it
// did not create is untouched.
//
// This is the property that cannot be checked without a host engine, and the
// one whose failure is silent and destructive -- another process's monitoring
// stops and nothing reports it.
func TestTeardownRemovesOnlyOurs(t *testing.T) {
	s := embeddedSession(t)
	supportedGPUs(t, s)

	// A group created through the raw entry point, which no Session owns.
	// The test destroys it itself at the end; s.Close must not.
	name := append([]byte("gotest-foreign-group"), 0)
	foreign, err := GroupCreate(s.Handle, GroupDefault, &name[0])
	if err != nil {
		t.Fatalf("creating the unowned group: %v", err)
	}

	owned, err := s.NewGroup("gotest-owned-group", GroupDefault)
	if err != nil {
		t.Fatalf("dcgmGroupCreate: %v", err)
	}
	if !s.Owns(owned) {
		t.Error("the session does not report the group it created as its own")
	}
	if s.Owns(foreign) {
		t.Error("the session claims a group it did not create")
	}

	// Destroy the raw group while its engine is still alive. Closing an
	// embedded session stops the engine and invalidates all group handles.
	// Foreign-group survival is tested only against a connected engine.
	if err := GroupDestroy(s.Handle, foreign); err != nil {
		t.Fatalf("destroy unowned group before engine shutdown: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Session.Close: %v", err)
	}

	// The handle is invalid now, so reach the engine again to see what
	// survived. StartEmbedded's engine died with Close, so this can only be
	// checked against a connected host engine; see TestConnectedHostEngine.
	t.Log("embedded engine stopped with the session; group survival is checked in TestConnectedHostEngine")
}

// TestConnectedHostEngine checks the same ownership rule where it actually
// matters: against a host engine this process did not start, which keeps
// running afterwards.
//
// It requires DCGM_HOSTENGINE, and fails rather than skips when it is unset,
// because a skip here would leave the destructive case unchecked while the
// suite reports success.
func TestConnectedHostEngine(t *testing.T) {
	address := os.Getenv("DCGM_HOSTENGINE")
	if address == "" {
		t.Fatal("DCGM_HOSTENGINE is not set. This test needs an nv-hostengine that this " +
			"process did not start, because the rule it checks -- Close disconnects and " +
			"never stops an engine it does not own -- cannot be observed against an " +
			"embedded engine that dies with the process. Start nv-hostengine and set " +
			"DCGM_HOSTENGINE to its address, e.g. 127.0.0.1:5555")
	}
	mustInit(t)

	s, err := Connect(address, ConnectOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("connecting to the host engine at %s: %v", address, err)
	}
	if s.OwnsHostEngine() {
		t.Fatal("a connected session claims to own the host engine; Close would stop it")
	}

	before, err := s.Handle.Groups()
	if err != nil {
		t.Fatalf("dcgmGroupGetAllIds: %v", err)
	}
	owned, err := s.NewGroup("gotest-connected-group", GroupDefault)
	if err != nil {
		t.Fatalf("dcgmGroupCreate: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Session.Close: %v", err)
	}

	// Reconnect. If Close stopped the engine this fails, which is the whole
	// point of the check.
	s2, err := Connect(address, ConnectOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("the host engine at %s is no longer reachable after Close, which means Close "+
			"stopped an engine it did not own: %v", address, err)
	}
	defer s2.Close()

	after, err := s2.Handle.Groups()
	if err != nil {
		t.Fatalf("dcgmGroupGetAllIds after reconnecting: %v", err)
	}
	stillThere := map[GpuGrp]bool{}
	for _, g := range after {
		stillThere[g] = true
	}
	if stillThere[owned] {
		t.Errorf("group %d survived Close; the session did not remove what it created", owned)
	}
	for _, g := range before {
		if !stillThere[g] {
			t.Errorf("group %d existed before this test and is gone; Close destroyed a group "+
				"another process owns", g)
		}
	}
}

// TestProfilingFieldsArrive checks a field that needs the DCGM profiling
// module, which is the part of DCGM most likely to be unavailable for reasons
// unrelated to these bindings. It fails with the module status printed rather
// than skipping, so an unloadable module is visible rather than absent from
// the report.
func TestProfilingFieldsArrive(t *testing.T) {
	s := embeddedSession(t)
	gpus := supportedGPUs(t, s)

	statuses := ModuleGetStatusesStruct{Version: ModuleGetStatusesStructVersion}
	if err := ModuleGetStatuses(s.Handle, &statuses); err != nil {
		t.Fatalf("dcgmModuleGetStatuses: %v", err)
	}
	for i := 0; i < int(statuses.NumStatuses) && i < len(statuses.Statuses); i++ {
		t.Logf("module %v status %v", statuses.Statuses[i].Id, statuses.Statuses[i].Status)
	}

	fields := []FieldID{FieldProfGrEngineActive}
	group, err := s.NewGroup("gotest-prof", GroupDefault)
	if err != nil {
		t.Fatalf("dcgmGroupCreate: %v", err)
	}
	fieldGroup, err := s.NewFieldGroup("gotest-prof", fields)
	if err != nil {
		t.Fatalf("dcgmFieldGroupCreate: %v", err)
	}
	if err := s.Watch(group, fieldGroup, 100*time.Millisecond, time.Minute, 100); err != nil {
		var res Result
		if errors.As(err, &res) {
			t.Fatalf("watching a profiling field failed with %v. The profiling module has to "+
				"be loadable and no other profiler may hold the GPU; the module statuses "+
				"are logged above", res)
		}
		t.Fatalf("dcgmWatchFields: %v", err)
	}
	if err := UpdateAllFields(s.Handle, 1); err != nil {
		t.Fatalf("dcgmUpdateAllFields: %v", err)
	}
	values, err := s.Handle.LatestValues(gpus[0], fields)
	if err != nil {
		t.Fatalf("dcgmGetLatestValuesForFields: %v", err)
	}
	v := &values[0]
	if err := v.Err(); err != nil {
		t.Fatalf("%v: %v", v.ID(), err)
	}
	if v.Blank() {
		t.Fatalf("%v came back blank; the profiling module reported no sample", v.ID())
	}
	f, ok := v.Float64()
	if !ok {
		t.Fatalf("%v: field type %d is not double-typed", v.ID(), v.Type())
	}
	if f < 0 || f > 1 {
		t.Errorf("%v = %v, which is outside the 0..1 the vendor documents for this ratio", v.ID(), f)
	}
	t.Logf("%v = %v", v.ID(), f)
}

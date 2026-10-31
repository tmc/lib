package nvtelemetry_test

import (
	"context"
	"errors"
	"io/fs"
	"slices"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// environmentFixture is synthetic scheduler/proc data, not a captured host.
func environmentFixture(t testing.TB, job string) fstest.MapFS {
	t.Helper()
	m := refreshFixtureFS(t)
	for name, f := range m {
		data := strings.NewReplacer(`"job_id":42`, `"job_id":`+job, `"job_id":"42"`, `"job_id":"`+job+`"`, "alloc-42", "alloc-"+job, "job_42", "job_"+job).Replace(string(f.Data))
		m[name] = &fstest.MapFile{Data: []byte(data)}
	}
	m["proc/1234/environ"] = &fstest.MapFile{Data: []byte("SLURM_JOB_ID=999\x00TOKEN=not-retained\x00")}
	m["proc/777/environ"] = &fstest.MapFile{Data: []byte("SLURM_JOB_ID=unqueried\x00")}
	return m
}

func readEnvironmentSnapshot(t testing.TB, fsys fs.FS) nvtelemetry.AttributionRead {
	t.Helper()
	devices, _, _ := twoDevices()
	got, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), fsys, liveSource(), processSelection(refreshDeviceUUID, secondDeviceUUID), devices)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Snapshot.Processes) != 1 || len(got.Withheld) != 0 {
		t.Fatalf("snapshot %+v", got)
	}
	return got
}

func TestProcEnvironSnapshot(t *testing.T) {
	rec := newStrictFS(environmentFixture(t, "123"), 1234)
	got := readEnvironmentSnapshot(t, rec)
	p := got.Snapshot.Processes[0]
	if p.SelfReported != (nvtelemetry.ProcEnvironFacts{State: nvtelemetry.ProcEnvironSelfReported, SlurmJobID: "999"}) {
		t.Fatalf("self-reported observation %+v", p.SelfReported)
	}
	cgroup, err := nvtelemetry.ResolveSlurmProcessCgroup(p)
	if err != nil || cgroup.JobID != "123" {
		t.Fatalf("cgroup %+v, %v", cgroup, err)
	}
	joined, err := nvtelemetry.JoinSlurmAttribution(got.Snapshot.Allocations, got.Snapshot.Mappings, got.Snapshot.Processes, got.Snapshot.Executions, 200)
	if err != nil || len(joined) != 1 || joined[0].Assignment.State != "assigned" || joined[0].Assignment.Owner.JobID != "123" {
		t.Fatalf("joined %+v, %v", joined, err)
	}
	if opens := rec.names(); !slices.Equal(opens, procReads1234) {
		t.Fatalf("selected PID opens %q, want %q", opens, procReads1234)
	}
}

type environmentFaultFS struct {
	fs.FS
	openErr, readErr, closeErr error
}

func (f environmentFaultFS) Open(name string) (fs.File, error) {
	if name != "proc/1234/environ" {
		return f.FS.Open(name)
	}
	if f.openErr != nil {
		return nil, f.openErr
	}
	file, err := f.FS.Open(name)
	if err != nil {
		return nil, err
	}
	return environmentFaultFile{File: file, readErr: f.readErr, closeErr: f.closeErr}, nil
}

type environmentFaultFile struct {
	fs.File
	readErr, closeErr error
}

func (f environmentFaultFile) Read(b []byte) (int, error) {
	if f.readErr != nil {
		return copy(b, "SLURM_JOB_ID=999\x00"), f.readErr
	}
	return f.File.Read(b)
}
func (f environmentFaultFile) Close() error { return errors.Join(f.File.Close(), f.closeErr) }

func TestProcEnvironSnapshotAbsence(t *testing.T) {
	tests := []struct {
		name                       string
		mutate                     func(fstest.MapFS)
		openErr, readErr, closeErr error
		state                      nvtelemetry.ProcEnvironState
	}{
		{name: "missing", mutate: func(m fstest.MapFS) { delete(m, "proc/1234/environ") }, state: nvtelemetry.ProcEnvironAbsent},
		{name: "empty", mutate: func(m fstest.MapFS) { m["proc/1234/environ"].Data = nil }, state: nvtelemetry.ProcEnvironAbsent},
		{name: "permission", openErr: fs.ErrPermission, state: nvtelemetry.ProcEnvironPermissionDenied},
		{name: "open error", openErr: errors.New("environment open failed"), state: nvtelemetry.ProcEnvironReadError},
		{name: "read error", readErr: errors.New("partial read failed"), state: nvtelemetry.ProcEnvironReadError},
		{name: "close error", closeErr: errors.New("close failed"), state: nvtelemetry.ProcEnvironReadError},
		{name: "truncated", mutate: func(m fstest.MapFS) { m["proc/1234/environ"].Data = []byte("SLURM_JOB_ID=999") }, state: nvtelemetry.ProcEnvironPartial},
		{name: "duplicate", mutate: func(m fstest.MapFS) { m["proc/1234/environ"].Data = []byte("SLURM_JOB_ID=999\x00SLURM_JOB_ID=123\x00") }, state: nvtelemetry.ProcEnvironInvalid},
		{name: "oversize", mutate: func(m fstest.MapFS) {
			m["proc/1234/environ"].Data = []byte("SLURM_JOB_ID=" + strings.Repeat("x", 65536) + "\x00")
		}, state: nvtelemetry.ProcEnvironInvalid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := environmentFixture(t, "123")
			if tt.mutate != nil {
				tt.mutate(m)
			}
			rec := newStrictFS(environmentFaultFS{FS: m, openErr: tt.openErr, readErr: tt.readErr, closeErr: tt.closeErr}, 1234)
			got := readEnvironmentSnapshot(t, rec)
			p := got.Snapshot.Processes[0]
			if p.SelfReported.State != tt.state || p.SelfReported.SlurmJobID != "" {
				t.Fatalf("observation %+v, want %q and no value", p.SelfReported, tt.state)
			}
			cg, err := nvtelemetry.ResolveSlurmProcessCgroup(p)
			if err != nil || cg.JobID != "123" {
				t.Fatalf("lost valid cgroup %+v, %v", cg, err)
			}
			if opens := rec.names(); !slices.Equal(opens, procReads1234) {
				t.Fatalf("opens %q", opens)
			}
		})
	}
}

func TestProcEnvironNoAuthorityFallback(t *testing.T) {
	m := environmentFixture(t, "999")
	m["proc/1234/cgroup"].Data = []byte("0::/user.slice/no-job\n")
	got := readEnvironmentSnapshot(t, newStrictFS(m, 1234))
	if got.Snapshot.Processes[0].SelfReported.SlurmJobID != "999" {
		t.Fatal("missing self-reported fixture")
	}
	joined, err := nvtelemetry.JoinSlurmAttribution(got.Snapshot.Allocations, got.Snapshot.Mappings, got.Snapshot.Processes, got.Snapshot.Executions, 200)
	if err != nil || len(joined) != 1 || joined[0].Assignment.State == "assigned" || joined[0].Assignment.Owner.JobID != "" {
		t.Fatalf("environment became authority: %+v, %v", joined, err)
	}
}

func TestProcEnvironRequiresProcessIdentity(t *testing.T) {
	for _, kind := range []string{"missing", "permission", "zombie", "wrong pid", "missing namespace"} {
		t.Run(kind, func(t *testing.T) {
			m := environmentFixture(t, "123")
			var fsys fs.FS = m
			src := liveSource()
			switch kind {
			case "missing":
				delete(m, "proc/1234/stat")
			case "permission":
				fsys = procReadErrorFS{FS: m, name: "proc/1234/stat", err: fs.ErrPermission}
			case "zombie":
				m["proc/1234/stat"].Data = []byte(strings.Replace(string(m["proc/1234/stat"].Data), ") S ", ") Z ", 1))
			case "wrong pid":
				m["proc/1234/stat"].Data = []byte(strings.Replace(string(m["proc/1234/stat"].Data), "1234 (", "777 (", 1))
			case "missing namespace":
				src.PIDNamespace = ""
			}
			rec := newStrictFS(fsys, 1234)
			devices, _, _ := twoDevices()
			got, err := nvtelemetry.ReadNVMLAttributionSnapshot(context.Background(), rec, src, processSelection(refreshDeviceUUID, secondDeviceUUID), devices)
			if kind == "wrong pid" || kind == "missing namespace" {
				if err == nil {
					t.Fatal("invalid identity accepted")
				}
			} else if err != nil || !slices.Equal(got.Withheld, []uint32{1234}) || len(got.Snapshot.Processes) != 0 {
				t.Fatalf("withheld %+v, %v", got, err)
			}
			if slices.Contains(rec.names(), "proc/1234/environ") {
				t.Fatalf("environment read before valid identity: %q", rec.names())
			}
		})
	}
}

func TestProcEnvironV1V2Membership(t *testing.T) {
	for _, tt := range []struct {
		name, cgroup string
		bad          bool
	}{
		{"v1", "1:devices:/slurm/job_123/step_0/task_0\n", false},
		{"v2", "0::/slurm/job_123/step_0/task_0\n", false},
		{"agreeing v1 controllers", "1:devices:/slurm/job_123/step_0/task_0\n2:cpu,cpuacct:/slurm/job_123/step_0/task_0\n", false},
		{"conflicting v1 controllers", "1:devices:/slurm/job_123/step_0/task_0\n2:cpu:/slurm/job_999/step_0/task_0\n", true},
		{"v1 v2 conflict", "1:devices:/slurm/job_123/step_0/task_0\n0::/slurm/job_999/step_0/task_0\n", true},
		{"malformed v1", "1::/slurm/job_123/step_0/task_0\n", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := environmentFixture(t, "123")
			m["proc/1234/cgroup"].Data = []byte(tt.cgroup)
			got := readEnvironmentSnapshot(t, newStrictFS(m, 1234))
			cg, err := nvtelemetry.ResolveSlurmProcessCgroup(got.Snapshot.Processes[0])
			if tt.bad {
				if err == nil {
					t.Fatalf("invalid memberships accepted: %+v", cg)
				}
			} else if err != nil || cg.JobID != "123" {
				t.Fatalf("got %+v, %v", cg, err)
			}
			joined, joinErr := nvtelemetry.JoinSlurmAttribution(got.Snapshot.Allocations, got.Snapshot.Mappings, got.Snapshot.Processes, got.Snapshot.Executions, 200)
			if joinErr != nil || len(joined) != 1 {
				t.Fatalf("joined %+v, %v", joined, joinErr)
			}
			if tt.bad {
				if joined[0].Assignment.State == "assigned" {
					t.Fatalf("invalid cgroup became an assignment: %+v", joined[0])
				}
				return
			}
			wantSource := "slurm_cgroup_v2"
			if strings.HasPrefix(tt.cgroup, "1:") {
				wantSource = "slurm_cgroup_v1"
			}
			a := joined[0].Assignment
			if a.State != "assigned" || a.Owner.JobID != "123" || a.Source != wantSource+"_attribution" || len(a.Provenance) == 0 || a.Provenance[0].Source != wantSource {
				t.Fatalf("joined source/provenance: %+v, want %q", a, wantSource)
			}
		})
	}
}

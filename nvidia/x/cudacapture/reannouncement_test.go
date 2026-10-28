package cudacapture

import (
	"crypto/sha256"
	"fmt"
	"reflect"
	"testing"
)

func TestReannouncementLifecycle(t *testing.T) {
	var m ModuleTracker
	digestA := fmt.Sprintf("%x", sha256.Sum256([]byte("__global__ void first() {}")))
	digestB := fmt.Sprintf("%x", sha256.Sum256([]byte("__global__ void replacement() {}")))
	code := CodeFiles{Sources: []string{"source.cu"}}
	g, err := m.Load("first", 10, digestA, code)
	if err != nil || g != 1 {
		t.Fatalf("load: %d %v", g, err)
	}
	code.Sources[0] = "changed"
	if g, err = m.Load("duplicate", 10, digestA, CodeFiles{}); err != nil || g != 1 {
		t.Fatalf("duplicate: %d %v", g, err)
	}
	if err := m.Unload(10); err != nil {
		t.Fatal(err)
	}
	if err := m.Unload(10); err != nil {
		t.Fatal(err)
	}
	if g, err = m.Load("reload", 10, digestA, CodeFiles{}); err != nil || g != 2 {
		t.Fatalf("reload: %d %v", g, err)
	}
	if g, err = m.Load("replacement", 10, digestB, CodeFiles{}); err != nil || g != 3 {
		t.Fatalf("replacement: %d %v", g, err)
	}
	old, err := m.Lookup(10, 1)
	if err != nil || old.Active || old.Code.Sources[0] != "source.cu" {
		t.Fatalf("old: %+v %v", old, err)
	}
	old.Code.Sources[0] = "corrupt"
	if err := m.Verify(10, 1, digestA); err != nil {
		t.Fatal(err)
	}
	if err := m.Verify(10, 1, digestB); err == nil {
		t.Fatal("borrowed another image digest")
	}
	r, err := m.Reannounce(ReannouncementLimits{MaxEntries: 3, MaxBytes: 4096})
	if err != nil || len(r.Entries) != 3 || r.Entries[0].Active || r.Entries[1].Active || !r.Entries[2].Active {
		t.Fatalf("snapshot: %+v %v", r, err)
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	if !r.Entries[1].Code.Empty() {
		t.Fatal("invented stripped code")
	}
	r.Entries[0].Code.Sources[0] = "bad snapshot"
	r2, _ := m.Reannounce(r.Limits)
	if r2.Entries[0].Code.Sources[0] != "source.cu" {
		t.Fatal("snapshot aliases owner")
	}
	c := r2.Clone()
	c.Entries[0].Code.Sources[0] = "bad clone"
	if r2.Entries[0].Code.Sources[0] != "source.cu" {
		t.Fatal("clone aliases snapshot")
	}
	if err := m.Unload(11); err == nil {
		t.Fatal("unobserved unload accepted")
	}
}

func TestReannouncementLimits(t *testing.T) {
	m := NewModuleTracker()
	for _, addr := range []uint64{30, 10, 20} {
		if _, err := m.Load("m", addr, "digest", CodeFiles{}); err != nil {
			t.Fatal(err)
		}
	}
	n := uint64(256 + len("m") + len("digest"))
	for _, tt := range []struct {
		name         string
		limit        ReannouncementLimits
		want         int
		count, bytes uint64
	}{
		{"exact", ReannouncementLimits{3, n * 3}, 3, 0, 0},
		{"count", ReannouncementLimits{2, n * 3}, 2, 1, 0},
		{"byte", ReannouncementLimits{3, n*3 - 1}, 2, 0, 1},
		{"too small", ReannouncementLimits{3, n - 1}, 0, 0, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := m.Reannounce(tt.limit)
			if err != nil || len(r.Entries) != tt.want || r.OmittedCountLimit != tt.count || r.OmittedByteLimit != tt.bytes {
				t.Fatalf("%+v %v", r, err)
			}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
			for i, e := range r.Entries {
				if e.BaseAddress != uint64((i+1)*10) {
					t.Fatal("nondeterministic order")
				}
			}
		})
	}
	if _, err := m.Reannounce(ReannouncementLimits{}); err == nil {
		t.Fatal("zero limit accepted")
	}
	a, _ := m.Reannounce(ReannouncementLimits{3, n * 3})
	b, _ := m.Reannounce(a.Limits)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("snapshot changed without lifecycle change")
	}
	a.Bytes++
	if err := a.Validate(); err == nil {
		t.Fatal("incorrect charge accepted")
	}
}

func ExampleModuleTracker_Unload() {
	m := NewModuleTracker()
	m.Load("image", 16, "digest", CodeFiles{})
	m.Unload(16)
	generation, _ := m.Load("image", 16, "digest", CodeFiles{})
	fmt.Println(generation)
	// Output: 2
}

func ExampleModuleTracker_Reannounce() {
	m := NewModuleTracker()
	m.Load("stripped image", 16, "digest", CodeFiles{})
	r, _ := m.Reannounce(ReannouncementLimits{MaxEntries: 8, MaxBytes: 4096})
	fmt.Println(len(r.Entries), r.Entries[0].Code.Empty())
	// Output: 1 true
}

func ExampleReannouncement_Clone() {
	m := NewModuleTracker()
	m.Load("image", 16, "digest", CodeFiles{})
	r, _ := m.Reannounce(ReannouncementLimits{MaxEntries: 8, MaxBytes: 4096})
	c := r.Clone()
	c.Entries[0].ID = "changed"
	fmt.Println(r.Entries[0].ID)
	// Output: image
}

func ExampleReannouncement_Validate() {
	m := NewModuleTracker()
	r, _ := m.Reannounce(ReannouncementLimits{MaxEntries: 8, MaxBytes: 4096})
	fmt.Println(r.Validate())
	// Output: <nil>
}

func TestReannouncementCodeByteCharge(t *testing.T) {
	m := NewModuleTracker()
	code := CodeFiles{Source: "a", Sources: []string{"a", "b"}, PTX: "c", CUBIN: "d", SASS: "e", Unit: "f", TileIR: "g", CFG: "h", BBCFG: "i"}
	if _, err := m.Load("m", 1, "digest", code); err != nil {
		t.Fatal(err)
	}
	// Independently enumerate all ten filename bytes and two Sources slots.
	const charged = 256 + 1 + 6 + 10 + 2*16
	r, err := m.Reannounce(ReannouncementLimits{MaxEntries: 1, MaxBytes: charged})
	if err != nil || len(r.Entries) != 1 || r.Bytes != charged || r.OmittedByteLimit != 0 {
		t.Fatalf("exact: %+v %v", r, err)
	}
	r, err = m.Reannounce(ReannouncementLimits{MaxEntries: 1, MaxBytes: charged - 1})
	if err != nil || len(r.Entries) != 0 || r.Bytes != 0 || r.OmittedByteLimit != 1 {
		t.Fatalf("one short: %+v %v", r, err)
	}
}

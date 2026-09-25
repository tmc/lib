package main

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// moduleSession is the session every test here writes journals under.
const moduleSession = "0123456789abcdef0123456789abcdef"

func moduleSessionID(t *testing.T) [16]byte {
	t.Helper()
	raw, err := hex.DecodeString(moduleSession)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)
	return sid
}

// TestModuleRecordsDecode pins the module record kind end to end through the
// journal reader. The kind is closed -- an unknown one costs the whole capture
// -- so this is also the check that a journal carrying modules still parses.
func TestModuleRecordsDecode(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, moduleSessionID(t), 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindModule, Payload: []byte(`{"kind":"module","module_id":22,"module_size":4096,"module_file":"module-0000-id22.cubin","module_stored":true,"module_reason":""}`)},
		{Kind: wire.KindModule, Payload: []byte(`{"kind":"module","module_id":23,"module_size":99999999,"module_file":"","module_stored":false,"module_reason":"file_budget"}`)},
		{Kind: wire.KindModule, Payload: []byte(`{"kind":"module","module_id":22,"module_size":4096,"module_file":"module-0002-id22.cubin","module_stored":true,"module_reason":"","module_unchecked":true}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200,"module_id":22}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, moduleSession, 1234, 0)
	if err != nil {
		t.Fatalf("a module record cost the whole capture: %v", err)
	}
	if len(activities) != 1 {
		t.Fatalf("got %d activities, want 1: module records must not enter the activity totals", len(activities))
	}
	// The module a kernel came from rides on the kernel's own record: a
	// kernel activity names only the kernel, so this is the only thing that
	// says which of two builds of one graph ran.
	if activities[0].ModuleID != 22 {
		t.Errorf("kernel activity module = %d, want 22", activities[0].ModuleID)
	}
	if len(metrics.modules) != 3 {
		t.Fatalf("got %d modules, want 3", len(metrics.modules))
	}
	if got := metrics.modules[0]; got.ID != 22 || got.Size != 4096 || got.File != "module-0000-id22.cubin" || !got.Stored {
		t.Errorf("stored module = %+v", got)
	}
	// The module the producer declined to keep must survive as a record of
	// its own. Dropping it would make a module too large to store
	// indistinguishable from one the target never loaded.
	if got := metrics.modules[1]; got.ID != 23 || got.Stored || got.Reason != "file_budget" || got.Size != 99999999 {
		t.Errorf("omitted module = %+v", got)
	}
	// A load past the agent's duplicate table may repeat an earlier image,
	// and only the record's own flag says so.
	if metrics.modules[0].Unchecked || metrics.modules[1].Unchecked || !metrics.modules[2].Unchecked {
		t.Errorf("unchecked flags = %v %v %v, want false false true",
			metrics.modules[0].Unchecked, metrics.modules[1].Unchecked, metrics.modules[2].Unchecked)
	}
	head := captureHeadline(captureResult{RunCompleteness: runComplete, ChildrenFollowed: true, Modules: metrics.modules})
	if !strings.Contains(head, "1 module load(s) arrived after the agent's table of seen modules was full") {
		t.Errorf("headline does not report the unchecked load:\n%s", head)
	}
	if head := captureHeadline(captureResult{RunCompleteness: runComplete, ChildrenFollowed: true, Modules: metrics.modules[:2]}); strings.Contains(head, "table of seen modules") {
		t.Errorf("headline reports unchecked loads where there are none:\n%s", head)
	}
}

// TestModuleImagesAreEmbedded runs the harvest: images the producer wrote
// beside a worker journal must reach the published capture, be named by it,
// and be reachable under the names it gives them.
func TestModuleImagesAreEmbedded(t *testing.T) {
	scratch := t.TempDir()
	worker := filepath.Join(scratch, "process-1234-1")
	if err := os.MkdirAll(filepath.Join(worker, "modules"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Two images and one file that is not one. The non-cubin is the control:
	// the harvest must select by role, not sweep the directory.
	images := map[string][]byte{
		"module-0000-id22.cubin": []byte("\x7fELFcubin-a"),
		"module-0001-id23.cubin": []byte("\x7fELFcubin-b"),
		"notes.txt":              []byte("not device code"),
	}
	for name, data := range images {
		if err := os.WriteFile(filepath.Join(worker, "modules", name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{
		Session:    moduleSession,
		TargetPID:  1234,
		Outcome:    captureOutcomeCollected,
		ScratchDir: scratch,
		Activities: []captureActivity{{Subtype: "kernel", Name: "k", Valid: true, Start: 100, End: 200}},
	}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatalf("write capture: %v", err)
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := cudacapture.Read(f)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}

	if len(c.Modules) != 2 {
		t.Fatalf("capture names %d modules, want 2: %+v", len(c.Modules), c.Modules)
	}
	for _, m := range c.Modules {
		if m.Code.CUBIN == "" {
			t.Errorf("module %q names no cubin", m.ID)
			continue
		}
		data, ok := c.Files[m.Code.CUBIN]
		if !ok {
			t.Errorf("module %q names %q, which is not in the capture", m.ID, m.Code.CUBIN)
			continue
		}
		// The bytes must be the producer's, not merely some bytes: a harvest
		// that attached the wrong file would still satisfy every check above.
		want, found := images[m.Code.CUBIN[strings.LastIndex(m.Code.CUBIN, "module-"):]]
		if !found {
			t.Errorf("module %q has no source image", m.Code.CUBIN)
			continue
		}
		if string(data) != string(want) {
			t.Errorf("module %q = %q, want %q", m.Code.CUBIN, data, want)
		}
		// Names carry the worker namespace, which is what keeps two processes
		// that each loaded a first module from colliding on module-0000.
		if !strings.HasPrefix(m.Code.CUBIN, "worker-process-1234-1-") {
			t.Errorf("module file %q is not scoped to its worker", m.Code.CUBIN)
		}
	}
	if _, ok := c.Files["worker-process-1234-1-notes.txt"]; ok {
		t.Error("the harvest swept a file that is not device code")
	}

	// The whole-capture cubin role stays unavailable, and must say why: a
	// reader that sees the role absent and the reason missing reads it as
	// "no device code was captured", which is the opposite of the truth.
	reason := c.CodeUnavailable["cubin"]
	if reason == "" {
		t.Error("cubin has no availability disposition")
	}
	if !strings.Contains(reason, "module") {
		t.Errorf("cubin disposition %q does not point at the modules that carry it", reason)
	}
}

// TestModuleAttributionLabelsSamples runs a numbered module and a kernel that
// names it through the published capture. The label the profile carries is
// the driver's number, which is what a consumer splitting its tables by
// module matches against; the name of the stored image is not it, because
// the producer never learns that name.
func TestModuleAttributionLabelsSamples(t *testing.T) {
	scratch := t.TempDir()
	worker := filepath.Join(scratch, "process-1234-1")
	if err := os.MkdirAll(filepath.Join(worker, "modules"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(worker, "modules", "module-0000-id22.cubin"), []byte("\x7fELFcubin-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{
		Session:    moduleSession,
		TargetPID:  1234,
		Outcome:    captureOutcomeCollected,
		ScratchDir: scratch,
		Modules:    []captureModule{{ID: 22, Size: 12, File: "module-0000-id22.cubin", Stored: true}},
		Activities: []captureActivity{{Subtype: "kernel", Name: "k", Valid: true, Start: 100, End: 200, ModuleID: 22}},
	}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatalf("write capture: %v", err)
	}
	f, err := os.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := cudacapture.Read(f)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	if len(c.Modules) != 1 || c.Modules[0].DriverID != 22 {
		t.Fatalf("modules = %+v, want one numbered 22", c.Modules)
	}
	var labelled int
	for _, s := range c.Profile.Sample {
		if ids := s.Label["module_id"]; len(ids) == 1 && ids[0] == "22" {
			labelled++
		}
	}
	if labelled != 1 {
		t.Errorf("%d samples name module 22, want 1", labelled)
	}
	if got := moduleLabel(c.Modules[0]); got != "22" {
		t.Errorf("moduleLabel = %q, want \"22\"", got)
	}
}

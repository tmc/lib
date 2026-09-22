package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// TestPCSampleRecordsDecode pins the PC sample record kind through the
// journal reader. Like every other kind it is closed: a reader that met it
// and did not know it would refuse the whole capture, which is why the agent
// writes it only when this process asked for it.
func TestPCSampleRecordsDecode(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, moduleSessionID(t), 1234, 1, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200,"module_id":22}`)},
		{Kind: recordPCSample, Payload: []byte(`{"kind":"pcsample","module_id":22,"pc_offset":112,"pc_samples":9,"pc_function":"fused_gate"}`)},
		{Kind: recordPCSample, Payload: []byte(`{"kind":"pcsample","module_id":23,"pc_offset":16,"pc_samples":4,"pc_function":"fused_gate"}`)},
		{Kind: recordPCSample, Payload: []byte(`{"kind":"pcsampling","pc_period":10,"pc_hardware_bytes":16777216,"pc_distinct":13,"pc_written":13,"pc_folded":78,"pc_total":144,"pc_hw_dropped":0,"pc_unkeyed":0,"pc_overflows":0}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, moduleSession, 1234, 0)
	if err != nil {
		t.Fatalf("a PC sample record cost the whole capture: %v", err)
	}
	if len(activities) != 1 {
		t.Fatalf("got %d activities, want 1: PC samples must not enter the activity totals", len(activities))
	}
	if len(metrics.pcSamples) != 2 {
		t.Fatalf("got %d PC samples, want 2", len(metrics.pcSamples))
	}
	if got := metrics.pcSamples[0]; got.ModuleID != 22 || got.Offset != 112 || got.Samples != 9 || got.Function != "fused_gate" {
		t.Errorf("PC sample = %+v", got)
	}
	// The summary is what says how much of the run the samples above account
	// for. Folded below Total is the honest state, not a failure, so the
	// numbers have to survive rather than be collapsed into the samples.
	if metrics.pcSampling == nil {
		t.Fatal("no sampling summary")
	}
	if got := *metrics.pcSampling; got.Period != 10 || got.Distinct != 13 || got.Folded != 78 || got.Total != 144 {
		t.Errorf("sampling summary = %+v", got)
	}
}

// TestPCSampleRecordRejectsUnknownKind keeps the kind closed from the inside
// too: the record kind carries several payload kinds, and one this reader
// does not know must fail rather than be counted as a sample of nothing.
func TestPCSampleRecordRejectsUnknownKind(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, moduleSessionID(t), 1234, 1, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordPCSample, Payload: []byte(`{"kind":"pcwhatever"}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
	})
	if _, _, _, _, _, _, err := observeJournal(journalPath, moduleSession, 1234, 0); err == nil {
		t.Fatal("an unknown PC sample payload was accepted")
	}
}

// fusedGateSASS is a disassembly with the shape nvdisasm produces: a function,
// addresses, and the line comments that carry the program site.
const fusedGateSASS = `	.target	sm_121

//--------------------- .text.fused_gate          --------------------------
	.section	.text.fused_gate,"ax",@progbits
        .global         fused_gate
        .type           fused_gate,@function
fused_gate:
.text.fused_gate:
	//## File "fused_gate.cu", line 3
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
	//## File "kernels.go", line 15
        /*0010*/                   S2R R11, SR_CTAID.X ;
        /*0020*/                   LDCU UR4, c[0x0][0x360] ;
	//## File "kernels.go", line 17
        /*0030*/                   FFMA R5, R4, R6, R7 ;
`

// TestPCSampleResolvesOffset is the join the whole projection rests on: CUPTI
// reports a function and a byte offset into it, and a per-line total needs the
// instruction's position and the line it came from.
func TestPCSampleResolvesOffset(t *testing.T) {
	dis, err := cudasass.ParseString(fusedGateSASS)
	if err != nil {
		t.Fatal(err)
	}
	code := &moduleCode{name: "module-0000-id22", dis: dis}
	tests := []struct {
		name    string
		fn      string
		offset  uint64
		ordinal int
		file    string
		line    int
	}{
		{"first", "fused_gate", 0x0, 1, "fused_gate.cu", 3},
		{"program line", "fused_gate", 0x10, 2, "kernels.go", 15},
		{"carried line", "fused_gate", 0x20, 3, "kernels.go", 15},
		{"last", "fused_gate", 0x30, 4, "kernels.go", 17},
		// An offset between instructions belongs to no instruction. It is
		// reported as unresolved rather than rounded down: a per-line total
		// that guesses is worse than one that says it does not know.
		{"unaligned", "fused_gate", 0x18, 0, "", 0},
		{"unknown function", "other", 0x0, 0, "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, ordinal, file, line := code.resolve(tt.fn, tt.offset)
			if ordinal != tt.ordinal || file != tt.file || line != tt.line {
				t.Errorf("resolve(%q, %#x) = %q %d %q:%d, want ordinal %d at %q:%d", tt.fn, tt.offset, name, ordinal, file, line, tt.ordinal, tt.file, tt.line)
			}
		})
	}
}

// TestPCSamplesProjectFrames builds the profile the sampler contributes and
// checks it has the shape the line tables read: a leaf SASS frame carrying
// the instruction's ordinal, the kernel's own frame carrying the program
// line, and the module label that says which build ran.
func TestPCSamplesProjectFrames(t *testing.T) {
	dis, err := cudasass.ParseString(fusedGateSASS)
	if err != nil {
		t.Fatal(err)
	}
	c := &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Value: []int64{100}}},
	}}
	b := &pcBuilder{p: c.Profile, width: 2, functions: map[string]*profile.Function{}, locations: map[string]*profile.Location{}}
	c.Profile.SampleType = append(c.Profile.SampleType, &profile.ValueType{Type: pcSampleType, Unit: "samples"})
	c.Profile.Sample[0].Value = append(c.Profile.Sample[0].Value, 0)
	b.add(deviceCount{ModuleID: 22, Function: "fused_gate", Offset: 0x10, Value: 7}, &moduleCode{name: "module-0000-id22", dis: dis})

	if len(c.Profile.Sample) != 2 {
		t.Fatalf("got %d samples, want 2", len(c.Profile.Sample))
	}
	s := c.Profile.Sample[1]
	if s.Value[1] != 7 {
		t.Errorf("sample value = %v, want 7 in the PC dimension", s.Value)
	}
	if ids := s.Label["module_id"]; len(ids) != 1 || ids[0] != "22" {
		t.Errorf("module label = %v, want 22", s.Label["module_id"])
	}
	f := deviceFrames(s)
	if f.kernel != "fused_gate" {
		t.Errorf("kernel = %q, want fused_gate", f.kernel)
	}
	if f.sassLine != 2 {
		t.Errorf("SASS ordinal = %d, want 2", f.sassLine)
	}
	if f.site.File != "kernels.go" || f.site.Line != 15 {
		t.Errorf("site = %+v, want kernels.go:15", f.site)
	}
	// A profile pprof cannot parse is not a profile. The projection writes
	// its own function and location IDs beside the ones the activity builder
	// wrote, and a collision there is invisible until a reader rejects it.
	if err := c.Profile.CheckValid(); err != nil {
		t.Errorf("projected profile is invalid: %v", err)
	}
}

// TestPCSamplesReachTheCapture runs the whole path: a result carrying PC
// samples must publish a capture whose profile totals them, with the samples
// that are not PC samples widened rather than silently truncated.
func TestPCSamplesReachTheCapture(t *testing.T) {
	scratch := t.TempDir()
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{
		Session:    moduleSession,
		TargetPID:  1234,
		Outcome:    captureOutcomeCollected,
		ScratchDir: scratch,
		Activities: []captureActivity{{Subtype: "kernel", Name: "fused_gate", Valid: true, Start: 100, End: 200}},
		PCSamples:  []capturePCSample{{ModuleID: 22, Function: "fused_gate", Offset: 0x10, Samples: 7}},
		PCSampling: &capturePCSampling{Period: 10, Distinct: 1, Written: 1, Folded: 7, Total: 7},
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
	idx := -1
	for i, st := range c.Profile.SampleType {
		if st.Type == pcSampleType {
			idx = i
		}
	}
	if idx < 0 {
		t.Fatalf("capture has no %s dimension", pcSampleType)
	}
	var total int64
	for _, s := range c.Profile.Sample {
		if idx >= len(s.Value) {
			t.Fatalf("sample %v was not widened to the new dimension", s.Value)
		}
		total += s.Value[idx]
	}
	if total != 7 {
		t.Errorf("PC samples total %d, want 7", total)
	}
}

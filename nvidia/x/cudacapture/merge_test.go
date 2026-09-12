package cudacapture_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func rankRecording(t testing.TB, device string) []byte {
	t.Helper()
	f := &profile.Function{ID: 1, Name: "same_kernel"}
	m := &profile.Mapping{ID: 1}
	l := &profile.Location{ID: 1, Mapping: m, Line: []profile.Line{{Function: f}}}
	c := &cudacapture.Capture{
		Profile:         &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}, Mapping: []*profile.Mapping{m}, Function: []*profile.Function{f}, Location: []*profile.Location{l}, Sample: []*profile.Sample{{Location: []*profile.Location{l}, Value: []int64{17}, Label: map[string][]string{"device": {device}, "stream": {"7"}}}}},
		Files:           map[string][]byte{"capture.journal": {0, 1, 2}, "code.cubin": {3, 4}},
		Modules:         []cudacapture.Module{{ID: "module", Generation: 7, Code: cudacapture.CodeFiles{CUBIN: "code.cubin"}}},
		CodeUnavailable: map[string]string{"source": "unavailable", "ptx": "unavailable", "cubin": "per module", "sass": "unavailable"},
		Metadata:        json.RawMessage(`{"dropped":18446744073709551615}`),
		Descriptors:     []cudacapture.MetricDescriptor{{ID: "bytes", Name: "bytes", NumericType: "uint64", Scope: "device"}},
		Observations:    []cudacapture.Observation{{DescriptorID: "bytes", EntityID: "device", RunID: "run", Value: json.RawMessage(`18446744073709551615`), Availability: "collected"}, {DescriptorID: "bytes", EntityID: "other", Availability: "unsupported"}},
	}
	var b bytes.Buffer
	if err := cudacapture.Write(&b, c); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestMergeRanks(t *testing.T) {
	a, b := rankRecording(t, "0"), rankRecording(t, "1")
	inputs := []cudacapture.RankCapture{{Session: "session", Run: "run", Rank: 1, Recording: b}, {Session: "session", Run: "run", Rank: 0, Recording: a}}
	got, err := cudacapture.MergeRanks(inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Profile.Sample) != 2 || len(got.Profile.Function) != 2 || len(got.Profile.Location) != 2 || len(got.Profile.Mapping) != 2 {
		t.Fatal("rank code or samples collapsed")
	}
	for i, s := range got.Profile.Sample {
		if s.Value[0] != 17 || s.Label["capture.rank"][0] != fmt.Sprint(i) || s.Label["device"][0] != fmt.Sprint(i) || s.Label["stream"][0] != "7" {
			t.Fatalf("sample=%+v", s)
		}
		if s.Location[0].ID != uint64(i+1) || s.Location[0].Line[0].Function.ID != uint64(i+1) {
			t.Fatal("ID collision")
		}
	}
	if !bytes.Equal(got.Files["rank0-original.pprof"], a) || !bytes.Equal(got.Files["rank1-original.pprof"], b) {
		t.Fatal("original recording changed")
	}
	if got.Observations[0].DescriptorID == got.Observations[2].DescriptorID || got.Observations[0].EntityID == got.Observations[2].EntityID {
		t.Fatal("typed identities collide")
	}
	if string(got.Observations[0].Value) != "18446744073709551615" || got.Observations[1].Availability != "unsupported" || !bytes.Contains(got.Metadata, []byte(`"dropped":18446744073709551615`)) {
		t.Fatal("value/loss/availability changed")
	}
	if got.Modules[0].ID == got.Modules[1].ID || !bytes.Equal(got.Files[got.Modules[1].Code.CUBIN], []byte{3, 4}) {
		t.Fatal("module namespace lost")
	}
	var first, second bytes.Buffer
	if err := cudacapture.Write(&first, got); err != nil {
		t.Fatal(err)
	}
	inputs[0], inputs[1] = inputs[1], inputs[0]
	again, err := cudacapture.MergeRanks(inputs)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.Write(&second, again); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("nondeterministic aggregate")
	}
	if _, err := cudacapture.Read(bytes.NewReader(first.Bytes())); err != nil {
		t.Fatal(err)
	}
	// An aggregate owns its data and does not mutate a caller's original bytes.
	got.Files["rank0-original.pprof"][0] ^= 1
	if !bytes.Equal(a, inputs[0].Recording) {
		t.Fatal("input modified")
	}
}

func TestMergeRanksRejectsConflicts(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([]cudacapture.RankCapture)
	}{
		{"session", func(r []cudacapture.RankCapture) { r[1].Session = "other" }},
		{"run", func(r []cudacapture.RankCapture) { r[1].Run = "other" }},
		{"rank", func(r []cudacapture.RankCapture) { r[1].Rank = 0 }},
		{"missing identity", func(r []cudacapture.RankCapture) { r[0].Session = "" }},
		{"invalid recording", func(r []cudacapture.RankCapture) { r[0].Recording = []byte("bad") }},
		{"observation run", func(r []cudacapture.RankCapture) { r[0].Run = "other"; r[1].Run = "other" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := []cudacapture.RankCapture{{Session: "session", Run: "run", Recording: rankRecording(t, "0")}, {Session: "session", Run: "run", Rank: 1, Recording: rankRecording(t, "1")}}
			tt.change(r)
			if _, err := cudacapture.MergeRanks(r); err == nil {
				t.Fatal("conflict accepted")
			}
		})
	}
	if _, err := cudacapture.MergeRanks(nil); err == nil {
		t.Fatal("empty ranks accepted")
	}
}

func ExampleMergeRanks() {
	_, err := cudacapture.MergeRanks(nil)
	fmt.Println(err)
	// Output: cudacapture: no ranks to merge
}

func TestMergeRanksRejectsProfileConflicts(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*profile.Profile)
	}{
		{"rank label", func(p *profile.Profile) { p.Sample[0].Label["capture.rank"] = []string{"9"} }},
		{"session label", func(p *profile.Profile) { p.Sample[0].Label["capture.session"] = []string{"other"} }},
		{"numeric rank", func(p *profile.Profile) { p.Sample[0].NumLabel = map[string][]int64{"capture.rank": {1}} }},
		{"different units", func(p *profile.Profile) { p.SampleType[0].Unit = "bytes" }},
		{"overflow", func(p *profile.Profile) { p.Sample[0].Value[0] = 1<<63 - 1 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := rankRecording(t, "0")
			c, err := cudacapture.Read(bytes.NewReader(rankRecording(t, "1")))
			if err != nil {
				t.Fatal(err)
			}
			tt.change(c.Profile)
			c.Profile.Comments = nil
			var b bytes.Buffer
			if err := cudacapture.Write(&b, c); err != nil {
				t.Fatal(err)
			}
			if _, err := cudacapture.MergeRanks([]cudacapture.RankCapture{{Session: "session", Run: "run", Recording: a}, {Session: "session", Run: "run", Rank: 1, Recording: b.Bytes()}}); err == nil {
				t.Fatal("profile conflict accepted")
			}
		})
	}
}

func ExampleRankCapture() {
	r := cudacapture.RankCapture{Session: "training", Run: "step-1", Rank: 0}
	fmt.Println(r.Session, r.Run, r.Rank)
	// Output: training step-1 0
}

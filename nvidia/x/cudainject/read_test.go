package cudainject_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudainject"
)

func frame(t *testing.T, sequence, dropped uint64) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := cudainject.WriteRecord(&buf, cudainject.Record{Kind: cudainject.KindActivity, Sequence: sequence, Dropped: dropped, Payload: []byte(`{}`)}, cudainject.Limits{}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestCaptureReaderUsesCaptureFormat(t *testing.T) {
	var stream bytes.Buffer
	capture := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}}, Files: map[string][]byte{"note": []byte("retained")}, CodeUnavailable: map[string]string{"source": "unavailable", "ptx": "unavailable", "cubin": "unavailable", "sass": "unavailable"}}
	if err := cudacapture.Write(&stream, capture); err != nil {
		t.Fatal(err)
	}
	read, err := cudainject.ReadCapture(&stream, cudacapture.Limits{})
	if err != nil || string(read.Files["note"]) != "retained" {
		t.Fatalf("%v %v", read, err)
	}
	if _, err := cudainject.ReadCapture(bytes.NewReader(frame(t, 1, 0)), cudacapture.Limits{}); err == nil {
		t.Fatal("wire record accepted as capture")
	}
}

func TestProtocolVersionRejectedWhenUnknown(t *testing.T) {
	b := frame(t, 1, 0)
	binary.BigEndian.PutUint16(b[4:6], cudainject.WireVersion+1)
	_, err := cudainject.NewDecoder(bytes.NewReader(b), cudainject.Limits{}).Next()
	if !errors.Is(err, cudainject.ErrBadVersion) {
		t.Fatalf("version: %v", err)
	}
}

func TestDrainReportsDropsNotSilence(t *testing.T) {
	d := cudainject.NewDecoder(bytes.NewReader(frame(t, 1, 7)), cudainject.Limits{})
	r, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if r.Dropped != 7 || d.Stats().Dropped != 7 {
		t.Fatalf("loss hidden: %v %v", r, d.Stats())
	}
	if _, err = d.Next(); err != io.EOF {
		t.Fatal(err)
	}
}

func TestSequenceGapDetected(t *testing.T) {
	d := cudainject.NewDecoder(bytes.NewReader(append(frame(t, 1, 0), frame(t, 4, 0)...)), cudainject.Limits{})
	for range 2 {
		if _, err := d.Next(); err != nil {
			t.Fatal(err)
		}
	}
	if d.Stats().Gaps != 2 {
		t.Fatal(d.Stats())
	}
}

func TestInputBoundsAndTornJournal(t *testing.T) {
	b := frame(t, 1, 0)
	for _, n := range []int{1, 79, len(b) - 1} {
		if _, err := cudainject.NewDecoder(bytes.NewReader(b[:n]), cudainject.Limits{}).Next(); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("n=%d: %v", n, err)
		}
	}
	if _, err := cudainject.NewDecoder(bytes.NewReader(b), cudainject.Limits{MaxPayload: 1}).Next(); !errors.Is(err, cudainject.ErrOversize) {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "journal")
	if err := os.WriteFile(path, append(b, b[:10]...), 0600); err != nil {
		t.Fatal(err)
	}
	recovery, err := cudainject.ReadJournal(path, cudainject.Limits{})
	if err != nil || !recovery.Truncated || recovery.Complete || len(recovery.Records) != 1 {
		t.Fatalf("%+v %v", recovery, err)
	}
}

func TestCorrelationIdJoinsLaunchToKernel(t *testing.T) {
	records := []cudainject.Record{
		{Kind: cudainject.KindAPI, Payload: []byte(`{"domain":"driver","callback_id":211,"thread_id":91,"correlation":37}`)},
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"vector","valid":true,"start":3,"end":9,"correlation":37}`)},
	}
	r, err := cudainject.JoinCorrelations(records)
	if err != nil || len(r.Matched) != 1 || r.Matched[0].HostThreadID != 91 || r.Matched[0].CorrelationID != 37 || r.APIs[0].CallbackID != 211 || r.APIs[0].Domain != "driver" {
		t.Fatalf("%+v %v", r, err)
	}
	duplicate := append(append([]cudainject.Record{}, records...), records[0])
	r, err = cudainject.JoinCorrelations(duplicate)
	if err != nil || len(r.Matched) != 0 || len(r.Ambiguous) != 1 {
		t.Fatalf("%+v %v", r, err)
	}
	records[0].Identity.Process = 1
	if _, err = cudainject.JoinCorrelations(records); err == nil {
		t.Fatal("cross-process join accepted")
	}
	r, err = cudainject.JoinCorrelations(records[1:])
	if err != nil || len(r.MissingAPI) != 1 || r.MissingAPI[0] != 37 {
		t.Fatalf("%+v %v", r, err)
	}
}

func TestExternalCorrelationReportsMissingStackAndAmbiguity(t *testing.T) {
	records := []cudainject.Record{
		{Kind: cudainject.KindAPI, Payload: []byte(`{"domain":"runtime","callback_id":0,"thread_id":5,"correlation":8}`)},
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":true,"start":1,"end":2,"correlation":8}`)},
		{Kind: cudainject.KindExternal, Payload: []byte(`{"external_kind":1,"external_id":17,"correlation":8}`)},
	}
	r, err := cudainject.JoinCorrelations(records)
	if err != nil || len(r.Matched) != 1 || !r.Matched[0].HasExternalID || r.Matched[0].ExternalID != 17 || !r.Matched[0].CallstackMissing {
		t.Fatalf("%+v %v", r, err)
	}
	records = append(records, cudainject.Record{Kind: cudainject.KindExternal, Payload: []byte(`{"external_kind":1,"external_id":18,"correlation":8}`)})
	r, err = cudainject.JoinCorrelations(records)
	if err != nil || len(r.Matched) != 0 || len(r.Ambiguous) != 1 {
		t.Fatalf("%+v %v", r, err)
	}
}

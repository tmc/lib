package dcgmjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func reportBody(result string) string {
	return `{"metadata":{"version":"4.6.1"},"DCGM Diagnostic":{"test_categories":[{"category":"Hardware","tests":[{"name":"memory","results":[` + result + `]}]}]}}`
}

const gpuFields = `"entity_group":"GPU","entity_group_id":1,"entity_id":0,`

func TestIndependentProducerNoEntity(t *testing.T) {
	r, err := Read(strings.NewReader(reportBody(`{"status":"Fail"}`)), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Categories[0].Tests[0].Results[0].Entity != nil {
		t.Fatal("invented entity")
	}
}
func TestIndependentNumericAndAvailability(t *testing.T) {
	for _, n := range []string{"0", "44", "4294967295"} {
		raw := reportBody(`{` + gpuFields + `"status":"Fail","warnings":[{"error_id":` + n + `,"error_category":1,"error_severity":2,"warning":"message44"}]}`)
		r, err := Read(strings.NewReader(raw), Limits{})
		if err != nil {
			t.Fatal(err)
		}
		v := r.Categories[0].Tests[0].Results[0]
		var want uint32
		if err = json.Unmarshal([]byte(n), &want); err != nil {
			t.Fatal(err)
		}
		if uint32(v.Warnings[0].Code) != want || !v.WarningsAvailable || v.InfoAvailable {
			t.Fatalf("numeric/availability %+v", v)
		}
	}
	for _, n := range []string{"null", "-1", "true", "1.5", "4294967296", `"44"`} {
		raw := reportBody(`{` + gpuFields + `"status":"Fail","warnings":[{"error_id":` + n + `,"error_category":1,"error_severity":2,"warning":"message"}]}`)
		if _, err := Read(strings.NewReader(raw), Limits{}); err == nil {
			t.Fatalf("accepted %s", n)
		}
	}
	for _, fields := range []string{``, `,"warnings":[],"info":[]`} {
		r, err := Read(strings.NewReader(reportBody(`{`+gpuFields+`"status":"Pass"`+fields+`}`)), Limits{})
		if err != nil {
			t.Fatal(err)
		}
		v := r.Categories[0].Tests[0].Results[0]
		want := fields != ""
		if v.WarningsAvailable != want || v.InfoAvailable != want || len(v.Warnings) != 0 || len(v.Info) != 0 {
			t.Fatal("availability fabricated")
		}
	}
}
func TestIndependentVersionStatusAndBounds(t *testing.T) {
	good := reportBody(`{` + gpuFields + `"status":"Fail"}`)
	for _, version := range []string{"", "4.5.0", "4.6.1.0", "future"} {
		_, err := Read(strings.NewReader(strings.Replace(good, "4.6.1", version, 1)), Limits{})
		if err == nil {
			t.Fatal("version accepted")
		}
		if version != "" {
			var unsupported *UnsupportedError
			if !errors.As(err, &unsupported) || unsupported.Version != version {
				t.Fatalf("typed version %v", err)
			}
		}
	}
	for _, status := range []string{"Pass", "Fail", "Skip"} {
		if _, err := Read(strings.NewReader(strings.Replace(good, "Fail", status, 1)), Limits{}); err != nil {
			t.Fatal(err)
		}
	}
	for _, status := range []string{"Warn", "Not Run", "", "pass"} {
		if _, err := Read(strings.NewReader(strings.Replace(good, "Fail", status, 1)), Limits{}); err == nil {
			t.Fatal("status invented")
		}
	}
	for _, l := range []Limits{{Bytes: len(good) - 1}, {Records: 2}} {
		if _, err := Read(strings.NewReader(good), l); err == nil {
			t.Fatal("limit omitted")
		}
	}
	if _, err := Read(strings.NewReader(good), Limits{Bytes: len(good), Records: 3}); err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{good + `{}`, strings.Replace(good, `"version":"4.6.1"`, `"version":"4.6.1","vers\u0069on":"4.6.1"`, 1), strings.Replace(good, `"entity_id":0,`, ``, 1), strings.Replace(good, `"DCGM Diagnostic":{`, `"DCGM Diagnostic":{"runtime_error":"failure",`, 1)} {
		if _, err := Read(strings.NewReader(raw), Limits{}); err == nil {
			t.Fatal("invalid profile accepted")
		}
	}
}
func TestIndependentUnknownRaw(t *testing.T) {
	raw := reportBody(`{` + gpuFields + `"status":"Fail","future":{"x":[1,null,true]}}`)
	raw = strings.Replace(raw, `"metadata":`, `"future_root":[2],"metadata":`, 1)
	r, err := Read(strings.NewReader(raw), Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(r.Raw, []byte(raw)) || string(r.Unknown["future_root"]) != "[2]" || string(r.Categories[0].Tests[0].Results[0].Unknown["future"]) != `{"x":[1,null,true]}` {
		t.Fatal("lost unknown/raw")
	}
}
func FuzzRead(f *testing.F) {
	f.Add(reportBody(`{` + gpuFields + `"status":"Pass"}`))
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 4096 {
			return
		}
		Read(strings.NewReader(s), Limits{Bytes: 4096, Records: 128})
	})
}

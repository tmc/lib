package cudaevidence_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func TestEvidenceStandardJSONRoundTrip(t *testing.T) {
	for _, class := range []cudaevidence.Class{cudaevidence.Measured, cudaevidence.Derived, cudaevidence.Inferred, cudaevidence.Modeled} {
		v, err := cudaevidence.New(record(class))
		if err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(struct {
			Value cudaevidence.Value `json:"value"`
		}{v})
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Value cudaevidence.Value `json:"value"`
		}
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(v.Record(), got.Value.Record()) {
			t.Fatalf("class %q lost: %s", class, data)
		}
		before := got.Value.Record()
		if err := json.Unmarshal([]byte(`{"class":"measured","number":null}`), &got.Value); err == nil {
			t.Fatal("accepted invalid replacement")
		}
		if !reflect.DeepEqual(before, got.Value.Record()) {
			t.Fatal("failed decoding changed destination")
		}
	}
}

func TestEvidenceDuplicateJSONFields(t *testing.T) {
	v, err := cudaevidence.New(record(cudaevidence.Inferred))
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		strings.Replace(string(data), `"class":"inferred"`, `"class":"inferred","Class":"measured"`, 1),
		strings.Replace(string(data), `"source":`, `"\u017fource":`, 1),
		strings.Replace(string(data), `"class":"inferred"`, `"class":"inferred","class":"measured"`, 1),
		strings.Replace(string(data), `"number":0`, `"number":4,"number":0`, 1),
		strings.Replace(string(data), `"entry":"kernel"`, `"entry":"other","entry":"kernel"`, 1),
	} {
		if _, err := cudaevidence.Decode([]byte(bad)); err == nil {
			t.Fatalf("accepted duplicate fields: %s", bad)
		}
	}
}

func TestEvidenceAliasUnmarshalLeavesValueUnchanged(t *testing.T) {
	v, err := cudaevidence.New(record(cudaevidence.Inferred))
	if err != nil {
		t.Fatal(err)
	}
	before := v.Record()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), `"class":"inferred"`, `"class":"inferred","Class":"measured"`, 1))
	if err := json.Unmarshal(data, &v); err == nil {
		t.Fatal("accepted class alias")
	}
	if !reflect.DeepEqual(before, v.Record()) {
		t.Fatal("bad decode changed value")
	}
	if v.CheckClaim(cudaevidence.Measured, identity()) == nil {
		t.Fatal("promoted inference")
	}
}

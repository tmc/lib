package cudaevidence_test

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func FuzzEvidenceJSON(f *testing.F) {
	f.Add([]byte(`{"class":"unread","number":null,"reason":"no provider"}`))
	f.Add([]byte(`{"class":"inferred","Class":"measured"}`))
	f.Add([]byte(`null`))
	f.Fuzz(func(t *testing.T, data []byte) {
		value, err := cudaevidence.Decode(data)
		if err != nil {
			return
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		round, err := cudaevidence.Decode(encoded)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(value.Record(), round.Record()) {
			t.Fatal("JSON changed evidence")
		}
	})
}

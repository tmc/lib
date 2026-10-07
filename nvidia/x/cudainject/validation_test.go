package cudainject_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudainject"
)

func TestValidationCorrelationPayloadRefusals(t *testing.T) {
	for _, payload := range []string{
		`{"domain":"driver","callback_id":211,"thread_id":17,"correlation":1,"Correlation":37}`,
		`{"domain":"driver","callback_id":211,"thread_id":17,"correlation":1,"correlation":37}`,
		`{"domain":"driver","callback_id":211,"correlation":37}`,
		`{"domain":"driver","callback_id":211,"thread_id":null,"correlation":37}`,
	} {
		records := []cudainject.Record{{Kind: cudainject.KindAPI, Payload: []byte(payload)}, {Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":true,"start":1,"end":2,"correlation":37}`)}}
		if _, err := cudainject.JoinCorrelations(records); err == nil {
			t.Fatalf("accepted ambiguous/missing payload %s", payload)
		}
	}
}

func TestValidationCorrelationKernelRefusals(t *testing.T) {
	for _, payload := range []string{
		`{"subtype":"kernel","name":"k","valid":true,"start":1,"end":2,"correlation":1,"Correlation":37}`,
		`{"subtype":"kernel","name":"k","valid":true,"end":2,"correlation":37}`,
		`{"subtype":"kernel","name":null,"valid":true,"start":1,"end":2,"correlation":37}`,
	} {
		if _, err := cudainject.JoinCorrelations([]cudainject.Record{{Kind: cudainject.KindActivity, Payload: []byte(payload)}}); err == nil {
			t.Fatalf("accepted invalid kernel %s", payload)
		}
	}
}

func FuzzValidationCorrelations(f *testing.F) {
	f.Add(`{"domain":"driver","callback_id":211,"thread_id":17,"correlation":37}`)
	f.Add(`{"domain":"driver","correlation":1,"Correlation":37}`)
	f.Fuzz(func(t *testing.T, payload string) {
		if len(payload) > 16384 {
			return
		}
		_, _ = cudainject.JoinCorrelations([]cudainject.Record{{Kind: cudainject.KindAPI, Payload: []byte(payload)}})
	})
}

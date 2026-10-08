package cudainject_test

import (
	"errors"
	"fmt"
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

func TestValidationCorrelationExternalRefusals(t *testing.T) {
	for _, payload := range []string{
		`{"external_kind":1,"external_id":17,"correlation":8,"Correlation":9}`,
		`{"external_kind":1,"external_id":17,"correlation":8,"external_id":18}`,
		`{"external_kind":null,"external_id":17,"correlation":8}`,
		`{"external_kind":1,"external_id":null,"correlation":8}`,
		`{"external_kind":1,"external_id":17,"correlation":null}`,
		`{"external_id":17,"correlation":8}`,
		`{"external_kind":1,"correlation":8}`,
		`{"external_kind":1,"external_id":17}`,
		`{"external_kind":1,"external_id":17,"correlation":8} {}`,
	} {
		records := []cudainject.Record{
			{Kind: cudainject.KindAPI, Payload: []byte(`{"domain":"driver","callback_id":211,"thread_id":17,"correlation":8}`)},
			{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":true,"start":1,"end":2,"correlation":8}`)},
			{Kind: cudainject.KindExternal, Payload: []byte(payload)},
		}
		if _, err := cudainject.JoinCorrelations(records); err == nil {
			t.Fatalf("accepted invalid external payload %s", payload)
		}
	}
}

func TestValidationCorrelationKernelLimits(t *testing.T) {
	if _, err := cudainject.JoinCorrelations([]cudainject.Record{
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":true,"start":10,"end":5,"correlation":1}`)},
	}); err == nil {
		t.Fatal("accepted end before start")
	}

	if _, err := cudainject.JoinCorrelations([]cudainject.Record{
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":false,"start":1,"end":2,"correlation":1}`)},
	}); err == nil {
		t.Fatal("accepted valid:false kernel")
	}

	if _, err := cudainject.JoinCorrelations([]cudainject.Record{
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"kernel","name":"k","valid":true,"start":1,"end":2,"correlation":0}`)},
	}); err == nil {
		t.Fatal("accepted zero correlation kernel")
	}

	joins, err := cudainject.JoinCorrelations([]cudainject.Record{
		{Kind: cudainject.KindActivity, Payload: []byte(`{"subtype":"memcpy"}`)},
	})
	if err != nil || len(joins.Matched) != 0 {
		t.Fatalf("memcpy activity handling failed: %v", err)
	}
}

func TestValidationCorrelationDomainRefusals(t *testing.T) {
	for _, domain := range []string{"cuda", "nvtx", "", "unknown"} {
		payload := fmt.Sprintf(`{"domain":%q,"callback_id":1,"thread_id":1,"correlation":1}`, domain)
		if _, err := cudainject.JoinCorrelations([]cudainject.Record{{Kind: cudainject.KindAPI, Payload: []byte(payload)}}); err == nil {
			t.Fatalf("accepted unsupported API domain %q", domain)
		}
	}
}

func TestValidationCorrelationLimits(t *testing.T) {
	tooMany := make([]cudainject.Record, 4097)
	for i := range tooMany {
		tooMany[i] = cudainject.Record{Kind: cudainject.KindAPI, Payload: []byte(`{"domain":"driver","callback_id":1,"thread_id":1,"correlation":0}`)}
	}
	if _, err := cudainject.JoinCorrelations(tooMany); !errors.Is(err, cudainject.ErrRecordLimit) {
		t.Fatalf("record limit: %v", err)
	}

	huge := []cudainject.Record{
		{Kind: cudainject.KindAPI, Payload: make([]byte, (16<<20)+1)},
	}
	if _, err := cudainject.JoinCorrelations(huge); !errors.Is(err, cudainject.ErrOversize) {
		t.Fatalf("payload oversize: %v", err)
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

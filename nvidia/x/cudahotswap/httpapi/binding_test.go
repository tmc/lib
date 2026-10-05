package httpapi

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestPointerIntentCannotBecomeInteger(t *testing.T) {
	f := new(fakeService)
	h, err := New(f, Resources{Arguments: map[string][]string{"kernel": {"buffer"}}, Streams: map[string]*cuda.Stream{"main": {}}})
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/launch", strings.NewReader(`{"name":"kernel","label":"caller","stream":"main","geometry":{},"args":[{"type":"u64","value":4096}]}`)))
	if w.Code != 400 {
		t.Fatalf("integer became application pointer: status %d %s", w.Code, w.Body.String())
	}
	for _, call := range f.calls {
		if call == "Launch" {
			t.Fatal("invalid pointer reached service")
		}
	}
}

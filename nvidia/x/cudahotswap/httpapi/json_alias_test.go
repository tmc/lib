package httpapi

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestJSONFieldAliases(t *testing.T) {
	for _, test := range []struct {
		name      string
		body      string
		wantError bool
	}{
		{"single alias", `{"NAME":"kernel"}`, false},
		{"separate objects", `{"source":{"Code":"a"},"name":"kernel"}`, false},
		{"name", `{"name":"a","NAME":"b"}`, true},
		{"label", `{"label":"a","Label":"b"}`, true},
		{"escaped alias", `{"name":"a","\u004eame":"b"}`, true},
		{"nested source", `{"source":{"code":"a","Code":"b"}}`, true},
		{"unicode alias", `{"source":{},"ſource":{}}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var request Request
			response := httptest.NewRecorder()
			err := decodeRequest(response, httptest.NewRequest("POST", "/v1/push", strings.NewReader(test.body)), &request)
			if (err != nil) != test.wantError {
				t.Fatalf("decode error = %v, want error %t", err, test.wantError)
			}
		})
	}
}

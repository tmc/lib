package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIndependentNegotiationPreferences(t *testing.T) {
	for _, accept := range []string{
		"application/openmetrics-text;q=0,*/*;q=1",
		"application/openmetrics-text;q=0.2,text/plain;q=0.8",
	} {
		t.Run(accept, func(t *testing.T) {
			a, err := NewAgent(independentCfg(), "", nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest("GET", "/metrics", nil)
			req.Header.Set("Accept", accept)
			rec := httptest.NewRecorder()
			a.handleMetrics(rec, req)
			if rec.Code != 200 || strings.Contains(rec.Header().Get("Content-Type"), "openmetrics") || strings.HasSuffix(rec.Body.String(), "# EOF\n") {
				t.Errorf("request requires legacy preference, got code=%d type=%q EOF=%v", rec.Code, rec.Header().Get("Content-Type"), strings.HasSuffix(rec.Body.String(), "# EOF\n"))
			}
		})
	}
}

func TestIndependentMultipleAcceptHeaders(t *testing.T) {
	a, err := NewAgent(independentCfg(), "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("GET", "/metrics", nil)
	req.Header.Add("Accept", "application/openmetrics-text;q=0.2")
	req.Header.Add("Accept", "text/plain;q=0.8")
	rec := httptest.NewRecorder()
	a.handleMetrics(rec, req)
	if rec.Code != 200 || strings.Contains(rec.Header().Get("Content-Type"), "openmetrics") || strings.HasSuffix(rec.Body.String(), "# EOF\n") {
		t.Errorf("request requires legacy preference, got code=%d type=%q EOF=%v", rec.Code, rec.Header().Get("Content-Type"), strings.HasSuffix(rec.Body.String(), "# EOF\n"))
	}
}

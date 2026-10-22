package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4server"
)

func TestReplayReportsTokenIdentityUnread(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("request = %s %s", r.Method, r.URL)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"candidate","answer":"ok"}`))
	}))
	defer srv.Close()
	base, err := parseBaseURL(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	corpus := dsv4server.Corpus{Cases: map[string][]dsv4server.CorpusCase{
		"G1": {{Step: "chat", Method: "POST", Endpoint: "/v1/chat/completions", Status: "HTTP/1.1 200 OK", Request: []byte(`{"model":"x"}`), Response: []byte(`{"id":"oracle","answer":"ok"}`), Tokens: []int{42}}},
	}}
	r := replay(corpus, base, srv.Client())
	if r.Status != "UNREAD" || r.Matched != 1 || r.Mismatched != 0 || r.TokensUnread != 1 {
		t.Fatalf("receipt = %+v, want wire match with unread tokens", r)
	}
	if len(r.Cases) != 1 || r.Cases[0].TokenStatus != "UNREAD" {
		t.Fatalf("case receipt = %+v, want UNREAD token status", r.Cases)
	}
}

func TestReplayFailsOnWireMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"answer":"wrong"}`))
	}))
	defer srv.Close()
	base, err := parseBaseURL(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	corpus := dsv4server.Corpus{Cases: map[string][]dsv4server.CorpusCase{
		"G1": {{Step: "chat", Method: "POST", Endpoint: "/v1/chat/completions", Status: "HTTP/1.1 200 OK", Request: []byte(`{"model":"x"}`), Response: []byte(`{"answer":"ok"}`)}},
	}}
	r := replay(corpus, base, srv.Client())
	if r.Status != "FAIL" || r.Mismatched != 1 || !strings.Contains(r.Cases[0].Error, "response differs") {
		t.Fatalf("receipt = %+v, want fail-closed wire mismatch", r)
	}
}

func TestParseBaseURLRejectsPath(t *testing.T) {
	if _, err := parseBaseURL("http://127.0.0.1:8080/prefix"); err == nil {
		t.Fatal("base URL with path unexpectedly accepted")
	}
}

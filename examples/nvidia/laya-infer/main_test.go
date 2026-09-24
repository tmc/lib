package main

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"testing"
)

func TestServeArgs(t *testing.T) {
	def := "127.0.0.1:" + strconv.Itoa(defaultPort)
	tests := []struct {
		name      string
		args      []string
		addr, key string
		wantErr   bool
	}{
		{name: "defaults", addr: def},
		{name: "port", args: []string{"8080"}, addr: "127.0.0.1:8080"},
		{name: "bind", args: []string{"--bind", "0.0.0.0", "9000"}, addr: "0.0.0.0:9000"},
		{name: "key", args: []string{"--api-key", "s3cret"}, addr: def, key: "s3cret"},
		{name: "all", args: []string{"9000", "--api-key", "k", "--bind", "::1"}, addr: "[::1]:9000", key: "k"},
		{name: "empty key", args: []string{"--api-key", ""}, wantErr: true},
		{name: "key without value", args: []string{"--api-key"}, wantErr: true},
		{name: "bad port", args: []string{"70000"}, wantErr: true},
		{name: "unknown", args: []string{"--nope"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			addr, key, err := serveArgs(tt.args)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("serveArgs(%q) = %q, %q, want error", tt.args, addr, key)
				}
				return
			}
			if err != nil {
				t.Fatalf("serveArgs(%q): %v", tt.args, err)
			}
			if addr != tt.addr || key != tt.key {
				t.Errorf("serveArgs(%q) = %q, %q, want %q, %q", tt.args, addr, key, tt.addr, tt.key)
			}
		})
	}
}

func TestRequireKey(t *testing.T) {
	ok := func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }
	tests := []struct {
		name string
		key  string
		auth string // Authorization header; empty sends none
		want int
	}{
		{name: "no key configured", want: http.StatusOK},
		{name: "no key configured, header sent", auth: "Bearer x", want: http.StatusOK},
		{name: "right key", key: "s3cret", auth: "Bearer s3cret", want: http.StatusOK},
		{name: "missing header", key: "s3cret", want: http.StatusUnauthorized},
		{name: "wrong key", key: "s3cret", auth: "Bearer s3creT", want: http.StatusUnauthorized},
		{name: "prefix of key", key: "s3cret", auth: "Bearer s3c", want: http.StatusUnauthorized},
		{name: "no scheme", key: "s3cret", auth: "s3cret", want: http.StatusUnauthorized},
		{name: "basic scheme", key: "s3cret", auth: "Basic s3cret", want: http.StatusUnauthorized},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/predict", nil)
			if tt.auth != "" {
				r.Header.Set("Authorization", tt.auth)
			}
			w := httptest.NewRecorder()
			requireKey(tt.key, ok)(w, r)
			if w.Code != tt.want {
				t.Errorf("status = %d, want %d", w.Code, tt.want)
			}
		})
	}
}

func TestParsePrecision(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		rest    []string
		prec    precision
		wantErr bool
	}{
		{name: "default", args: []string{"dir", "req.json"}, rest: []string{"dir", "req.json"}, prec: fast},
		{name: "exact", args: []string{"dir", "req.json", "--exact"}, rest: []string{"dir", "req.json"}, prec: exact},
		{name: "half first", args: []string{"-half", "dir", "req.json"}, rest: []string{"dir", "req.json"}, prec: half},
		{name: "half twice", args: []string{"--half", "dir", "--half"}, rest: []string{"dir"}, prec: half},
		{name: "both", args: []string{"dir", "--exact", "--half"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rest, prec, err := parsePrecision(tt.args)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parsePrecision(%q) = %q, %v, want error", tt.args, rest, prec)
				}
				return
			}
			if err != nil {
				t.Fatalf("parsePrecision(%q): %v", tt.args, err)
			}
			if !slices.Equal(rest, tt.rest) || prec != tt.prec {
				t.Errorf("parsePrecision(%q) = %q, %v, want %q, %v", tt.args, rest, prec, tt.rest, tt.prec)
			}
		})
	}
}

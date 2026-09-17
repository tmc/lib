package nvrtcx

import (
	"testing"
	"testing/fstest"
)

func TestWithHeadersFSDefaultsAndPatterns(t *testing.T) {
	fsys := fstest.MapFS{
		"include/z.cuh": &fstest.MapFile{Data: []byte("z")},
		"include/a.cuh": &fstest.MapFile{Data: []byte("a")},
		"README":        &fstest.MapFile{Data: []byte("ignored by pattern")},
	}

	var cfg options
	WithHeadersFS(fsys)(&cfg)
	if cfg.err != nil {
		t.Fatal(cfg.err)
	}
	if cfg.headers["include/a.cuh"] != "a" || cfg.headers["include/z.cuh"] != "z" || cfg.headers["README"] == "" {
		t.Fatalf("default headers = %#v", cfg.headers)
	}

	var selected options
	WithHeadersFS(fsys, "include/*.cuh", "include/a.cuh")(&selected)
	if selected.err != nil {
		t.Fatal(selected.err)
	}
	if len(selected.headers) != 2 || selected.headers["include/a.cuh"] != "a" {
		t.Fatalf("selected headers = %#v", selected.headers)
	}
}

func TestWithHeadersFSErrorsAndOverrides(t *testing.T) {
	var nilFS options
	WithHeadersFS(nil)(&nilFS)
	if nilFS.err == nil {
		t.Fatal("nil filesystem accepted")
	}

	fsys := fstest.MapFS{"x.cuh": &fstest.MapFile{Data: []byte("fs")}}
	var cfg options
	WithHeadersFS(fsys, "missing/*.cuh")(&cfg)
	if cfg.err == nil {
		t.Fatal("unmatched pattern accepted")
	}

	var override options
	WithHeadersFS(fsys)(&override)
	WithHeader("x.cuh", "explicit")(&override)
	if override.err != nil || override.headers["x.cuh"] != "explicit" {
		t.Fatalf("override headers = %#v, err = %v", override.headers, override.err)
	}
}

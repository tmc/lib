package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type collectorFailClose struct {
	path   string
	err    error
	called bool
	early  bool
}

func TestCollectorSuccessfulCleanupPublication(t *testing.T) {
	p, err := newProfileCollectors(false, true, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for layer := 0; layer < 43; layer++ {
		if err := p.observe(layer, 7, 1, []int32{0, 1, 2, 3, 4, 5}, []float32{1, .5, .25, .125, .0625, .03125}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	dir := t.TempDir()
	cfg := config{backend: "metal", collectors: collectorOptions{expertProfile: filepath.Join(dir, "profile.json"), hotlist: filepath.Join(dir, "hotlist")}}
	closer := &collectorFailClose{path: cfg.collectors.expertProfile}
	if err := finishProfileOutput(nil, closer, cfg, p); err != nil {
		t.Fatal(err)
	}
	if !closer.called || closer.early {
		t.Fatal("publication preceded successful cleanup")
	}
	for _, path := range []string{cfg.collectors.expertProfile, cfg.collectors.hotlist} {
		data, err := os.ReadFile(path)
		if err != nil || len(data) == 0 {
			t.Fatalf("missing output: %s %v", path, err)
		}
	}
	// Compare both complete outputs to the same observed data's serializers.
	reference := config{backend: "metal", collectors: collectorOptions{expertProfile: filepath.Join(dir, "reference.json"), hotlist: filepath.Join(dir, "reference.hotlist")}}
	if err := writeExpertProfile(reference, p); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]string{{cfg.collectors.expertProfile, reference.collectors.expertProfile}, {cfg.collectors.hotlist, reference.collectors.hotlist}} {
		a, _ := os.ReadFile(pair[0])
		b, _ := os.ReadFile(pair[1])
		if !bytes.Equal(a, b) {
			t.Fatal("publication changed serialized values")
		}
	}
}

func TestCollectorNilOrdinaryCleanup(t *testing.T) {
	for _, failure := range []bool{false, true} {
		closer := &collectorFailClose{path: filepath.Join(t.TempDir(), "unused")}
		if failure {
			closer.err = errors.New("close failed")
		}
		err := finishProfileOutput(nil, closer, config{}, nil)
		if !closer.called {
			t.Fatal("ordinary cleanup skipped")
		}
		if !failure && err != nil {
			t.Fatal(err)
		}
		if failure {
			var owner *backendCleanupOwner
			if !errors.As(err, &owner) || owner.backend != closer {
				t.Fatal("ordinary cleanup owner lost")
			}
		}
	}
}

func (c *collectorFailClose) Close() error {
	c.called = true
	_, err := os.Stat(c.path)
	c.early = !os.IsNotExist(err)
	return c.err
}

func TestCollectorCleanupBeforeOutput(t *testing.T) {
	p, err := newProfileCollectors(false, true, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for layer := 0; layer < 43; layer++ {
		if err := p.observe(layer, 0, 1, []int32{0, 1, 2, 3, 4, 5}, []float32{1, 1, 1, 1, 1, 1}, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "profile.json")
	stop := errors.New("teardown failed")
	closer := &collectorFailClose{path: path, err: stop}
	err = finishProfileOutput(nil, closer, config{backend: "metal", collectors: collectorOptions{expertProfile: path}}, p)
	var owner *backendCleanupOwner
	if !errors.Is(err, stop) || !errors.As(err, &owner) || owner.backend != closer {
		t.Fatalf("cleanup owner lost: %v", err)
	}
	if !closer.called || closer.early {
		t.Fatal("publication preceded cleanup")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("teardown failure published output: %v", err)
	}
}

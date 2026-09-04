//go:build darwin && arm64

package main

import "testing"

func TestMetalTinyFixture(t *testing.T) {
	m := readFixture(t, "testdata/fixture.json")
	metal, err := newMetalTinyBackend(m.Payload.Seed)
	if err != nil {
		t.Fatal(err)
	}
	defer metal.Close()
	if name := metal.DeviceName(); name == "" {
		t.Fatal("Metal device has no name")
	} else {
		t.Logf("physical Metal device: %s", name)
	}

	tokens := append(append([]int(nil), m.Payload.Prompt...), m.Payload.Decode...)
	for i, token := range tokens {
		got, err := metal.Step(token)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		compareTrace(t, i, got, m.Payload.Traces[i])
	}
}

func TestMetalTinyLifecycle(t *testing.T) {
	for i := 0; i < 10; i++ {
		metal, err := newMetalTinyBackend(uint64(i + 1))
		if err != nil {
			t.Fatalf("open %d: %v", i, err)
		}
		if _, err := metal.Step(i % tinyVocab); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if err := metal.Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
	}
}

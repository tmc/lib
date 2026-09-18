package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"testing"
)

// The record kind numbers are a wire contract between two files that share no
// header: the producer's KIND_ defines in internal/cudainject/native/cupti_agent.c
// and the record* constants above in capture_controller.go. observeJournal's
// kind switch ends in a default that rejects the whole capture, so a number
// the two halves disagree about does not degrade a capture -- it fails every
// capture that contains one record of that kind.
//
// Nothing else in the tree catches that. observeJournal is exercised only
// against fixtures built in this package, and those fixtures write through the
// same record* constants they are then decoded by: one author, two copies. A
// constant here can be changed to any value at all and every test in this
// package still passes, because the fixture moves with it. The producer's own
// numbers are pinned from the other side -- the nativecontrol controls decode
// a real producer journal against their own literals -- but no test in the
// tree feeds a producer-written journal to observeJournal, so the agreement
// BETWEEN the two sets rests on review.
//
// This is a source-consistency check, not a behavioural one, and that is a
// real limitation: it reads the C file as text and would not notice a producer
// that wrote a kind other than the one its own define names. What it does
// convert is the failure that actually happens -- someone adds a kind on one
// side, or renumbers one, and nothing says so until a capture fails in the
// field -- into a test failure naming both spellings and both values.
//
// The stronger shape, running the C control harness and feeding its journal
// through observeJournal, was rejected: it would pull the whole nativecontrol
// build (a C toolchain, the stub CUPTI, three translation units) into this
// package's tests, and it would have to skip where no toolchain exists. A skip
// reports as ok, so the coverage would silently vanish on exactly the hosts
// that have no compiler -- while this check runs everywhere and cannot skip.
func TestRecordKindsMatchProducer(t *testing.T) {
	// The Go half. Keyed by the producer's spelling so a mismatch can name
	// both sides of the pair it broke.
	golang := map[string]uint16{
		"KIND_BOOTSTRAP":  recordBootstrap,
		"KIND_ACTIVITY":   recordActivity,
		"KIND_FINALIZED":  recordFinalized,
		"KIND_CAPABILITY": recordCapability,
		"KIND_FAILURE":    recordFailure,
		"KIND_EXTERNAL":   recordExternal,
		"KIND_API":        recordAPI,
		"KIND_HOST":       recordHost,
	}
	goSpelling := map[string]string{
		"KIND_BOOTSTRAP":  "recordBootstrap",
		"KIND_ACTIVITY":   "recordActivity",
		"KIND_FINALIZED":  "recordFinalized",
		"KIND_CAPABILITY": "recordCapability",
		"KIND_FAILURE":    "recordFailure",
		"KIND_EXTERNAL":   "recordExternal",
		"KIND_API":        "recordAPI",
		"KIND_HOST":       "recordHost",
	}

	agent := filepath.Join("..", "..", "..", "internal", "cudainject", "native", "cupti_agent.c")
	src, err := os.ReadFile(agent)
	if err != nil {
		t.Fatalf("read producer source: %v", err)
	}

	// KIND_COUNT is the width of the producer's per-kind counter arrays, not a
	// kind on the wire. It is excluded by name rather than by value so that
	// adding a real kind, which also moves KIND_COUNT, does not quietly start
	// matching it.
	const sizingDefine = "KIND_COUNT"
	define := regexp.MustCompile(`(?m)^#define (KIND_[A-Z_]+) ([0-9]+)$`)
	producer := map[string]uint16{}
	for _, m := range define.FindAllStringSubmatch(string(src), -1) {
		if m[1] == sizingDefine {
			continue
		}
		v, err := strconv.ParseUint(m[2], 10, 16)
		if err != nil {
			t.Fatalf("%s: %s is not a 16-bit kind: %v", agent, m[0], err)
		}
		if prior, dup := producer[m[1]]; dup {
			t.Fatalf("%s defines %s twice, as %d and %d", agent, m[1], prior, v)
		}
		producer[m[1]] = uint16(v)
	}
	if len(producer) == 0 {
		t.Fatalf("no KIND_ defines found in %s: this test asserts nothing", agent)
	}

	// Both directions. A kind the producer gained and this file has not is the
	// case that fails every capture containing one; a constant here the
	// producer does not define is a consumer admitting a kind nothing writes.
	for name, want := range producer {
		got, ok := golang[name]
		if !ok {
			t.Errorf("%s defines %s = %d and this package has no constant for it: observeJournal rejects the whole capture on an unknown kind", agent, name, want)
			continue
		}
		if got != want {
			t.Errorf("%s = %d but %s = %d: a journal written by this producer fails every capture", name, want, goSpelling[name], got)
		}
	}
	for name, got := range golang {
		if _, ok := producer[name]; !ok {
			t.Errorf("%s = %d names %s, which %s does not define", goSpelling[name], got, name, agent)
		}
	}

	// The numbers must also be distinct. Two kinds sharing a value is not a
	// mismatch either loop above would see -- both sides would agree -- and it
	// makes one kind decode as the other.
	byValue := map[uint16][]string{}
	for name, v := range producer {
		byValue[v] = append(byValue[v], name)
	}
	for v, names := range byValue {
		if len(names) > 1 {
			sort.Strings(names)
			t.Errorf("%s: kind %d is defined by %v; records of one decode as the other", agent, v, names)
		}
	}

	if t.Failed() {
		pairs := make([]string, 0, len(producer))
		for name, v := range producer {
			pairs = append(pairs, fmt.Sprintf("%s=%d", name, v))
		}
		sort.Strings(pairs)
		t.Logf("producer defines: %v", pairs)
	}
}

package main

import (
	"bytes"
	"io"
	"math"
	"os"
	"strings"
	"testing"
)

// Independently computed from struct.pack("<4f", 1, -0.0, 0, -2.5).
const checkFixtureHash = "0d532fef5715cec6771edfc4011fecbb270992dd0fc8f0a120837a0c2397f541"

func checkFixture() []float32 { return []float32{1, math.Float32frombits(1 << 31), 0, -2.5} }

func TestCheckLogits(t *testing.T) {
	for _, tt := range []struct {
		name    string
		logits  []float32
		vocab   int
		want    string
		wantErr bool
	}{
		{"exact bits", checkFixture(), 4, checkFixtureHash, false},
		{"unset", nil, 0, "", false},
		{"wrong digest", checkFixture(), 4, strings.Repeat("0", 64), true},
		{"zero sign changed", []float32{1, 0, 0, -2.5}, 4, checkFixtureHash, true},
		{"one bit changed", []float32{math.Float32frombits(math.Float32bits(1) + 1), math.Float32frombits(1 << 31), 0, -2.5}, 4, checkFixtureHash, true},
		{"short vector", checkFixture()[:3], 4, checkFixtureHash, true},
		{"NaN", []float32{float32(math.NaN())}, 1, checkFixtureHash, true},
		{"infinity", []float32{float32(math.Inf(1))}, 1, checkFixtureHash, true},
		{"empty", nil, 0, checkFixtureHash, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var log bytes.Buffer
			err := checkLogits("prefill", tt.logits, tt.vocab, tt.want, &log)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error %v", err, tt.wantErr)
			}
			if tt.name == "exact bits" && !strings.Contains(log.String(), checkFixtureHash) {
				t.Fatal("measured hash not reported")
			}
			if tt.name == "wrong digest" && (!strings.Contains(log.String(), checkFixtureHash) || !strings.Contains(err.Error(), checkFixtureHash)) {
				t.Fatal("mismatch lost measured hash")
			}
			if tt.name == "unset" && log.Len() != 0 {
				t.Fatal("disabled check produced output")
			}
		})
	}
}

func TestCheckFinalLogitsCount(t *testing.T) {
	cfg := config{tokens: 2, expectFinalLogits: checkFixtureHash}
	for _, count := range []int{0, 1, 2, 3} {
		err := cfg.checkFinalLogits(checkFixture(), count, 4, io.Discard)
		if (err != nil) != (count != 2) {
			t.Fatalf("count %d: %v", count, err)
		}
	}
	cfg.expectFinalLogits = ""
	if err := cfg.checkFinalLogits(nil, 0, 0, io.Discard); err != nil {
		t.Fatal(err)
	}
}

func TestLogitsCheckFlags(t *testing.T) {
	for _, name := range []string{"expect-prefill-logits", "expect-final-logits"} {
		for _, value := range []string{"", "abc", strings.Repeat("z", 64), strings.ToUpper(checkFixtureHash)} {
			cfg, err := parseConfig([]string{"-model", "m", "-backend", "cpu", "-prompt", "x", "-" + name, value}, strings.NewReader(""), io.Discard)
			valid := value == strings.ToUpper(checkFixtureHash)
			if (err == nil) != valid {
				t.Fatalf("%s=%q: %v", name, value, err)
			}
			if valid && (!cfg.checkingLogits() || cfg.expectPrefillLogits != checkFixtureHash && cfg.expectFinalLogits != checkFixtureHash) {
				t.Fatalf("expectation not normalized: %+v", cfg)
			}
		}
	}
}

func TestLogLogitsWorkload(t *testing.T) {
	data, err := os.ReadFile("../../scripts/testdata/oracle-workload-tokens2048.json")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := parsePromptTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config{backend: "cuda", model: "pinned.gguf", context: 4096, tokens: 2, expectPrefillLogits: checkFixtureHash}
	var log bytes.Buffer
	cfg.logLogitsWorkload(ids, &log)
	for _, want := range []string{"backend=cuda", `model="pinned.gguf"`, "context=4096", "prefill_tokens=2048", "requested_tokens=2", oracleWorkloadTokenHash} {
		if !strings.Contains(log.String(), want) {
			t.Fatalf("log missing %q: %s", want, log.String())
		}
	}
	log.Reset()
	cfg.expectPrefillLogits = ""
	cfg.logLogitsWorkload(ids, &log)
	if log.Len() != 0 {
		t.Fatal("disabled checking logged workload")
	}
}

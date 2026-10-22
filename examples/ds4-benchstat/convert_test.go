package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseCSV(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want row
		err  string
	}{{
		name: "current nine column form",
		in: "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n" +
			"2048,2048,310.54,128,28.59,34.049,127,28.61,0\n",
		want: row{ctxTokens: 2048, prefillTokens: 2048, prefillTPS: 310.54, genTokens: 128, genTPS: 28.59,
			genFirstMS: 34.049, genSteadyTokens: 127, genSteadyTPS: 28.61, hasFirst: true, hasSteady: true},
	}, {
		name: "published six column form",
		in: "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n" +
			"4096,2048,303.46,128,26.52,80373132\n",
		want: row{ctxTokens: 4096, prefillTokens: 2048, prefillTPS: 303.46, genTokens: 128, genTPS: 26.52,
			kvcacheBytes: 80373132},
	}, {
		name: "byte order mark before the first header",
		in: "\ufeffctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n" +
			"2048,2048,343.76,128,26.76,52184460\n",
		want: row{ctxTokens: 2048, prefillTokens: 2048, prefillTPS: 343.76, genTokens: 128, genTPS: 26.76,
			kvcacheBytes: 52184460},
	}, {
		name: "columns in an unexpected order",
		in: "gen_tps,ctx_tokens,gen_tokens,prefill_tps,prefill_tokens\n" +
			"9.5,512,64,100.25,512\n",
		want: row{ctxTokens: 512, prefillTokens: 512, prefillTPS: 100.25, genTokens: 64, genTPS: 9.5},
	}, {
		name: "no ctx_tokens column",
		in:   "a,b\n1,2\n",
		err:  "no ctx_tokens column",
	}, {
		name: "missing required column",
		in:   "ctx_tokens,prefill_tokens\n2048,2048\n",
		err:  "missing prefill_tps",
	}, {
		name: "unparseable rate",
		in:   "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps\n2048,2048,fast,128,1\n",
		err:  "prefill_tps",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseCSV(strings.NewReader(tt.in))
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("parseCSV error = %v, want one containing %q", err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 {
				t.Fatalf("got %d rows, want 1", len(got))
			}
			if got[0] != tt.want {
				t.Errorf("got  %+v\nwant %+v", got[0], tt.want)
			}
		})
	}
}

func TestParseReceipt(t *testing.T) {
	qualified := `{"result":"QUALIFIED_SINGLE_REP","backend":"metal","host":"m4max.local M4 Max",
	 "oracle":"b0a147a","model_sha256":"aa","ds4_c_sha256":"bb","binary_sha256":"cc",
	 "csv":{"ctx_tokens":"2048","prefill_tokens":"2048","prefill_tps":"310.54",
	        "gen_tokens":"128","gen_tps":"28.59","gen_first_ms":"34.049",
	        "gen_steady_tokens":"127","gen_steady_tps":"28.61","kvcache_bytes":"0"}}`

	cfg, rows, err := parseReceipt([]byte(qualified))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].prefillTPS != 310.54 {
		t.Fatalf("rows = %+v", rows)
	}
	want := []config{{"pkg", "ds4.c"}, {"result", "QUALIFIED_SINGLE_REP"}, {"backend", "metal"},
		{"host", "m4max.local M4 Max"}, {"oracle", "b0a147a"},
		{"model_sha256", "aa"}, {"ds4_c_sha256", "bb"}, {"binary_sha256", "cc"}}
	if len(cfg) != len(want) {
		t.Fatalf("config = %v, want %v", cfg, want)
	}
	for i := range want {
		if cfg[i] != want[i] {
			t.Errorf("config[%d] = %v, want %v", i, cfg[i], want[i])
		}
	}

	// A run that failed its after-qualification must not become a rate.
	disqualified := strings.Replace(qualified, "QUALIFIED_SINGLE_REP", "DISQUALIFIED", 1)
	if _, _, err := parseReceipt([]byte(disqualified)); err == nil {
		t.Fatal("parseReceipt accepted a DISQUALIFIED receipt")
	} else if !strings.Contains(err.Error(), "not qualified") {
		t.Errorf("error = %v, want one saying the receipt is not qualified", err)
	}

	// "DISQUALIFIED" must not be admitted by a loose prefix match either.
	for _, result := range []string{"DISQUALIFIED", "UNQUALIFIED_DIAGNOSTIC", "ABORTED", "SKIPPED", "FAILED"} {
		bad := strings.Replace(qualified, "QUALIFIED_SINGLE_REP", result, 1)
		if _, _, err := parseReceipt([]byte(bad)); err == nil {
			t.Errorf("parseReceipt accepted result %q", result)
		}
	}
}

func TestWrite(t *testing.T) {
	tests := []struct {
		name    string
		in      row
		want    []string
		absent  []string
		skipped int
	}{{
		name: "every phase reported",
		in: row{ctxTokens: 2048, prefillTokens: 2048, prefillTPS: 400, genTokens: 128, genTPS: 25,
			genFirstMS: 40, genSteadyTokens: 127, genSteadyTPS: 20, hasFirst: true, hasSteady: true,
			kvcacheBytes: 52184460},
		want: []string{
			"BenchmarkDS4Prefill/ctx=2048 2048 0.0025 sec/op 400 tok/s",
			"BenchmarkDS4Generate/ctx=2048 128 0.04 sec/op 25 tok/s 0.04 sec/first-token",
			"BenchmarkDS4GenerateSteady/ctx=2048 127 0.05 sec/op 20 tok/s",
			"BenchmarkDS4KVCache/ctx=2048 1 52184460 B/op",
		},
	}, {
		name: "unreported kv cache emits no size",
		in: row{ctxTokens: 2048, prefillTokens: 2048, prefillTPS: 400, genTokens: 128, genTPS: 25,
			kvcacheBytes: 0},
		absent:  []string{"BenchmarkDS4KVCache", "B/op\n"},
		skipped: 1,
	}, {
		name:    "unmeasured phases emit no rate",
		in:      row{ctxTokens: 2048, prefillTokens: 2048, prefillTPS: 0, genTokens: 128, genTPS: 0},
		absent:  []string{"BenchmarkDS4Prefill", "BenchmarkDS4Generate"},
		skipped: 3,
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf strings.Builder
			_, skipped, err := write(&buf, []config{{"pkg", "ds4.c"}}, []row{tt.in})
			if err != nil {
				t.Fatal(err)
			}
			got := buf.String()
			for _, line := range tt.want {
				if !strings.Contains(got, line+"\n") {
					t.Errorf("output missing line:\n%s\ngot:\n%s", line, got)
				}
			}
			for _, s := range tt.absent {
				if strings.Contains(got, s) {
					t.Errorf("output contains %q, want it omitted:\n%s", s, got)
				}
			}
			if skipped != tt.skipped {
				t.Errorf("skipped = %d, want %d", skipped, tt.skipped)
			}
			if !strings.Contains(got, "Unit tok/s better=higher\n") {
				t.Errorf("output does not declare tok/s as higher-is-better:\n%s", got)
			}
		})
	}
}

func TestWriteSortsByContext(t *testing.T) {
	rows := []row{
		{ctxTokens: 4096, prefillTokens: 1, prefillTPS: 1, genTokens: 1, genTPS: 1},
		{ctxTokens: 2048, prefillTokens: 1, prefillTPS: 1, genTokens: 1, genTPS: 1},
	}
	var buf strings.Builder
	if _, _, err := write(&buf, nil, rows); err != nil {
		t.Fatal(err)
	}
	got := buf.String()
	if i, j := strings.Index(got, "ctx=2048"), strings.Index(got, "ctx=4096"); i < 0 || j < 0 || i > j {
		t.Errorf("ctx=2048 should precede ctx=4096:\n%s", got)
	}
	// write must not reorder its caller's slice.
	if rows[0].ctxTokens != 4096 {
		t.Errorf("write sorted the caller's slice in place")
	}
}

// TestReceiptFieldsAreProvenanceKeys pins every field this command reads out of a
// receipt to provenanceKeys, from which the benchstat -ignore hint is built.
// Reading a new receipt field without listing it there would emit a
// configuration line the hint does not fold away, and that key alone would split
// benchstat's tables so two files could not be compared.
func TestReceiptFieldsAreProvenanceKeys(t *testing.T) {
	known := make(map[string]bool, len(provenanceKeys))
	for _, k := range provenanceKeys {
		known[k] = true
	}
	rt := reflect.TypeFor[receipt]()
	if rt.NumField() == 0 {
		t.Fatal("receipt has no fields; the check below would pass vacuously")
	}
	want := 0
	for i := range rt.NumField() {
		tag, _, _ := strings.Cut(rt.Field(i).Tag.Get("json"), ",")
		if tag == "" || tag == "csv" {
			continue // csv carries the measurements, not the provenance
		}
		want++
		if !known[tag] {
			t.Errorf("receipt field %q is read but not in provenanceKeys, so %s omits it",
				tag, ignoreFlag())
		}
	}
	if want != len(provenanceKeys) {
		t.Errorf("provenanceKeys has %d keys for %d receipt fields", len(provenanceKeys), want)
	}
}

// TestEmittedConfigMatchesProvenanceKeys checks the other direction: a receipt
// with every field set writes exactly the keys the hint names.
func TestEmittedConfigMatchesProvenanceKeys(t *testing.T) {
	full := `{"result":"QUALIFIED","backend":"metal","host":"h","oracle":"o",
	  "model_sha256":"m","ds4_c_sha256":"d","binary_sha256":"b",
	  "csv":{"ctx_tokens":"1","prefill_tokens":"1","prefill_tps":"1","gen_tokens":"1","gen_tps":"1"}}`
	cfg, _, err := parseReceipt([]byte(full))
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, c := range cfg {
		if c.key != "pkg" {
			got = append(got, c.key)
		}
	}
	if strings.Join(got, ",") != strings.Join(provenanceKeys, ",") {
		t.Errorf("emitted config keys %v, want %v", got, provenanceKeys)
	}
}

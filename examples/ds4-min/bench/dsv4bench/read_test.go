package dsv4bench

import (
	"strings"
	"testing"
)

const testCSV = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,peak_rss_bytes\n2048,2048,800,8,18,50,7,19,1,2,3,4,5,6\n4096,2048,780,8,17,52,7,18,2,3,4,5,6,7\n"

func TestReadCSVAndValidateFrontier(t *testing.T) {
	rows, err := ReadCSV(strings.NewReader(testCSV))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[1].CtxTokens != 4096 {
		t.Fatalf("rows = %#v", rows)
	}
	if err := ValidateFrontier(rows); err != nil {
		t.Fatal(err)
	}
}

func TestReadCSVFailsClosed(t *testing.T) {
	cases := map[string]string{
		"missing header": "",
		"wrong header":   "ctx_tokens\n",
		"bad number":     strings.Replace(testCSV, "800", "oops", 1),
		"negative count": strings.Replace(testCSV, "2048,2048", "-1,2048", 1),
	}
	for name, input := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ReadCSV(strings.NewReader(input)); err == nil {
				t.Fatal("accepted invalid CSV")
			}
		})
	}
}

func TestValidateFrontierRejectsEmptyUnorderedAndDuplicate(t *testing.T) {
	rows, err := ReadCSV(strings.NewReader(testCSV))
	if err != nil {
		t.Fatal(err)
	}
	for name, candidate := range map[string][]Row{
		"empty":        nil,
		"unordered":    {rows[1], rows[0]},
		"duplicate":    {rows[0], rows[0]},
		"zero context": {{CtxTokens: 0}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateFrontier(candidate); err == nil {
				t.Fatal("accepted invalid frontier")
			}
		})
	}
}

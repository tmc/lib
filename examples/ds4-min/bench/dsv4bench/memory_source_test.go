package dsv4bench

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestMemoryPinnedUpstreamSchema(t *testing.T) {
	checkMemorySourceRules(t, []byte(memorySourceFixture))
}

// This narrow fixture retains literal rules from ds4_bench.c at upstream
// 11e8cad4473716a41baf4d62e92ed9764a00fce6. It is not a full-source seal.
const memorySourceFixture = `
"ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,peak_rss_bytes\n"
return (uint64_t)ru.ru_maxrss;
return (uint64_t)ru.ru_maxrss * 1024ull;
(unsigned long long)(have_snapshot ? snap.len : 0),
`

// TestMemoryUpstreamSourceControl verifies the full external source only when
// explicitly requested. An ordinary offline test run needs no upstream checkout.
func TestMemoryUpstreamSourceControl(t *testing.T) {
	if os.Getenv("DS4_BENCH_REQUIRE_SOURCE") != "1" {
		return
	}
	path := os.Getenv("DS4_BENCH_SOURCE")
	if path == "" {
		t.Fatal("DS4_BENCH_SOURCE is required for the pinned memory/CSV control")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != "b69df1447beab1b2dcf298762af4b6730f6eb9af55db6b55b96b43511c74a423" {
		t.Fatal("upstream bench source seal differs")
	}
	checkMemorySourceRules(t, data)
}

func checkMemorySourceRules(t *testing.T, data []byte) {
	t.Helper()
	const literal = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,peak_rss_bytes"
	if !strings.Contains(string(data), `"`+literal+`\n"`) || strings.Join(header, ",") != literal {
		t.Fatal("owning upstream CSV literal differs")
	}
	for _, body := range []string{"return (uint64_t)ru.ru_maxrss;", "return (uint64_t)ru.ru_maxrss * 1024ull;", "have_snapshot ? snap.len : 0"} {
		if !strings.Contains(string(data), body) {
			t.Fatalf("missing literal source rule: %s", body)
		}
	}
}

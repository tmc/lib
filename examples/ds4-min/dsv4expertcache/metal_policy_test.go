package dsv4expertcache

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func TestMetalSlabTarget(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  uint64
	}{
		{"", 4 << 30}, {"0", 4 << 30}, {"  +8", 8 << 20},
		{"8suffix", 4 << 30}, {"8 ", 4 << 30}, {"-1", math.MaxUint64},
		{"18446744073709551615", math.MaxUint64},
		{"18446744073709551616", math.MaxUint64},
		{"-18446744073709551616", math.MaxUint64},
		{"-18446744073709551615", 1 << 20},
		{"18446744073709551616x", 4 << 30}, {"+", 4 << 30},
		{"17592186044415", 18446744073708503040},
		{"17592186044416", math.MaxUint64},
	} {
		t.Run(tt.input, func(t *testing.T) {
			t.Setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", tt.input)
			if got := metalSlabTarget(); got != tt.want {
				t.Fatalf("got %d want %d", got, tt.want)
			}
			if oracle := os.Getenv("DS4_TEST_METAL_SLAB_ORACLE"); oracle != "" {
				out, err := exec.Command(oracle).Output()
				if err != nil {
					t.Fatal(err)
				}
				want, err := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
				if err != nil || metalSlabTarget() != want {
					t.Fatalf("literal C got %q err=%v Go=%d", out, err, metalSlabTarget())
				}
			}
		})
	}
}

func TestMetalPrefillPolicy(t *testing.T) {
	for _, tt := range []struct {
		name                                         string
		tokens, experts, selected, gate, down, slots uint32
		quality, want                                bool
	}{
		{"flash-min", 2, 256, 6, 16, 10, 256, false, true},
		{"flash-max", 760, 256, 6, 16, 10, 256, false, true},
		{"flash-over", 761, 256, 6, 16, 10, 256, false, false},
		{"pro-max", 800, 384, 6, 16, 10, 384, false, true},
		{"pro-over", 801, 384, 6, 16, 10, 384, false, false},
		{"scalar", 1, 256, 6, 16, 10, 256, false, false},
		{"small-cache", 64, 256, 6, 16, 10, 255, false, false},
		{"quality", 64, 256, 6, 16, 10, 256, true, false},
		{"q4", 64, 256, 6, 12, 12, 256, false, false},
		{"slots", 64, 256, 5, 16, 10, 256, false, false},
		{"other-shape", 64, 128, 6, 16, 10, 128, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := MetalPrefillEnabled(tt.tokens, tt.experts, tt.selected, tt.gate, tt.down, tt.slots, tt.quality); got != tt.want {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
	t.Run("explicit", func(t *testing.T) {
		t.Setenv("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "")
		if !MetalPrefillEnabled(1024, 256, 6, 16, 10, 256, false) || MetalPrefillEnabled(1024, 256, 6, 16, 10, 255, false) {
			t.Fatal("explicit admission")
		}
	})
	t.Run("disabled-present", func(t *testing.T) {
		t.Setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE", "")
		if MetalPrefillEnabled(64, 256, 6, 16, 10, 256, false) {
			t.Fatal("disabled admission")
		}
	})
	t.Run("prefix-limit", func(t *testing.T) {
		t.Setenv("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX", "+10suffix")
		if !MetalPrefillEnabled(10, 256, 6, 16, 10, 256, false) || MetalPrefillEnabled(11, 256, 6, 16, 10, 256, false) {
			t.Fatal("strtol prefix")
		}
	})
}

func TestMetalCachePolicy(t *testing.T) {
	p, err := PlanMetalCache(1000, 0, 100, 10, 100, 80)
	if err != nil || p.TargetBytes != 800 || p.CacheBytes != 700 || p.Slots != 70 || p.EffectiveBytes != 700 {
		t.Fatal(p, err)
	}
	p, err = PlanMetalCache(1000, 500, 600, 10, 100, 80)
	if err != nil || p.TargetBytes != 500 || p.CacheBytes != 0 || p.Slots != 1 || p.EffectiveBytes != 10 {
		t.Fatal(p, err)
	}
	p, err = PlanMetalCache(math.MaxUint64, 0, 0, 1, 100, 80)
	if err != nil || p.TargetBytes != math.MaxUint64 || p.Slots != 100 {
		t.Fatal(p, err)
	}
	for _, tt := range []struct {
		bytes, expert   uint64
		experts, layers uint32
		cache, reserve  uint64
	}{
		{1000, 10, 20, 3, 600, 400}, {400, 10, 20, 3, 10, 390}, {10, 10, 20, 3, 10, 0}, {5, 10, 20, 3, 5, 0},
	} {
		cache, reserve, err := ReserveMetalPrefill(tt.bytes, tt.expert, tt.experts, tt.layers)
		if err != nil || cache != tt.cache || reserve != tt.reserve {
			t.Fatal(cache, reserve, err)
		}
	}
	t.Run("percent-override", func(t *testing.T) {
		t.Setenv("DS4_SSD_AUTO_CACHE_PCT", "+90")
		p, err := PlanMetalCache(1000, 0, 100, 10, 100, 80)
		if err != nil || p.TargetBytes != 900 {
			t.Fatal(p, err)
		}
	})
}

func TestCompactMetalRoutes(t *testing.T) {
	u, r, err := CompactMetalRoutes([]int32{7, 2, 7, 3, 2}, 8)
	if err != nil || !reflect.DeepEqual(u, []int32{7, 2, 3}) || !reflect.DeepEqual(r, []int32{0, 1, 0, 2, 1}) {
		t.Fatal(u, r, err)
	}
	if _, _, err := CompactMetalRoutes([]int32{-1}, 8); err == nil {
		t.Fatal("bad expert")
	}
}

func ExamplePlanMetalCache() {
	p, _ := PlanMetalCache(1000, 500, 100, 10, 100, 80)
	fmt.Println(p.Slots)
	// Output: 40
}

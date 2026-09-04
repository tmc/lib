// SPDX-License-Identifier: Apache-2.0

package metalattn

import "testing"

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig(8, 2, 64)
	if cfg.NumQHeads != 8 || cfg.NumKVHeads != 2 || cfg.HeadDim != 64 || cfg.Scale != 0.125 {
		t.Fatalf("DefaultConfig = %+v", cfg)
	}
}

//go:build linux

package main

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestShapeAttentionAdmission(t *testing.T) {
	for _, variant := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, _ := dsv4shape.For(variant)
		for _, fault := range []string{"closed", "family", "position", "beyond-context", "raw-row", "raw-count", "compressed-count"} {
			t.Run(string(rune('0'+variant))+"/"+fault, func(t *testing.T) {
				r := new(cudaFoundation)
				cache := &cudaShapeCache{r: r, plan: shapeCachePlan{layers: c.Layers, buffers: []shapeCacheBufferPlan{{layer: 2, name: "raw", elements: 512}, {layer: 2, name: "attention-cache", elements: 1024}}}, buffers: []cudamem.Slice[float32]{cacheTestSlice(t, 512, 1), cacheTestSlice(t, 1024, 2)}}
				state := &cudaShapeForward{r: r, cache: cache, plan: shapeForwardPlan{config: c, context: 128}, scratch: map[string]cudamem.Slice[float32]{}}
				req := shapeAttentionRequest{layer: 2}
				want := ""
				switch fault {
				case "closed":
					r.closing = true
				case "family":
					cache.plan.layers--
					want = "cache family mismatch"
				case "position":
					req.position = math.MaxUint32
					want = "invalid position or sinks"
				case "beyond-context":
					req.position = 128
					want = "invalid position or sinks"
				case "raw-row":
					cache.plan.buffers[0].elements = 513
					cache.buffers[0] = cacheTestSlice(t, 513, 1)
					want = "truncated raw cache row"
				case "raw-count":
					req.rawRows = 2
					want = "invalid raw cache bounds"
				case "compressed-count":
					req.compressedRows = 1
					want = "invalid compressed cache bounds"
				}
				err := encodeShapeAttention(state, 1, req, cacheTestSlice(t, c.Heads, 3), cudamem.Slice[float32]{})
				if fault == "closed" {
					if !errors.Is(err, dsv4.ErrClosed) {
						t.Fatalf("closed error = %v", err)
					}
				} else if err == nil || !strings.Contains(err.Error(), want) {
					t.Fatalf("ATTENTION_ADMISSION fault=%s error=%v want=%s", fault, err, want)
				}
			})
		}
	}
}

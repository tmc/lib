package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestCollectiveIntentPayloadValidation(t *testing.T) {
	maxInt := uint64(^uint(0) >> 1)
	for _, tt := range []struct {
		name, dtype string
		count       uint64
		wantError   string
	}{
		{"float32-limit", "f32", maxInt / 4, ""},
		{"float32-overflow", "f32", maxInt/4 + 1, "overflows"},
		{"int64-limit", "i64", maxInt / 8, ""},
		{"int64-overflow", "i64", maxInt/8 + 1, "overflows"},
		{"byte-limit", "u8", maxInt, ""},
		{"byte-overflow", "u8", maxInt + 1, "overflows"},
		{"unknown", "fp4", 4, "unsupported dtype"},
		{"alias", "FLOAT32", 4, ""},
		{"empty", "f32", 0, "count"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := egraph.CollectiveIntent{Communicator: "comm", Generation: 1, WorldSize: 2, Operation: "allreduce", DType: tt.dtype, Count: tt.count, EffectToken: 1}
			err := c.Validate()
			desc, descriptorError := c.OperationDescriptor()
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) || descriptorError == nil || egraph.CanRewrite(c, c) {
					t.Fatalf("invalid payload admitted: validation=%v descriptor=%v rewrite=%v", err, descriptorError, egraph.CanRewrite(c, c))
				}
				return
			}
			if err != nil || descriptorError != nil || !egraph.CanRewrite(c, c) {
				t.Fatalf("valid payload refused: %v, %v", err, descriptorError)
			}
			if got := desc.ExpectedBytes(); got <= 0 || uint64(got) != c.Count*uint64(desc.DataType.SizeBytes()) {
				t.Fatalf("runtime payload wrapped: %d", got)
			}
		})
	}
}

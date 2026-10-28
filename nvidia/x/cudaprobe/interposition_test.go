package cudaprobe

import (
	"strings"
	"testing"
	"time"
)

func TestInterpositionValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value Interposition
		ok    bool
	}{
		{"zero", Interposition{}, true}, {"unknown", Interposition{Status: "unknown"}, true},
		{"active", Interposition{Status: "declared_active", Evidence: "caller"}, true},
		{"inactive", Interposition{Status: "declared_inactive", Evidence: strings.Repeat("x", 1024)}, true},
		{"unknown evidence", Interposition{Evidence: "caller"}, false},
		{"no evidence", Interposition{Status: "declared_inactive"}, false},
		{"long evidence", Interposition{Status: "declared_active", Evidence: strings.Repeat("x", 1025)}, false},
		{"invalid", Interposition{Status: "verified", Evidence: "caller"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.value.Validate(); (err == nil) != tc.ok {
				t.Fatalf("Validate=%v", err)
			}
			_, err := (Event{Identity: Identity{Interposition: tc.value}}).Record(time.Unix(100, 0))
			if (err == nil) != tc.ok {
				t.Fatalf("Record=%v", err)
			}
		})
	}
}

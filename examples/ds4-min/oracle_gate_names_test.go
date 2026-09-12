package main

import (
	"reflect"
	"testing"
)

func TestOracleGateNames(t *testing.T) {
	tests := []struct {
		name, source string
		want         []string
		bad          bool
	}{
		{name: "environment read", source: `package p; import "os"; var x = os.Getenv("DS4_MODEL")`, want: []string{"DS4_MODEL"}},
		{name: "helper argument", source: `package p; var x = helper("DS4_PREFILL_GATE")`, want: []string{"DS4_PREFILL_GATE"}},
		{name: "kernel macros", source: "package p; const kernel = `#define DS4_SCORE_TILE_HEADS 16u\n#define DS4_SCORE_TILE_ROWS 16u\n#define DS4_SCORE_TILE_STRIDE 516u`"},
		{name: "comment", source: "package p; // DS4_COMMENT_ONLY\n"},
		{name: "diagnostic prose", source: `package p; const message = "set DS4_MODEL before running"`},
		{name: "invalid source", source: "package", bad: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := oracleGateNames("fixture.go", []byte(tt.source))
			if (err != nil) != tt.bad {
				t.Fatalf("error = %v, want failure %v", err, tt.bad)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("names = %q, want %q", got, tt.want)
			}
		})
	}
}

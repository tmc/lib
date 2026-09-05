package nvrtcx

import (
	"fmt"
	"testing"
)

func TestJITLogString(t *testing.T) {
	for _, tt := range []struct{ name, input, want string }{
		{"empty", "", ""},
		{"terminated", "message\x00stale", "message"},
		{"full", "truncated", "truncated"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := jitLogString([]byte(tt.input)); got != tt.want {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}

func ExampleCompiledModule_LoadOutputWithLog() {
	var cm *CompiledModule
	_, log, err := cm.LoadOutputWithLog(nil, OutputPTX)
	fmt.Println(log.Output, err)
	// Output: 1 nvrtcx: nil compiled module
}

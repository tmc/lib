package nvtelemetry_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleOpenRetention() {
	_, err := nvtelemetry.OpenRetention("unused", nvtelemetry.RetentionConfig{})
	fmt.Println(err)
	// Output: invalid retention configuration
}
func ExampleRecoverRetention() {
	// After establishing that a crashed writer has stopped, the owning task
	// can explicitly release its stale lease before OpenRetention reconciles
	// an unpublished pending file. No process is stopped by this API.
	fmt.Println("recover only after the writer has stopped")
	// Output: recover only after the writer has stopped
}

func ExampleRetention_Store() {
	var retained nvtelemetry.Retention
	fmt.Println(retained.Store(1, nil))
	// Output: retention is closed
}

func ExampleRetention_Close() {
	var retained nvtelemetry.Retention
	fmt.Println(retained.Close())
	// Output: <nil>
}

func ExampleRetention_Status() {
	var retained nvtelemetry.Retention
	status := retained.Status()
	fmt.Println(status.Closed, status.Ready, status.CountsAvailable)
	// Output: true false false
}

package cudagraph

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestUpdateErrorNodes(t *testing.T) {
	tests := []struct {
		name string
		info driver.GraphExecUpdateResultInfo
		want string
	}{{
		// A topology change is not any one node's fault and the driver leaves
		// both handles zero. Naming a node here would invent one.
		name: "neither set",
		info: driver.GraphExecUpdateResultInfo{Result: driver.GraphExecUpdateErrorTopologyChanged},
		want: "",
	}, {
		name: "node only",
		info: driver.GraphExecUpdateResultInfo{ErrorNode: 0x7f2c},
		want: ", node=0x7f2c",
	}, {
		name: "from node only",
		info: driver.GraphExecUpdateResultInfo{ErrorFromNode: 0x40},
		want: ", from node=0x40",
	}, {
		name: "both set",
		info: driver.GraphExecUpdateResultInfo{ErrorNode: 0x7f2c, ErrorFromNode: 0x40},
		want: ", node=0x7f2c, from node=0x40",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := updateErrorNodes(tt.info); got != tt.want {
				t.Errorf("updateErrorNodes() = %q, want %q", got, tt.want)
			}
		})
	}
}

//go:build darwin

package metalprof

import (
	apple "github.com/tmc/apple/metal"
)

// PushDebugGroup attaches an active pprof context label string as a Metal debug scope.
func PushDebugGroup(enc apple.MTLCommandEncoder, groupName string) {
	if enc.GetID() != 0 && groupName != "" {
		enc.PushDebugGroup(groupName)
	}
}

// PopDebugGroup pops the current Metal debug scope.
func PopDebugGroup(enc apple.MTLCommandEncoder) {
	if enc.GetID() != 0 {
		enc.PopDebugGroup()
	}
}

// InsertDebugSignpost inserts an instant debug signpost marker into the Metal command stream.
func InsertDebugSignpost(enc apple.MTLCommandEncoder, signpost string) {
	if enc.GetID() != 0 && signpost != "" {
		enc.InsertDebugSignpost(signpost)
	}
}

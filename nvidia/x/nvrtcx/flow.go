package nvrtcx

import (
	"context"
	"sync"

	"github.com/ebitengine/purego"
)

var (
	flowCallbackOnce sync.Once
	flowCallbackPtr  uintptr

	flowMu       sync.Mutex
	flowNextID   uintptr = 1
	flowContexts         = make(map[uintptr]context.Context)
)

func globalFlowCallback(payload, reserved uintptr) int32 {
	flowMu.Lock()
	ctx, ok := flowContexts[payload]
	flowMu.Unlock()
	if ok && ctx != nil && ctx.Err() != nil {
		return 1
	}
	return 0
}

func getFlowCallback() uintptr {
	flowCallbackOnce.Do(func() {
		flowCallbackPtr = purego.NewCallback(globalFlowCallback)
	})
	return flowCallbackPtr
}

func registerFlowContext(ctx context.Context) (uintptr, func()) {
	flowMu.Lock()
	token := flowNextID
	flowNextID++
	flowContexts[token] = ctx
	flowMu.Unlock()

	return token, func() {
		flowMu.Lock()
		delete(flowContexts, token)
		flowMu.Unlock()
	}
}

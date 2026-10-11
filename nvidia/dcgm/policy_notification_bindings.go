package dcgm

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// Only the read-only notification registration pair is resolved here. Policy
// setters, triggers and remediation remain outside the callable surface.
var notificationBindings struct {
	sync.Mutex
	register   func(Handle, GpuGrp, PolicyCondition, FpRecvUpdates, uint64) Result
	unregister func(Handle, GpuGrp, PolicyCondition) Result
	err        error
}

func loadNotificationBindings() error {
	if err := checkLib(); err != nil {
		return err
	}
	func() {
		notificationBindings.err = nil
		defer func() {
			if p := recover(); p != nil {
				notificationBindings.err = fmt.Errorf("dcgm: bind policy notifications: %v", p)
			}
		}()
		loadMu.Lock()
		defer loadMu.Unlock()
		for _, binding := range []struct {
			name   string
			target any
		}{
			{"dcgmPolicyRegister_v2", &notificationBindings.register},
			{"dcgmPolicyUnregister", &notificationBindings.unregister},
		} {
			ptr, err := lookupLibrarySymbol(lib, binding.name)
			if err != nil {
				notificationBindings.err = &SymbolError{Symbol: binding.name, Err: err}
				return
			}
			purego.RegisterFunc(binding.target, ptr)
		}
	}()
	return notificationBindings.err
}

func registerPolicyNotifications(h Handle, g GpuGrp, mask PolicyCondition, callback FpRecvUpdates, token uint64) error {
	notificationBindings.Lock()
	defer notificationBindings.Unlock()
	if err := loadNotificationBindings(); err != nil {
		return err
	}
	return toError(notificationBindings.register(h, g, mask, callback, token), "dcgmPolicyRegister_v2")
}

func unregisterPolicyNotifications(h Handle, g GpuGrp, mask PolicyCondition) error {
	notificationBindings.Lock()
	defer notificationBindings.Unlock()
	if err := loadNotificationBindings(); err != nil {
		return err
	}
	return toError(notificationBindings.unregister(h, g, mask), "dcgmPolicyUnregister")
}

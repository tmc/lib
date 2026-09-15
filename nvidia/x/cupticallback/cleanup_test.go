package cupticallback

import (
	"errors"
	"testing"
)

func TestUnsubscribeRetainsHandlerOnFailure(t *testing.T) {
	previous := handlerSlot.Load()
	state := &handlerState{}
	handlerSlot.Store(state)
	t.Cleanup(func() { handlerSlot.Store(previous) })
	calls := 0
	fail := errors.New("native unsubscribe")
	s := &Subscription{unsubscribe: func() error {
		calls++
		if calls == 1 {
			return fail
		}
		return nil
	}}
	if err := s.Unsubscribe(); !errors.Is(err, fail) {
		t.Fatalf("Unsubscribe=%v", err)
	}
	if handlerSlot.Load() != state || s.closed {
		t.Fatal("failed unsubscribe discarded active handler")
	}
	if err := s.Unsubscribe(); err != nil {
		t.Fatal(err)
	}
	if handlerSlot.Load() != nil || !s.closed {
		t.Fatal("successful unsubscribe retained handler")
	}
	if err := s.Unsubscribe(); err != nil || calls != 2 {
		t.Fatalf("repeat=%v calls=%d", err, calls)
	}
}

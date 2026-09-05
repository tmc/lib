package nvrtcx

import (
	"errors"
	"io"
	"sync"
	"testing"
)

func TestProgramCleanupRetry(t *testing.T) {
	compileErr, destroyErr := errors.New("compile failed"), errors.New("destroy failed")
	calls := 0
	result, err := finishProgram(&CompiledModule{}, compileErr, func() error {
		calls++
		if calls < 3 {
			return destroyErr
		}
		return nil
	})
	if result != nil || !errors.Is(err, compileErr) || !errors.Is(err, destroyErr) {
		t.Fatalf("result=%v error=%v", result, err)
	}
	var closer io.Closer
	if !errors.As(err, &closer) {
		t.Fatal("lost program cleanup owner")
	}
	if err := closer.Close(); !errors.Is(err, destroyErr) {
		t.Fatalf("retry=%v", err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := closer.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if calls != 3 {
		t.Fatalf("destruction calls=%d", calls)
	}
}

func TestProgramCleanupSuccess(t *testing.T) {
	want := &CompiledModule{}
	got, err := finishProgram(want, nil, func() error { return nil })
	if got != want || err != nil {
		t.Fatalf("result=%v error=%v", got, err)
	}
}

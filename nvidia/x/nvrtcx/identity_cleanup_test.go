package nvrtcx

import (
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"
)

func TestFinishCompilationIdentityCleanup(t *testing.T) {
	compileErr := errors.New("compile failed")
	destroyErr := errors.New("destroy failed")
	identityErr := errors.New("identity changed")
	tests := []struct {
		name     string
		compile  error
		destroy  error
		identity error
	}{
		{"identity only", nil, nil, identityErr},
		{"compile and identity", compileErr, nil, identityErr},
		{"cleanup and identity", nil, destroyErr, identityErr},
		{"compile cleanup and identity", compileErr, destroyErr, identityErr},
		{"cleanup only", nil, destroyErr, nil},
		{"compile only", compileErr, nil, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var order []string
			calls, cacheCalls := 0, 0
			input := &CompiledModule{ptx: "candidate"}
			if tt.compile != nil {
				input = nil
			}
			result, err := finishCompilation(input, tt.compile, func() error {
				order = append(order, "destroy")
				calls++
				if calls <= 2 {
					return tt.destroy
				}
				return nil
			}, func() error {
				order = append(order, "verify")
				return tt.identity
			}, func(*CompiledModule) error {
				cacheCalls++
				return nil
			})
			if result != nil || err == nil || cacheCalls != 0 || !reflect.DeepEqual(order, []string{"destroy", "verify"}) {
				t.Fatalf("result=%v error=%v cache calls=%d order=%v", result, err, cacheCalls, order)
			}
			for _, want := range []error{tt.compile, tt.destroy, tt.identity} {
				if want != nil && !errors.Is(err, want) {
					t.Errorf("lost error %v in %v", want, err)
				}
			}
			var closer io.Closer
			if errors.As(err, &closer) != (tt.destroy != nil) {
				t.Fatalf("cleanup owner presence differs: %v", err)
			}
			if tt.destroy == nil {
				return
			}
			if retryErr := closer.Close(); !errors.Is(retryErr, destroyErr) {
				t.Fatalf("cleanup retry = %v, want destroy error", retryErr)
			}
			var wg sync.WaitGroup
			for range 8 {
				wg.Go(func() {
					if retryErr := closer.Close(); retryErr != nil {
						t.Error(retryErr)
					}
				})
			}
			wg.Wait()
			if calls != 3 || cacheCalls != 0 {
				t.Fatalf("destroy calls=%d cache calls=%d, want 3 and 0", calls, cacheCalls)
			}
		})
	}
}

func TestFinishCompilationCacheAdmission(t *testing.T) {
	cacheErr := errors.New("cache failed")
	for _, tt := range []struct {
		name     string
		cache    bool
		cacheErr error
	}{
		{"cache disabled", false, nil},
		{"cache success", true, nil},
		{"cache error", true, cacheErr},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input := &CompiledModule{ptx: "candidate"}
			var order []string
			var cache func(*CompiledModule) error
			if tt.cache {
				cache = func(result *CompiledModule) error {
					if result != input {
						t.Fatal("cache received different artifact")
					}
					order = append(order, "cache")
					return tt.cacheErr
				}
			}
			result, err := finishCompilation(input, nil,
				func() error { order = append(order, "destroy"); return nil },
				func() error { order = append(order, "verify"); return nil }, cache)
			wantOrder := []string{"destroy", "verify"}
			if tt.cache {
				wantOrder = append(wantOrder, "cache")
			}
			if !reflect.DeepEqual(order, wantOrder) {
				t.Fatalf("order = %v, want %v", order, wantOrder)
			}
			if tt.cacheErr != nil {
				if result != nil || !errors.Is(err, cacheErr) {
					t.Fatalf("cache refusal: result=%v error=%v", result, err)
				}
			} else if result != input || err != nil {
				t.Fatalf("success: result=%v error=%v", result, err)
			}
		})
	}
}

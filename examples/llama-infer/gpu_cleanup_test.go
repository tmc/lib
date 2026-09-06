package main

import (
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestCUDAInitializationCleanupRetry(t *testing.T) {
	original := errors.New("load failed")
	failure := errors.New("unload failed")
	fail := true
	var calls []string
	err := finishCUDAInit(original,
		func() error {
			calls = append(calls, "module")
			if fail {
				return failure
			}
			return nil
		},
		func() error { calls = append(calls, "context"); return nil },
	)
	if !errors.Is(err, original) || !errors.Is(err, failure) {
		t.Fatalf("lost error: %v", err)
	}
	if !reflect.DeepEqual(calls, []string{"module"}) {
		t.Fatalf("closed dependency: %v", calls)
	}
	var retry io.Closer
	if !errors.As(err, &retry) {
		t.Fatal("cleanup cannot be retried")
	}
	fail = false
	if err := retry.Close(); err != nil {
		t.Fatal(err)
	}
	if err := retry.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, []string{"module", "module", "context"}) {
		t.Fatalf("retry order: %v", calls)
	}
}

func TestCUDAInitializationNestedCleanup(t *testing.T) {
	fail := true
	closed := false
	inner := finishCUDAInit(errors.New("compile failed"), func() error {
		if fail {
			return errors.New("destroy failed")
		}
		return nil
	})
	outer := finishCUDAInit(inner, func() error { closed = true; return nil })
	if closed {
		t.Fatal("closed dependency before compiler cleanup")
	}
	var retry io.Closer
	if !errors.As(outer, &retry) {
		t.Fatal("cleanup cannot be retried")
	}
	fail = false
	if err := retry.Close(); err != nil {
		t.Fatal(err)
	}
	if !closed {
		t.Fatal("dependency not closed")
	}
}

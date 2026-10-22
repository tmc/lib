//go:build linux

package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/driver"
)

func TestCUDASteeringCleanupFailureRetry(t *testing.T) {
	for _, failureAt := range []string{"module", "directions", "both"} {
		t.Run(failureAt, func(t *testing.T) {
			failure := errors.New("release failed")
			attempt := 0
			moduleCalls, directionCalls := 0, 0
			g := &cudaSteering{module: driver.Module(1)}
			g.releaseDirections = func() error {
				directionCalls++
				if attempt == 0 && (failureAt == "directions" || failureAt == "both") {
					return failure
				}
				return nil
			}
			unload := func(driver.Module) error {
				moduleCalls++
				if attempt == 0 && (failureAt == "module" || failureAt == "both") {
					return failure
				}
				return nil
			}
			r := &cudaRuntime{steering: g}
			if err := r.closeSteering(func(g *cudaSteering) error { return g.closeWithModuleUnload(unload) }); !errors.Is(err, failure) {
				t.Fatal(err)
			}
			if r.steering != g {
				t.Fatal("failed cleanup lost owner")
			}
			if (g.module != 0) != (failureAt == "module" || failureAt == "both") {
				t.Fatal("module handle not retained exactly on failed unload")
			}
			if (g.releaseDirections != nil) != (failureAt == "directions" || failureAt == "both") {
				t.Fatal("direction owner not retained exactly on failed free")
			}
			attempt++
			if err := r.closeSteering(func(g *cudaSteering) error { return g.closeWithModuleUnload(unload) }); err != nil {
				t.Fatal(err)
			}
			if r.steering != nil || g.module != 0 || g.releaseDirections != nil {
				t.Fatal("successful retry retained resource")
			}
			wantModule, wantDirection := 1, 1
			if failureAt == "module" || failureAt == "both" {
				wantModule = 2
			}
			if failureAt == "directions" || failureAt == "both" {
				wantDirection = 2
			}
			if moduleCalls != wantModule || directionCalls != wantDirection {
				t.Fatalf("release counts %d %d", moduleCalls, directionCalls)
			}
		})
	}
}

func TestCUDASteeringUnloadSuccessContextRestoreFailure(t *testing.T) {
	for _, failedDirections := range []bool{false, true} {
		t.Run(fmt.Sprint(failedDirections), func(t *testing.T) {
			restoreFailure := errors.New("restore context failed")
			directionFailure := errors.New("release directions failed")
			contextCalls, unloadCalls, directionCalls := 0, 0, 0
			g := &cudaSteering{module: driver.Module(1)}
			g.releaseDirections = func() error {
				directionCalls++
				if failedDirections && directionCalls == 1 {
					return directionFailure
				}
				return nil
			}
			withContext := func(fn func() error) error {
				contextCalls++
				if err := fn(); err != nil {
					return err
				}
				return restoreFailure
			}
			unload := func(driver.Module) error { unloadCalls++; return nil }
			r := &cudaRuntime{steering: g}
			close := func(g *cudaSteering) error { return g.closeWithContext(withContext, unload) }
			err := r.closeSteering(close)
			if !errors.Is(err, restoreFailure) || failedDirections && !errors.Is(err, directionFailure) {
				t.Fatalf("lost release errors: %v", err)
			}
			if g.module != 0 || r.steering != g || (g.releaseDirections != nil) != failedDirections {
				t.Fatal("restoration failure retained released handle or lost pending directions")
			}
			if err := r.closeSteering(close); err != nil {
				t.Fatal(err)
			}
			wantDirections := 1
			if failedDirections {
				wantDirections++
			}
			if unloadCalls != 1 || contextCalls != 1 || directionCalls != wantDirections || r.steering != nil {
				t.Fatalf("retry counts unload=%d context=%d directions=%d", unloadCalls, contextCalls, directionCalls)
			}
		})
	}
}

func TestCUDASteeringFailedLoadRetainsCleanupOwner(t *testing.T) {
	loadFailure, cleanupFailure := errors.New("load failed"), errors.New("cleanup failed")
	r := &cudaRuntime{}
	s := &cudaRequestState{runtime: r}
	g := &cudaSteering{r: r}
	err := s.retainFailedSteering(g, loadFailure, cleanupFailure)
	if !errors.Is(err, loadFailure) || !errors.Is(err, cleanupFailure) || !errors.Is(err, dsv4.ErrDesynchronized) || r.steering != g || r.failed == nil {
		t.Fatalf("error %v owner %p", err, r.steering)
	}
	if err := r.closeSteering(func(*cudaSteering) error { return nil }); err != nil || r.steering != nil {
		t.Fatalf("cleanup retry %v", err)
	}
	other := &cudaRequestState{runtime: &cudaRuntime{}}
	if err := other.retainFailedSteering(g, loadFailure, nil); !errors.Is(err, loadFailure) || other.runtime.steering != nil || other.runtime.failed != nil {
		t.Fatal("successful cleanup published failed steering")
	}
}

func TestCUDASteeringCloseRetryRetention(t *testing.T) {
	failure := errors.New("stream sync failed")
	r := &cudaRuntime{steering: &cudaSteering{}}
	s := &cudaRequestState{runtime: r}
	if err := s.finishClose(failure); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if s.closed || s.runtime != r || s.runtime.steering == nil {
		t.Fatal("early close failure lost retryable resources")
	}
	if err := s.Reset(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatalf("terminal request reset: %v", err)
	}
	if err := s.setSteeringFFN(1); !errors.Is(err, failure) {
		t.Fatalf("terminal request changed control: %v", err)
	}
	if err := s.loadSteering(nil, 0, 0); !errors.Is(err, failure) {
		t.Fatalf("terminal request admitted steering: %v", err)
	}
	if s.logits() != nil {
		t.Fatal("terminal request exposed logits")
	}
	r.steering = nil
	if err := s.finishClose(nil); err != nil {
		t.Fatal(err)
	}
	if !s.closed || s.runtime != nil {
		t.Fatal("successful close retained resources")
	}
}

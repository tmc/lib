package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestAFMoEMoEModuleOwnership(t *testing.T) {
	lookupErr := errors.New("lookup failed")
	closeErr := errors.New("close failed")
	for _, test := range []struct {
		name      string
		closeFail bool
	}{
		{"lookup cleanup", false},
		{"failed cleanup retains owner", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			lookups, closes := 0, 0
			m, err := lookupAFMoEMoEKernels(driver.Module(123), func(name string) (driver.Function, error) {
				lookups++
				if name == "AFMoESelectedGateUpBF16" {
					return 0, lookupErr
				}
				return driver.Function(lookups), nil
			}, func(mod *driver.Module) error {
				closes++
				if *mod != 123 {
					t.Errorf("closing module %d, want 123", *mod)
				}
				if test.closeFail {
					return closeErr
				}
				*mod = 0
				return nil
			})
			if !errors.Is(err, lookupErr) || closes != 1 || lookups != 5 {
				t.Fatalf("lookup = %v; %d lookups, %d closes", err, lookups, closes)
			}
			if !test.closeFail {
				if m != nil {
					t.Fatal("returned owner after successful cleanup")
				}
				return
			}
			if !errors.Is(err, closeErr) || m == nil || m.mod != 123 {
				t.Fatal("failed cleanup lost module or error")
			}
			if err := m.close(func(mod *driver.Module) error { *mod = 0; return nil }); err != nil {
				t.Fatal(err)
			}
			if m.mod != 0 || m.routerLinearFn != 0 || m.selectedGateFn != 0 {
				t.Fatal("successful close retained module or function")
			}
			if err := m.close(func(*driver.Module) error { t.Fatal("closed twice"); return nil }); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAFMoEMoEModuleRejectsZeroFunction(t *testing.T) {
	closes := 0
	m, err := lookupAFMoEMoEKernels(driver.Module(123), func(string) (driver.Function, error) {
		return 0, nil
	}, func(mod *driver.Module) error { closes++; *mod = 0; return nil })
	if err == nil || m != nil || closes != 1 {
		t.Fatalf("zero function = %v, %v, %d closes", m, err, closes)
	}
}

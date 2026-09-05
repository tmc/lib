package dcgm

import (
	"errors"
	"os"
	"testing"
)

// This file checks the other half of the ABI. layout_test.go and
// versions_test.go check that the Go types describe the pinned headers; that
// can be done with no library present, and it is what the rest of the test
// suite does. What it cannot check is that the names this package asks the
// dynamic linker for are names libdcgm actually exports. A misspelled or
// removed symbol is invisible until the first call, which is the machine with
// the GPU on it.
//
// So the checks here need a real libdcgm. They skip when there is none, and
// the skip message says what is missing. A skip is not a pass: it is a record
// that nothing was checked. Set DCGM_REQUIRE_LIBRARY=1 to turn the skip into a
// failure on a machine where the library is supposed to be there.

// requireLibrary loads libdcgm or explains why it could not.
func requireLibrary(t *testing.T) {
	t.Helper()
	if err := Load(""); err != nil {
		if os.Getenv("DCGM_REQUIRE_LIBRARY") != "" {
			t.Fatalf("DCGM_REQUIRE_LIBRARY is set but libdcgm could not be loaded: %v", err)
		}
		t.Skipf("libdcgm is not loadable here, so no symbol in this test was checked. "+
			"DCGM ships for Linux only; install the NVIDIA datacenter-gpu-manager package "+
			"or point DCGM_LIB_PATH at libdcgm.so.4, and set DCGM_REQUIRE_LIBRARY=1 to make "+
			"this a failure instead of a skip: %v", err)
	}
}

// boundSymbols is every C symbol the bound surface calls, spelled as this
// package asks the linker for it. Versioned entry points appear under the
// versioned name because that is what libdcgm exports: dcgmConnect_v2, not
// dcgmConnect.
var boundSymbols = []string{
	// Library and connection lifetime.
	"dcgmInit",
	"dcgmShutdown",
	"dcgmConnect_v2",
	"dcgmDisconnect",
	"dcgmStartEmbedded_v2",
	"dcgmStopEmbedded",

	// Discovery.
	"dcgmGetAllDevices",
	"dcgmGetAllSupportedDevices",
	"dcgmGetDeviceAttributes",
	"dcgmGetEntityGroupEntities",
	"dcgmGetGpuInstanceHierarchy",

	// Groups.
	"dcgmGroupCreate",
	"dcgmGroupDestroy",
	"dcgmGroupAddEntity",
	"dcgmGroupAddDevice",
	"dcgmGroupGetAllIds",
	"dcgmGroupGetInfo",

	// Field groups and watches.
	"dcgmFieldGroupCreate",
	"dcgmFieldGroupDestroy",
	"dcgmFieldGroupGetAll",
	"dcgmFieldGroupGetInfo",
	"dcgmWatchFields",
	"dcgmUnwatchFields",
	"dcgmUpdateAllFields",

	// Value retrieval.
	"dcgmGetLatestValuesForFields",
	"dcgmEntitiesGetLatestValues",
	"dcgmEntityGetLatestValues",
	"dcgmGetFieldSummary",

	// Read-only diagnostic and report retrieval.
	"dcgmHealthGet",
	"dcgmHealthCheck",
	"dcgmHostengineIsHealthy",
	"dcgmHostengineVersionInfo",
	"dcgmModuleGetStatuses",
	"dcgmVersionInfo",

	// Status handles, which carry the per-entity errors a group call
	// accumulates.
	"dcgmStatusCreate",
	"dcgmStatusDestroy",
	"dcgmStatusClear",
	"dcgmStatusGetCount",
	"dcgmStatusPopError",
}

// TestBoundSymbolsResolve checks that every symbol the bound surface calls
// exists in the loaded library.
//
// The soname is libdcgm.so.4 for the whole DCGM 4.x line, so a symbol missing
// from a library with that soname is this package naming something DCGM does
// not export, not an old release. There is no version check that would excuse
// it.
func TestBoundSymbolsResolve(t *testing.T) {
	requireLibrary(t)
	for _, symbol := range boundSymbols {
		if !Has(symbol) {
			t.Errorf("%s did not resolve in %s, which every DCGM 4.x release should export", symbol, Path())
		}
	}
}

// TestUnboundSurfaceStaysUnbound checks from the other side: the
// configuration-changing entry points are exported by the library and are
// deliberately not reachable through this package. If one of them ever becomes
// reachable it will be because the generator's skip list stopped matching, and
// this catches that on any machine with the library.
//
// It asserts about the library, not about this package's API. That the Go
// wrapper does not exist is checked without a GPU by TestGeneration in the
// generator; this only confirms the names are real, so the skip list is not
// guarding against nothing.
func TestUnboundSurfaceStaysUnbound(t *testing.T) {
	requireLibrary(t)
	for _, symbol := range []string{
		"dcgmConfigSet",
		"dcgmPolicySet",
		"dcgmHealthSet",
		"dcgmActionValidate",
	} {
		if !Has(symbol) {
			t.Logf("%s is absent from %s; the generator skips it either way", symbol, Path())
			continue
		}
		if checkLibSym(symbol) != nil {
			t.Errorf("%s resolved but checkLibSym rejects it", symbol)
		}
	}
}

// TestAbsentSymbolIsAnError checks the contract for a symbol the loaded
// library does not have: a structured error naming it, never a nil-pointer
// call. This is the failure mode a caller on an older library hits, and it
// must be recoverable.
func TestAbsentSymbolIsAnError(t *testing.T) {
	requireLibrary(t)
	err := checkLibSym("dcgmThisSymbolDoesNotExist")
	if err == nil {
		t.Fatal("checkLibSym returned nil for a symbol libdcgm does not export")
	}
	var se *SymbolError
	if !errors.As(err, &se) {
		t.Fatalf("checkLibSym returned %T (%v), want *SymbolError", err, err)
	}
	if se.Symbol != "dcgmThisSymbolDoesNotExist" {
		t.Errorf("SymbolError names %q, want the symbol asked for", se.Symbol)
	}
}

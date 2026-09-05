package curand

import (
	"errors"
	"os"
	"testing"
)

func TestWindowsLoaderFixture(t *testing.T) {
	path := os.Getenv("CURAND_TEST_DLL")
	if path == "" {
		t.Skip("set CURAND_TEST_DLL to the compiled synthetic loader fixture")
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := Close(); err != nil {
			t.Error(err)
		}
	})
	var loadError *LibraryError
	if err := Load(path + ".missing"); !errors.As(err, &loadError) {
		t.Fatalf("missing DLL error = %v", err)
	}
	if len(loadError.Attempts) != 1 {
		t.Fatalf("missing DLL attempts = %v", loadError.Attempts)
	}
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	if !Loaded() || !HasGetVersion() {
		t.Fatal("fixture symbol was not loaded")
	}
	version, err := GetVersion()
	if err != nil || version != 4242 {
		t.Fatalf("synthetic version = %d, %v", version, err)
	}
	var conflict *LibraryConflictError
	if err := Load(path + ".other"); !errors.As(err, &conflict) {
		t.Fatalf("conflicting DLL = %v", err)
	}
	if HasCreateGenerator() {
		t.Fatal("fixture unexpectedly exports CreateGenerator")
	}
	var symbolError *SymbolError
	if _, err := CreateGenerator(CURAND_RNG_PseudoDefault); !errors.As(err, &symbolError) {
		t.Fatalf("missing symbol = %v", err)
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	if Loaded() || Path() != "" {
		t.Fatal("close retained loader state")
	}
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	version, err = GetVersion()
	if err != nil || version != 4242 {
		t.Fatalf("reloaded synthetic version = %d, %v", version, err)
	}
}

//go:build !linux && !darwin

package stage

import "fmt"

func openLibrary(string) (uintptr, error) {
	return 0, fmt.Errorf("stage libraries require Linux or Darwin")
}
func closeLibrary(uintptr) error { return fmt.Errorf("stage libraries require Linux or Darwin") }
func bindEntry(uintptr, string, any) error {
	return fmt.Errorf("stage libraries require Linux or Darwin")
}

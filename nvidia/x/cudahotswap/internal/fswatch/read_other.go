//go:build !unix

package fswatch

import "fmt"

// ReadFile refuses platforms without a nonblocking, no-follow regular-file
// opener. It does not replace that contract with a potentially blocking open.
func ReadFile(path string, limit int64) ([]byte, error) {
	return nil, fmt.Errorf("read file: safe regular-file reads unsupported on this platform")
}

// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"os"
	"testing"
)

func TestSSDOffloaderCreation(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "metalkv-test-*")
	if err != nil {
		t.Fatalf("MkdirTemp failed: %v", err)
	}
	defer os.RemoveAll(tempDir)

	offloader, err := NewSSDOffloader(tempDir)
	if err != nil {
		t.Fatalf("NewSSDOffloader failed: %v", err)
	}
	defer offloader.Close()
}

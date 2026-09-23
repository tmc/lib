package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

// engineOutputPath returns where the n-th engine built in this process writes
// an output named by a single flag. The first engine keeps path unchanged, so
// the common one-model case is stable; each later engine gets -n inserted
// before the extension, so two configs in one process do not truncate each
// other's file. The empty path and "-" (standard error) pass through.
func engineOutputPath(path string, n int) string {
	if n <= 0 || path == "" || path == "-" {
		return path
	}
	ext := filepath.Ext(path)
	return fmt.Sprintf("%s-%d%s", strings.TrimSuffix(path, ext), n+1, ext)
}

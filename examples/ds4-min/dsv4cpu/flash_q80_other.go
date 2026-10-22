//go:build !linux || !arm64

package dsv4cpu

// nativeQ80Dot retains the DOTPROD schedule. Its bit qualification applies
// to macOS ARM64 clang; this selection does not qualify other platforms.
func nativeQ80Dot(row []byte, x []Q80Activation) float32 {
	return dotprodQ80Dot(row, x)
}

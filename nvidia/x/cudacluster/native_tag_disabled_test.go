//go:build !native

package cudacluster

// nativeBuild reports whether the test binary was compiled with the native
// build tag. The compiler selects this file or native_tag_enabled_test.go,
// never both and never neither, so the value does not depend on build
// metadata that a binary may lack.
const nativeBuild = false

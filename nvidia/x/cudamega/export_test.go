package cudamega

// CUDASourceBackend returns the rendering backend for p.
//
// It is exported to the package's external test only. The backend is not part
// of the package's API: a caller wanting the text calls [Program.Source],
// which is the same backend driven the same way. What the test needs and the
// API does not is a backend value to wrap, so that the driver's calls into it
// can be counted.
func CUDASourceBackend(p *Program) Backend { return &cudaSource{p: p} }

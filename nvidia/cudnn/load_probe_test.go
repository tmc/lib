package cudnn

import "testing"

// TestLoadBindsEveryResolvedSymbol forces the library load, which runs
// registerFuncs over every entry point. Before register recovered from
// purego.RegisterFunc, an unbindable signature panicked here and took the
// process down. On a host without libcudnn the load fails and the test is a
// no-op; it is meaningful only where the library is present.
func TestLoadBindsEveryResolvedSymbol(t *testing.T) {
	if err := checkLib(); err != nil {
		t.Skipf("libcudnn not loadable: %v", err)
	}
	if !Has("cudnnNormalizationBackward") {
		t.Log("cudnnNormalizationBackward unbound: absent from the library, or purego cannot bind its 33-parameter signature on this architecture")
		return
	}
	t.Log("cudnnNormalizationBackward bound")
}

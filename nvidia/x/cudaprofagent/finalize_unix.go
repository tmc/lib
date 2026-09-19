//go:build unix

package cudaprofagent

import "github.com/ebitengine/purego"

// lookupFinalize resolves cudaprof_finalize in whatever is already loaded.
//
// RTLD_DEFAULT rather than opening the agent by path: the agent is injected
// into this process by cudaprof, so it is already mapped, and a path would be
// a second way to be wrong about which agent is attached. A process with no
// agent simply finds nothing, which is the no-op case.
func lookupFinalize() func() {
	sym, err := purego.Dlsym(purego.RTLD_DEFAULT, "cudaprof_finalize")
	if err != nil || sym == 0 {
		return nil
	}
	var call func()
	purego.RegisterFunc(&call, sym)
	return call
}

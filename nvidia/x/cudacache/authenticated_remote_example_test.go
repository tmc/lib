package cudacache_test

import (
	"crypto/ed25519"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudacache"
)

func ExampleSignRemoteProof() {
	// Deterministic key for this example only. Producers must protect private keys.
	private := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	proof, err := cudacache.SignRemoteProof(cudacache.RemoteArtifact, "cache-key", "producer-A", "release-key", []byte("artifact bytes"), private)
	fmt.Println(len(proof) > 0, err)
	// Output: true <nil>
}

func ExampleRemoteProofSource() {
	var source cudacache.RemoteProofSource = cudacache.NewHTTPRemoteSource("https://registry.example", nil)
	fmt.Println(source != nil)
	// Output: true
}

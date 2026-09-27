package cudacache_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleDigestTool() {
	dir, err := os.MkdirTemp("", "cudacache-tool")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "libnvrtc.so.13")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		fmt.Println(err)
		return
	}

	d, err := cudacache.DigestTool("nvrtc", path)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(d.SHA256)
	fmt.Println(cudacache.VerifyTool(d))
	// Output:
	// ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
	// <nil>
}

func ExampleToolProvenance_Diff() {
	recorded := cudacache.ToolProvenance{Recipe: cudacache.RecipeVersion, DriverVersion: 13000}
	active := recorded
	active.DriverVersion = 13010
	fmt.Println(active.Fingerprint() == recorded.Fingerprint())
	fmt.Println(active.Diff(recorded))
	// Output:
	// false
	// [driver_version 13010 != 13000]
}

func ExampleNVRTCAdapter_GetOrCompile() {
	dir, err := os.MkdirTemp("", "cudacache-adapter")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		fmt.Println(err)
		return
	}
	a := cudacache.NewNVRTCAdapter(c)

	// In a program that has loaded NVRTC and the driver, take the identity
	// from cudacache.CurrentEnvironmentIdentity instead.
	id := cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		TargetArch:   "sm_121a",
		Provenance:   cudacache.ToolProvenance{Recipe: cudacache.RecipeVersion},
	}
	compiles := 0
	compile := func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
		compiles++
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX,
			[]byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry k() { ret; }\n"),
			"k", cudakernel.Schema{}, "sm_121a")
		return art, id.Provenance, err
	}
	for range 3 {
		art, err := a.GetOrCompile(context.Background(), "source-sha256", id, compile)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(art.EntryPoint)
	}
	fmt.Println("compiles:", compiles)
	// Output:
	// k
	// k
	// k
	// compiles: 1
}

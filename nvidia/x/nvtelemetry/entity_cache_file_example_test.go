package nvtelemetry_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func entityCacheExample() (*nvtelemetry.EntityCache, nvtelemetry.EntityAssignment, string, func()) {
	base, err := os.UserCacheDir()
	if err != nil {
		panic(err)
	}
	base = filepath.Join(base, "nvtelemetry-examples")
	if err := os.MkdirAll(base, 0700); err != nil {
		panic(err)
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp(base, "entity-")
	if err != nil {
		panic(err)
	}
	cache, err := nvtelemetry.NewEntityCache(8, 10)
	if err != nil {
		panic(err)
	}
	a := nvtelemetry.EntityAssignment{Process: nvtelemetry.EntityProcess{Process: cudaworkload.ProcessGenID{PID: 12, Generation: 400}, BootID: "boot-a", NodeID: "node-a", PIDNamespace: "pid:[100]", HasExecGeneration: true}, Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{JobID: "42", Attempt: "1"}, Source: "scheduler", SourceRevision: "fixture-v1", State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot-a", StartNanos: 10, EndNanos: 30, Sharing: "unknown"}
	return cache, a, filepath.Join(dir, "entities.json"), func() { os.RemoveAll(dir) }
}

func ExampleEntityCache_SaveFile() {
	cache, a, path, cleanup := entityCacheExample()
	defer cleanup()
	if err := cache.Put([]nvtelemetry.EntityAssignment{a}, 10); err != nil {
		panic(err)
	}
	if err := cache.SaveFile(path); err != nil {
		panic(err)
	}
	fmt.Println(len(cache.Get(a, 19)))
	// Output: 1
}

func ExampleEntityCache_LoadFile() {
	cache, a, path, cleanup := entityCacheExample()
	defer cleanup()
	if err := cache.Put([]nvtelemetry.EntityAssignment{a}, 10); err != nil {
		panic(err)
	}
	if err := cache.SaveFile(path); err != nil {
		panic(err)
	}
	restored, err := nvtelemetry.NewEntityCache(8, 100)
	if err != nil {
		panic(err)
	}
	scope := nvtelemetry.EntityCacheScope{BootID: a.Process.BootID, NodeID: a.Process.NodeID, ClockDomain: a.ClockDomain, ClockEpoch: a.ClockEpoch}
	discarded, err := restored.LoadFile(path, scope, 19)
	if err != nil {
		panic(err)
	}
	fmt.Println(discarded, len(restored.Get(a, 19)), len(restored.Get(a, 20)))
	// Output: false 1 0
}

func ExampleEntityCache_CompleteJob() {
	cache, a, _, cleanup := entityCacheExample()
	defer cleanup()
	if err := cache.Put([]nvtelemetry.EntityAssignment{a}, 10); err != nil {
		panic(err)
	}
	if err := cache.CompleteJob(a.Owner); err != nil {
		panic(err)
	}
	fmt.Println(len(cache.Get(a, 11)), cache.Put([]nvtelemetry.EntityAssignment{a}, 11) != nil)
	// Output: 0 true
}

func ExampleEntityCacheScope() {
	scope := nvtelemetry.EntityCacheScope{BootID: "boot-a", NodeID: "node-a", ClockDomain: "monotonic", ClockEpoch: "boot-a"}
	fmt.Println(scope.BootID, scope.ClockDomain)
	// Output: boot-a monotonic
}

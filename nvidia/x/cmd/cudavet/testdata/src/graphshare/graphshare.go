package graphshare

import (
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

func concurrent(exec driver.GraphExec, s1, s2 driver.Stream) error {
	go func() {
		exec.Launch(s1)
	}()
	return exec.Launch(s2) // want `exec is used here while the goroutine started at line \d+ may be using it; calls on a graph must be serialized`
}

func waited(exec driver.GraphExec, s1, s2 driver.Stream) error {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		exec.Launch(s1)
	}()
	wg.Wait()
	return exec.Launch(s2)
}

func locked(g driver.Graph, mu *sync.Mutex) (driver.Graph, error) {
	go func() {
		mu.Lock()
		defer mu.Unlock()
		g.Clone()
	}()
	mu.Lock()
	defer mu.Unlock()
	return g.Clone()
}

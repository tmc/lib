package cudagraph

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Graph represents an uninstantiated CUDA execution graph DAG.
type Graph struct {
	raw driver.Graph
}

// New creates an empty CUDA graph.
func New() (*Graph, error) {
	raw, err := driver.NewGraph(0)
	if err != nil {
		return nil, fmt.Errorf("cudagraph: create graph: %w", err)
	}
	return &Graph{raw: raw}, nil
}

// Capture executes fn on stream while recording all enqueued operations into a Graph.
func Capture(s *cuda.Stream, fn func()) (*Graph, error) {
	if s == nil {
		return nil, errors.New("cudagraph: nil stream passed to Capture")
	}

	if err := s.Raw().BeginCapture(driver.StreamCaptureModeGlobal); err != nil {
		return nil, fmt.Errorf("cudagraph: begin capture: %w", err)
	}

	fn()

	rawGraph, err := s.Raw().EndCapture()
	if err != nil {
		return nil, fmt.Errorf("cudagraph: end capture: %w", err)
	}

	return &Graph{raw: rawGraph}, nil
}

// Instantiate validates and instantiates the graph for execution on the GPU.
func (g *Graph) Instantiate() (*Exec, error) {
	rawExec, err := g.raw.Instantiate(0)
	if err != nil {
		return nil, fmt.Errorf("cudagraph: instantiate graph: %w", err)
	}
	return &Exec{raw: rawExec}, nil
}

// Raw returns the underlying driver.Graph handle.
func (g *Graph) Raw() driver.Graph {
	return g.raw
}

// Destroy destroys the uninstantiated graph.
func (g *Graph) Destroy() error {
	if g.raw == 0 {
		return nil
	}
	if err := g.raw.Destroy(); err != nil {
		return fmt.Errorf("cudagraph: destroy graph: %w", err)
	}
	g.raw = 0
	return nil
}

// Exec represents an instantiated, executable CUDA graph.
type Exec struct {
	raw driver.GraphExec
}

// Launch enqueues execution of the instantiated graph on stream.
func (e *Exec) Launch(s *cuda.Stream) error {
	var streamHandle driver.Stream
	if s != nil {
		streamHandle = s.Raw()
	}
	if err := e.raw.GraphLaunch(streamHandle); err != nil {
		return fmt.Errorf("cudagraph: launch graph: %w", err)
	}
	return nil
}

// Update updates the parameters of this executable graph from a topologically identical graph.
func (e *Exec) Update(g *Graph) error {
	if g == nil {
		return errors.New("cudagraph: nil graph passed to Update")
	}
	var result driver.GraphExecUpdateResultInfo
	if err := e.raw.Update(g.raw, &result); err != nil {
		return fmt.Errorf("cudagraph: update graph (result=%v): %w", result.Result, err)
	}
	return nil
}

// Destroy destroys the executable graph.
func (e *Exec) Destroy() error {
	if e.raw == 0 {
		return nil
	}
	if err := e.raw.Destroy(); err != nil {
		return fmt.Errorf("cudagraph: destroy exec graph: %w", err)
	}
	e.raw = 0
	return nil
}

// BucketManager manages a discrete lattice of instantiated graphs for dynamic tensor dimensions,
// avoiding expensive runtime recaptures and topology change errors.
type BucketManager struct {
	mu       sync.Mutex
	buckets  []int
	builder  func(s *cuda.Stream, dim int) (*Graph, error)
	execs    map[int]*Exec
	builds   map[int]*bucketBuild
	err      error
	closed   bool
	close    chan struct{}
	closeErr error
}

type bucketBuild struct {
	done chan struct{}
	exec *Exec
	err  error
}

// NewBucketManager creates a new manager caching instantiated graphs across discrete dimension buckets.
func NewBucketManager(buckets []int, builder func(s *cuda.Stream, dim int) (*Graph, error)) *BucketManager {
	sortedBuckets := make([]int, len(buckets))
	copy(sortedBuckets, buckets)
	sort.Ints(sortedBuckets)

	bm := &BucketManager{
		buckets: sortedBuckets,
		builder: builder,
		execs:   make(map[int]*Exec),
		builds:  make(map[int]*bucketBuild),
		close:   make(chan struct{}),
	}
	switch {
	case len(sortedBuckets) == 0:
		bm.err = errors.New("cudagraph: no buckets configured")
	case builder == nil:
		bm.err = errors.New("cudagraph: nil bucket builder")
	}
	for i, bucket := range sortedBuckets {
		if bucket <= 0 {
			bm.err = fmt.Errorf("cudagraph: invalid bucket %d", bucket)
			break
		}
		if i > 0 && bucket == sortedBuckets[i-1] {
			bm.err = fmt.Errorf("cudagraph: duplicate bucket %d", bucket)
			break
		}
	}
	return bm
}

// GetExec returns the executable graph matching the smallest bucket that accommodates dim.
func (bm *BucketManager) GetExec(s *cuda.Stream, dim int) (*Exec, int, error) {
	if bm == nil {
		return nil, 0, errors.New("cudagraph: nil bucket manager")
	}
	bm.mu.Lock()
	if bm.err != nil {
		bm.mu.Unlock()
		return nil, 0, bm.err
	}
	if dim <= 0 {
		bm.mu.Unlock()
		return nil, 0, fmt.Errorf("cudagraph: invalid dimension %d", dim)
	}
	if bm.closed {
		bm.mu.Unlock()
		return nil, 0, errors.New("cudagraph: bucket manager closed")
	}

	targetBucket := -1
	for _, b := range bm.buckets {
		if b >= dim {
			targetBucket = b
			break
		}
	}
	if targetBucket == -1 {
		bm.mu.Unlock()
		return nil, 0, fmt.Errorf("cudagraph: dim %d exceeds maximum configured bucket %d", dim, bm.buckets[len(bm.buckets)-1])
	}

	if exec, exists := bm.execs[targetBucket]; exists {
		bm.mu.Unlock()
		return exec, targetBucket, nil
	}
	if build := bm.builds[targetBucket]; build != nil {
		bm.mu.Unlock()
		<-build.done
		return build.exec, targetBucket, build.err
	}
	build := &bucketBuild{done: make(chan struct{})}
	bm.builds[targetBucket] = build
	bm.mu.Unlock()

	g, err := bm.builder(s, targetBucket)
	if err != nil {
		err = fmt.Errorf("cudagraph: build graph for bucket %d: %w", targetBucket, err)
	} else if g == nil {
		err = fmt.Errorf("cudagraph: builder returned nil graph for bucket %d", targetBucket)
	}
	var exec *Exec
	if err == nil {
		exec, err = g.Instantiate()
		if err != nil {
			err = fmt.Errorf("cudagraph: instantiate bucket %d: %w", targetBucket, err)
		}
		if destroyErr := g.Destroy(); destroyErr != nil {
			if exec != nil {
				_ = exec.Destroy()
				exec = nil
			}
			err = errors.Join(err, fmt.Errorf("cudagraph: destroy source graph for bucket %d: %w", targetBucket, destroyErr))
		}
	}

	bm.mu.Lock()
	if bm.closed && err == nil {
		err = errors.New("cudagraph: bucket manager closed")
	}
	if err == nil {
		bm.execs[targetBucket] = exec
	}
	build.exec, build.err = exec, err
	bm.mu.Unlock()
	if err != nil && exec != nil {
		_ = exec.Destroy()
		build.exec = nil
	}
	bm.mu.Lock()
	delete(bm.builds, targetBucket)
	close(build.done)
	bm.mu.Unlock()
	return build.exec, targetBucket, build.err
}

// Close destroys all cached instantiated graphs.
func (bm *BucketManager) Close() error {
	if bm == nil {
		return nil
	}
	bm.mu.Lock()
	if bm.closed {
		done := bm.close
		bm.mu.Unlock()
		<-done
		return bm.closeErr
	}
	bm.closed = true
	builds := make([]<-chan struct{}, 0, len(bm.builds))
	for _, build := range bm.builds {
		builds = append(builds, build.done)
	}
	execs := bm.execs
	bm.execs = make(map[int]*Exec)
	bm.mu.Unlock()

	for _, done := range builds {
		<-done
	}

	var errs []error
	for _, exec := range execs {
		if err := exec.Destroy(); err != nil {
			errs = append(errs, err)
		}
	}
	err := errors.Join(errs...)
	bm.mu.Lock()
	bm.closeErr = err
	close(bm.close)
	bm.mu.Unlock()
	return err
}

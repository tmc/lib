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
	raw    driver.Graph
	device *cuda.Device
}

// New creates an empty CUDA graph.
func New(dev ...*cuda.Device) (*Graph, error) {
	var d *cuda.Device
	if len(dev) > 0 {
		d = dev[0]
	}
	var raw driver.Graph
	create := func() error {
		var err error
		raw, err = driver.NewGraph(0)
		return err
	}
	var err error
	if d != nil {
		err = d.WithContext(create)
	} else {
		err = create()
	}
	if err != nil {
		return nil, fmt.Errorf("cudagraph: create graph: %w", err)
	}
	return &Graph{raw: raw, device: d}, nil
}

// Capture executes fn on stream while recording all enqueued operations into a Graph.
// The caller must not destroy stream until Capture returns.
func Capture(s *cuda.Stream, fn func()) (*Graph, error) {
	if s == nil {
		return nil, errors.New("cudagraph: nil stream passed to Capture")
	}
	if fn == nil {
		return nil, errors.New("cudagraph: nil capture function")
	}

	if err := s.Device().WithContext(func() error {
		return s.Raw().BeginCapture(driver.StreamCaptureModeGlobal)
	}); err != nil {
		return nil, fmt.Errorf("cudagraph: begin capture: %w", err)
	}

	fn()

	var rawGraph driver.Graph
	err := s.Device().WithContext(func() error {
		var err error
		rawGraph, err = s.Raw().EndCapture()
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cudagraph: end capture: %w", err)
	}

	return &Graph{raw: rawGraph, device: s.Device()}, nil
}

// Instantiate validates and instantiates the graph for execution on the GPU.
func (g *Graph) Instantiate() (*Exec, error) {
	return g.InstantiateWithOptions(InstantiateOptions{})
}

// InstantiateOptions controls graph instantiation. The zero value preserves
// ordinary host-launch behavior. DeviceLaunch enables device-side launch; it
// does not upload the graph or supply a device-side launcher. CUDA validates
// graph contents and device support during instantiation.
type InstantiateOptions struct {
	// AutoFreeOnLaunch frees allocations left by a previous graph launch.
	// It cannot be combined with DeviceLaunch.
	AutoFreeOnLaunch bool
	DeviceLaunch     bool
	// UseNodePriority uses kernel-node priorities instead of stream priority.
	UseNodePriority bool
}

func (o InstantiateOptions) flags() (uint64, error) {
	if o.AutoFreeOnLaunch && o.DeviceLaunch {
		return 0, errors.New("cudagraph: auto-free and device launch are incompatible")
	}
	var flags uint64
	if o.AutoFreeOnLaunch {
		flags |= uint64(driver.GraphInstantiateFlagAutoFreeOnLaunch)
	}
	if o.DeviceLaunch {
		flags |= uint64(driver.GraphInstantiateFlagDeviceLaunch)
	}
	if o.UseNodePriority {
		flags |= uint64(driver.GraphInstantiateFlagUseNodePriority)
	}
	return flags, nil
}

// InstantiateWithOptions validates options and instantiates the graph. Upload
// is separate because the native upload-on-instantiation flag requires the
// parameter-based API and an upload stream, not cuGraphInstantiateWithFlags.
func (g *Graph) InstantiateWithOptions(options InstantiateOptions) (*Exec, error) {
	if g == nil || g.raw == 0 {
		return nil, errors.New("cudagraph: graph is closed")
	}
	flags, err := options.flags()
	if err != nil {
		return nil, err
	}
	var rawExec driver.GraphExec
	err = g.withContext(func() error {
		var err error
		rawExec, err = g.raw.Instantiate(flags)
		return err
	})
	if err != nil {
		return nil, fmt.Errorf("cudagraph: instantiate graph: %w", err)
	}
	return &Exec{raw: rawExec, device: g.device}, nil
}

// Raw returns the underlying driver.Graph handle.
func (g *Graph) Raw() driver.Graph {
	if g == nil {
		return 0
	}
	return g.raw
}

// Destroy destroys the uninstantiated graph.
func (g *Graph) Destroy() error {
	if g == nil || g.raw == 0 {
		return nil
	}
	if err := g.withContext(g.raw.Destroy); err != nil {
		return fmt.Errorf("cudagraph: destroy graph: %w", err)
	}
	g.raw = 0
	return nil
}

// Exec represents an instantiated, executable CUDA graph.
type Exec struct {
	raw    driver.GraphExec
	device *cuda.Device
}

// Launch enqueues execution of the instantiated graph on stream.
func (e *Exec) Launch(s *cuda.Stream) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	launch := func(stream driver.Stream) error { return e.raw.GraphLaunch(stream) }
	var err error
	if s == nil {
		err = e.withContext(func() error { return launch(0) })
	} else {
		dev := s.Device()
		err = s.WithHandle(func(stream driver.Stream) error {
			return dev.WithContext(func() error { return launch(stream) })
		})
	}
	if err != nil {
		return fmt.Errorf("cudagraph: launch graph: %w", err)
	}
	return nil
}

// Update updates this executable graph from a topologically identical graph.
// CUDA checks the graph topology, but does not check that kernel nodes use the
// same kernel function, parameters, or shared-memory size. Callers must ensure
// those properties themselves.
func (e *Exec) Update(g *Graph) error {
	if e == nil || e.raw == 0 {
		return errors.New("cudagraph: executable graph is closed")
	}
	if g == nil {
		return errors.New("cudagraph: nil graph passed to Update")
	}
	if g.raw == 0 {
		return errors.New("cudagraph: graph is closed")
	}
	var result driver.GraphExecUpdateResultInfo
	if err := e.withContext(func() error { return e.raw.Update(g.raw, &result) }); err != nil {
		return fmt.Errorf("cudagraph: update graph (result=%v%s): %w", result.Result, updateErrorNodes(result), err)
	}
	return nil
}

// updateErrorNodes formats the nodes a failed update blamed, for appending to
// the error. The driver leaves them zero when the failure is not a node's, as a
// topology change is not, so they are named only when it set them.
//
// The handles are printed because they are matchable: [Node.Raw] returns the
// same value, so a caller can tell which of the nodes it added was rejected.
func updateErrorNodes(info driver.GraphExecUpdateResultInfo) string {
	switch {
	case info.ErrorNode == 0 && info.ErrorFromNode == 0:
		return ""
	case info.ErrorFromNode == 0:
		return fmt.Sprintf(", node=%#x", uintptr(info.ErrorNode))
	case info.ErrorNode == 0:
		return fmt.Sprintf(", from node=%#x", uintptr(info.ErrorFromNode))
	}
	return fmt.Sprintf(", node=%#x, from node=%#x", uintptr(info.ErrorNode), uintptr(info.ErrorFromNode))
}

// Destroy destroys the executable graph.
func (e *Exec) Destroy() error {
	if e == nil || e.raw == 0 {
		return nil
	}
	if err := e.withContext(e.raw.Destroy); err != nil {
		return fmt.Errorf("cudagraph: destroy exec graph: %w", err)
	}
	e.raw = 0
	return nil
}

func (g *Graph) withContext(fn func() error) error {
	if g.device == nil {
		return fn()
	}
	return g.device.WithContext(fn)
}

func (e *Exec) withContext(fn func() error) error {
	if e.device == nil {
		return fn()
	}
	return e.device.WithContext(fn)
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

package cudamega_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
)

// The example uses one cleanup stack, not independent defers. Keep the stack
// after an error: the failed closer and every backing owner remain available
// for retry. Already completed closes are removed and are not repeated.
type moeExampleCleanup struct {
	steps []moeExampleClose
}

type moeExampleClose struct {
	name  string
	close func() error
}

func (c *moeExampleCleanup) add(name string, close func() error) {
	c.steps = append(c.steps, moeExampleClose{name, close})
}

func (c *moeExampleCleanup) close() error {
	for len(c.steps) > 0 {
		s := c.steps[len(c.steps)-1]
		if err := s.close(); err != nil {
			return fmt.Errorf("close %s: %w", s.name, err)
		}
		c.steps = c.steps[:len(c.steps)-1]
	}
	return nil
}

func (c *moeExampleCleanup) remaining() string {
	names := make([]string, 0, len(c.steps))
	for i := len(c.steps) - 1; i >= 0; i-- {
		names = append(names, c.steps[i].name)
	}
	return strings.Join(names, ", ")
}

// A real application returns this retry owner to its caller. The optional
// example retains it here because Example functions have no return value.
// Retry owner.close only after resolving the reported drain/cleanup error.
var unresolvedMoEExamples []*moeExampleCleanup

func ExampleMoEWeightSource() {
	s, err := cudamega.MoEWeightSource(cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17}, "sm_121")
	fmt.Println(s != "", err) // Output: true <nil>
}

// This optional hardware example uses a compiled healthy source-bound packet.
// Run it only under gpu-lock with CUDAMEGA_EXAMPLE_ARTIFACT set to its artifact
// JSON. The default example executes no device work. Failures are reported as
// errors; none are accepted as successful native execution.
func ExampleMoEWeightConsumer_Submit() {
	fmt.Println("hardware setup is opt-in")
	path := os.Getenv("CUDAMEGA_EXAMPLE_ARTIFACT")
	if path == "" {
		return
	}
	lockPID, lockErr := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if lockErr != nil || lockPID <= 0 {
		fmt.Println("gpu-lock is required")
		return
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	raw, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	err = errors.Join(err, f.Close())
	if err != nil || len(raw) > 16<<20 {
		fmt.Println("artifact read failed")
		return
	}
	var artifact cudakernel.Artifact
	if err := json.Unmarshal(raw, &artifact); err != nil {
		fmt.Println(err)
		return
	}
	if err := driver.Load(""); err != nil {
		fmt.Println(err)
		return
	}
	cleanup := &moeExampleCleanup{}
	defer func() {
		if err := cleanup.close(); err != nil {
			// In particular, failed consumer drain stops before any buffer,
			// public view, storage, stream or device closer can run.
			unresolvedMoEExamples = append(unresolvedMoEExamples, cleanup)
			fmt.Println("unresolved owners retained for cleanup retry:", cleanup.remaining(), err)
		}
	}()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("device", dev.Close)
	stream, err := dev.NewStream()
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("stream", stream.Close)
	storage, err := cudamoe.NewMoEWeightStorage(dev, cudamoe.MoEWeightShape{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17, ActiveLayers: []int{0, 2}})
	if storage != nil {
		cleanup.add("storage", storage.Close)
	}
	if err != nil {
		fmt.Println(err)
		return
	}
	weights := make([]byte, 2*16*17)
	scales := make([]byte, 2*16*4)
	for i := 0; i < 2*16; i++ {
		binary.LittleEndian.PutUint32(scales[i*4:], 0x3f800000)
	}
	if err := storage.Load(2, 1, bytes.NewReader(weights), bytes.NewReader(scales)); err != nil {
		fmt.Println(err)
		return
	}
	view, err := storage.View(2, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("view", view.Close)
	input, err := cudamem.Alloc[float32](dev, 17)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("input", input.Free)
	output, err := cudamem.Alloc[float32](dev, 16)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("output", output.Free)
	completed, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("completed", completed.Free)
	abort, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("abort", abort.Free)
	order, err := cudamem.Alloc[uint64](dev, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	cleanup.add("order", order.Free)
	consumer, err := cudamega.LoadMoEWeightConsumer(dev, cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17}, &artifact)
	// Add every nonnil constructor result, including errors. Last acquired
	// means first closed: drain succeeds before any backing cleanup proceeds.
	if consumer != nil {
		cleanup.add("consumer", consumer.Close)
	}
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := input.CopyFromHost(make([]float32, 17)); err != nil {
		fmt.Println(err)
		return
	}
	b := cudamega.MoEWeightBuffers{Input: input, Output: output, Completed: completed, Abort: abort, StartOrder: order}
	if err := consumer.Submit(stream, view, b, 1); err != nil {
		fmt.Println(err)
		return
	}
	if err := consumer.Wait(context.Background()); err != nil {
		fmt.Println(err)
		return
	}
	done, aborted, started := make([]uint32, 1), make([]uint32, 1), make([]uint64, 2)
	for _, read := range []func() error{func() error { return completed.CopyToHost(done) }, func() error { return abort.CopyToHost(aborted) }, func() error { return order.CopyToHost(started) }} {
		if err := read(); err != nil {
			fmt.Println(err)
			return
		}
	}
	if done[0] != 1 || aborted[0] != 0 || started[0] != 1 || started[1] != 1 {
		fmt.Println("publication incomplete")
		return
	}
	// Execute also admits completion publication before returning. A Wait
	// caller checks publication and numerical output explicitly.
	if err := consumer.Execute(context.Background(), stream, view, b, 2); err != nil {
		fmt.Println(err)
		return
	}
	actual := make([]float32, 16)
	if err := output.CopyToHost(actual); err != nil {
		fmt.Println(err)
		return
	}
	for _, v := range actual {
		if v != 0 {
			fmt.Println("zero-weight oracle failed")
			return
		}
	}
	// Output: hardware setup is opt-in
}
func ExampleMoEWeightSchema() {
	s, err := cudamega.MoEWeightSchema()
	fmt.Println(len(s.Params), s.TotalParamBytes, err) // Output: 11 80 <nil>
}

func ExampleLoadMoEWeightConsumer() {
	config := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17}
	source, err := cudamega.MoEWeightSource(config, "sm_121")
	fmt.Println(source != "", err)
	// Compile source to a checked CUBIN before calling the loader. A missing
	// artifact is a refusal, not a successful device-free execution.
	consumer, err := cudamega.LoadMoEWeightConsumer(nil, config, nil)
	if consumer != nil {
		defer consumer.Close()
	}
	fmt.Println(consumer == nil, err != nil)
	// Output:
	// true <nil>
	// true true
}
func ExampleMoEWeightConsumer_Wait() {
	var c *cudamega.MoEWeightConsumer
	fmt.Println(c.Wait(nil), c.Close(), c.BlocksPerSM()) // Output: <nil> <nil> 0
}

func TestMoEWeightExampleCleanup(t *testing.T) {
	for _, tt := range []struct {
		name, fail, first, remaining, retry string
		acquired                            []string
	}{
		{"success", "", "consumer,input,view,storage,stream,device", "", "", []string{"device", "stream", "storage", "view", "input", "consumer"}},
		{"failed-drain", "consumer", "consumer", "consumer, input, view, storage, stream, device", "consumer,input,view,storage,stream,device", []string{"device", "stream", "storage", "view", "input", "consumer"}},
		{"failed-stream", "stream", "consumer,input,view,storage,stream", "stream, device", "stream,device", []string{"device", "stream", "storage", "view", "input", "consumer"}},
		{"acquisition-failed-before-consumer", "", "input,view,storage,stream,device", "", "", []string{"device", "stream", "storage", "view", "input"}},
		{"partial-constructor-owner", "consumer", "consumer", "consumer, storage, stream, device", "consumer,storage,stream,device", []string{"device", "stream", "storage", "consumer"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cleanup := &moeExampleCleanup{}
			var calls []string
			cause := errors.New("injected cleanup failure")
			fail := tt.fail
			for _, name := range tt.acquired {
				cleanup.add(name, func() error {
					calls = append(calls, name)
					if name == fail {
						fail = ""
						return cause
					}
					return nil
				})
			}
			err := cleanup.close()
			if (tt.fail != "") != errors.Is(err, cause) {
				t.Fatal("cleanup error lost", err)
			}
			if got := strings.Join(calls, ","); got != tt.first {
				t.Fatalf("first closes %q want %q", got, tt.first)
			}
			if got := cleanup.remaining(); got != tt.remaining {
				t.Fatalf("unresolved %q want %q", got, tt.remaining)
			}
			calls = nil
			if err := cleanup.close(); err != nil {
				t.Fatal("retry failed", err)
			}
			if got := strings.Join(calls, ","); got != tt.retry {
				t.Fatalf("retry closes %q want %q", got, tt.retry)
			}
			calls = nil
			if err := cleanup.close(); err != nil || len(calls) != 0 || cleanup.remaining() != "" {
				t.Fatal("completed owners closed again", err, calls)
			}
		})
	}
}

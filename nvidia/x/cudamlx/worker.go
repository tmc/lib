package cudamlx

import (
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// Worker runs completion handlers on a background goroutine after the
// GPU work they follow has finished. Port of worker.cpp; instead of
// cudaLaunchHostFunc on a signal stream, each committed batch carries a
// recorded event that the goroutine synchronizes on.
type Worker struct {
	dev *Device

	mu      sync.Mutex
	cond    *sync.Cond
	stopped bool
	batches []batch

	pending []func()
}

type batch struct {
	event driver.Event
	tasks []func()
}

// NewWorker creates and starts a worker for the device.
func NewWorker(dev *Device) *Worker {
	w := &Worker{dev: dev}
	w.cond = sync.NewCond(&w.mu)
	go w.run()
	return w
}

// AddTask queues a pending task that runs after the next Commit's GPU
// work completes. Not safe for concurrent use; call from the goroutine
// that owns the encoder, as in the C++.
func (w *Worker) AddTask(task func()) {
	w.pending = append(w.pending, task)
}

// Commit moves pending tasks into a batch that runs once the kernels
// currently queued on stream finish.
func (w *Worker) Commit(stream driver.Stream) error {
	if len(w.pending) == 0 {
		return nil
	}
	var ev driver.Event
	err := w.dev.WithCurrent(func() error {
		var err error
		ev, err = driver.NewEvent(uint32(driver.EventDisableTiming | driver.EventBlockingSync))
		if err != nil {
			return err
		}
		return ev.Record(stream)
	})
	if err != nil {
		return err
	}
	w.mu.Lock()
	w.batches = append(w.batches, batch{event: ev, tasks: w.pending})
	w.pending = nil
	w.mu.Unlock()
	w.cond.Signal()
	return nil
}

// Stop ends the worker after all committed batches have run.
func (w *Worker) Stop() {
	w.mu.Lock()
	w.stopped = true
	w.mu.Unlock()
	w.cond.Signal()
}

func (w *Worker) run() {
	w.dev.MakeCurrent()
	for {
		w.mu.Lock()
		for len(w.batches) == 0 && !w.stopped {
			w.cond.Wait()
		}
		if len(w.batches) == 0 && w.stopped {
			w.mu.Unlock()
			return
		}
		b := w.batches[0]
		w.batches = w.batches[1:]
		w.mu.Unlock()

		b.event.Synchronize()
		b.event.Destroy()
		for _, task := range b.tasks {
			task()
		}
	}
}

package cudamegacpu

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestPublicationWaitsForProducerThreads(t *testing.T) {
	for _, scope := range []cudataskgraph.Scope{cudataskgraph.ScopeDevice, cudataskgraph.ScopeSystem} {
		t.Run(fmt.Sprint(scope), func(t *testing.T) {
			g, err := cudataskgraph.New([]cudataskgraph.Task{
				{ID: "independent", Rank: 0, Scope: scope},
				{ID: "producer", Rank: 0, Scope: scope},
				{ID: "consumer", Rank: 1, Deps: []string{"producer"}, Scope: scope},
				{ID: "other", Rank: 1, Deps: []string{"independent"}, Scope: scope},
			})
			if err != nil {
				t.Fatal(err)
			}
			ops := map[string]cudamega.Operation{
				"independent": {Kind: cudamega.Add, Scalar: 3},
				"producer":    {Kind: cudamega.Add, Scalar: 5},
				"consumer":    {Kind: cudamega.Multiply, Scalar: 7},
				"other":       {Kind: cudamega.Multiply, Scalar: 11},
			}
			p, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 2, ThreadsPerCTA: 32, Policy: cudamega.CountedPolicy}, cudamega.SM90)
			if err != nil {
				t.Fatal(err)
			}
			e, err := New(cudamega.Admission{ResidentCTAs: 2, WorkerThreads: 32, SharedBytes: 1})
			if err != nil {
				t.Fatal(err)
			}
			probe := &publicationProbe{held: make(chan struct{}), independent: make(chan struct{}), consumer: make(chan struct{}), release: make(chan struct{})}
			e.wrap = func(b cudamega.Backend) cudamega.Backend {
				return &publicationWorker{worker: b.(*worker), probe: probe}
			}
			control := make(chan error, 1)
			go func() {
				defer close(probe.release)
				for _, event := range []<-chan struct{}{probe.held, probe.independent} {
					select {
					case <-event:
					case <-time.After(2 * time.Second):
						control <- fmt.Errorf("independent work failed to progress while producer was held")
						return
					}
				}
				select {
				case <-probe.consumer:
					control <- fmt.Errorf("consumer read producer output before its second thread stored")
				case <-time.After(30 * time.Millisecond):
					control <- nil
				}
			}()
			input := []uint32{2, 13, 29, 41, 53, 67, 79, 97}
			got, err := e.Run(p, input, 1, 0)
			controlErr := <-control
			if err != nil {
				t.Fatal(err)
			}
			want, err := Reference(p, input)
			if err != nil {
				t.Fatal(err)
			}
			if controlErr != nil {
				t.Error(controlErr)
			}
			if got.Abort != 0 || !reflect.DeepEqual(got.Scratch, want) {
				t.Fatalf("abort=%d; scratch=%v, reference=%v", got.Abort, got.Scratch, want)
			}
		})
	}
}

type publicationProbe struct {
	held, independent, consumer, release chan struct{}
	once                                 sync.Once
}

type publicationWorker struct {
	*worker
	probe *publicationProbe
}

func (w *publicationWorker) Compute(i, src int, op cudamega.Operation) error {
	if i == 1 && w.tid == 1 {
		close(w.probe.held)
		<-w.probe.release
	}
	if err := w.worker.Compute(i, src, op); err != nil {
		return err
	}
	if i == 2 && w.tid == 1 {
		w.probe.once.Do(func() { close(w.probe.consumer) })
	}
	return nil
}

func (w *publicationWorker) Publish(i int) error {
	if err := w.worker.Publish(i); err != nil {
		return err
	}
	if i == 0 && w.tid == 0 {
		close(w.probe.independent)
	}
	return nil
}

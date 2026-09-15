package cuptirange

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

type kernelGraph struct {
	symbols  []string
	validate func() (driver.GraphExec, []string, error)
}

func (k *kernelNames) configureGraphs(graphs []*cudagraph.Exec, nativeReplay bool) error {
	k.nativeKernelReplay = nativeReplay
	if len(graphs) == 0 {
		return nil
	}
	if !nativeReplay {
		// CUDA 13.2 release notes 1.2.1.1 explicitly exclude AutoRange with user
		// replay for graphs under both Range Profiling and legacy Profiling APIs:
		// https://docs.nvidia.com/cupti/13.2.0/release-notes/release-notes.html#range-profiling-api-and-profiling-api
		return errors.New("cuptirange: graph kernel identity requires native kernel replay")
	}
	k.graphs = make(map[driver.GraphExec]kernelGraph)
	for _, graph := range graphs {
		handle, symbols, err := graph.KernelOrder()
		if err != nil {
			return fmt.Errorf("cuptirange: graph identity proof: %w", err)
		}
		if handle == 0 || len(symbols) == 0 {
			return errors.New("cuptirange: empty graph identity proof")
		}
		if _, ok := k.graphs[handle]; ok {
			return errors.New("cuptirange: duplicate graph identity proof")
		}
		if uint(len(symbols)) > k.limit {
			return errors.New("cuptirange: graph kernel identity capacity exceeded")
		}
		k.graphs[handle] = kernelGraph{symbols: append([]string(nil), symbols...), validate: graph.KernelOrder}
	}
	return nil
}

func (k *kernelNames) checkGraphs() error {
	if k.graphFailure != nil {
		return k.graphFailure
	}
	for handle, graph := range k.graphs {
		current, symbols, err := graph.validate()
		if err != nil {
			return fmt.Errorf("cuptirange: graph identity proof invalidated: %w", err)
		}
		if current != handle || !reflect.DeepEqual(symbols, graph.symbols) {
			return errors.New("cuptirange: graph identity proof changed")
		}
	}
	return nil
}

// GraphLaunch and the graph-exec mutation APIs take CUgraphExec as their first
// argument. Read it only while the native callback owns FunctionParams. The
// retained callback layout gate establishes this field's offset and pointer ABI.
func graphExecArgument(evt *cupticallback.Event) (driver.GraphExec, error) {
	if evt.Raw == nil {
		return 0, errors.New("cuptirange: graph callback has no native arguments")
	}
	params := *(*unsafe.Pointer)(unsafe.Add(evt.Raw, unsafe.Offsetof(cupti.CallbackData{}.FunctionParams)))
	if params == nil {
		return 0, errors.New("cuptirange: graph callback has no native parameters")
	}
	handle := *(*driver.GraphExec)(params)
	if handle == 0 {
		return 0, errors.New("cuptirange: graph callback has null executable")
	}
	return handle, nil
}

func graphMutation(name string) bool {
	return strings.HasPrefix(name, "cuGraphExec") && !strings.HasPrefix(name, "cuGraphExecGet") || strings.HasPrefix(name, "cuGraphNodeSetEnabled")
}

func (k *kernelNames) observeGraphMutation(evt *cupticallback.Event) {
	if len(k.graphs) == 0 {
		return
	}
	handle, err := graphExecArgument(evt)
	if err != nil {
		k.graphFailure = err
		return
	}
	if _, ok := k.graphs[handle]; ok {
		k.graphFailure = errors.New("cuptirange: native graph executable mutated during identity collection")
	}
}

func (k *kernelNames) appendGraph(evt *cupticallback.Event) {
	if !k.nativeKernelReplay {
		k.failure = errors.New("cuptirange: graph kernel identity requires native kernel replay")
		return
	}
	handle, err := graphExecArgument(evt)
	if err != nil {
		k.failure = err
		return
	}
	graph, ok := k.graphs[handle]
	if !ok {
		k.failure = errors.New("cuptirange: graph kernel identity has no registered native order proof")
		return
	}
	if uint(len(graph.symbols)) > k.limit-uint(len(k.current)) {
		k.failure = errors.New("cuptirange: graph kernel identity capacity exceeded")
		return
	}
	for _, symbol := range graph.symbols {
		k.current = append(k.current, kernelLaunch{API: evt.Api.FunctionName, Symbol: symbol, Graph: handle})
	}
}

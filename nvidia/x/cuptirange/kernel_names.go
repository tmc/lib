package cuptirange

import (
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

// AutoRange native names are launch ordinals, not CUDA symbols. NVIDIA's
// callback_profiling guidance maps same-context launch sequence to range index:
// https://forums.developer.nvidia.com/t/associating-kernel-names-with-range-id/308365
// The window below excludes preparation, counts every direct launch, and checks
// the sequence again on every replay. It adds no CUDA work or synchronization.
type kernelNames struct {
	mu                 sync.Mutex
	context            cupti.Context
	limit              uint
	active             bool
	current            []kernelLaunch
	expected           []kernelLaunch
	passes             uint
	failure            error
	nativeKernelReplay bool
	graphs             map[driver.GraphExec]kernelGraph
	graphFailure       error
	subscription       interface{ Unsubscribe() error }
}

type kernelLaunch struct {
	API, Symbol string
	Graph       driver.GraphExec
}

func newKernelNames(context cupti.Context, limit uint) (*kernelNames, error) {
	k := &kernelNames{context: context, limit: limit}
	sub, err := cupticallback.Subscribe(cupticallback.Options{AllowReentrant: true}, cupticallback.HandlerFunc(k.callback))
	if err != nil {
		return nil, err
	}
	k.subscription = sub
	if err := sub.EnableDomain(cupti.CbDomainDriverApi, true); err != nil {
		return k, err
	}
	return k, nil
}

func (k *kernelNames) Close() error {
	if k == nil || k.subscription == nil {
		return nil
	}
	if err := k.subscription.Unsubscribe(); err != nil {
		return err
	}
	k.subscription = nil
	return nil
}

func (k *kernelNames) begin() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.active {
		return errors.New("cuptirange: kernel identity window already active")
	}
	if err := k.checkGraphs(); err != nil {
		return err
	}
	k.active = true
	k.current = nil
	k.failure = nil
	return nil
}

func (k *kernelNames) end() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.active = false
	if err := k.checkGraphs(); err != nil {
		return err
	}
	if k.failure != nil {
		return k.failure
	}
	if k.passes == 0 {
		k.expected = append([]kernelLaunch(nil), k.current...)
	} else if !reflect.DeepEqual(k.expected, k.current) {
		return errors.New("cuptirange: kernel launch replay trace changed")
	}
	k.passes++
	return nil
}

func (k *kernelNames) callback(evt *cupticallback.Event) {
	if evt == nil || evt.Domain != cupti.CbDomainDriverApi || evt.Api == nil {
		return
	}
	a := evt.Api
	if a.CallbackSite != cupti.ApiEnter {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if graphMutation(a.FunctionName) {
		k.observeGraphMutation(evt)
		return
	}
	if a.Context != k.context {
		return
	}
	if !k.active || k.failure != nil {
		return
	}
	if strings.HasPrefix(a.FunctionName, "cuGraphLaunch") {
		k.appendGraph(evt)
		return
	}
	switch cupti.DriverTraceCBID(evt.ID) {
	case cupti.DriverCBIDCuLaunch, cupti.DriverCBIDCuLaunchGrid, cupti.DriverCBIDCuLaunchGridAsync,
		cupti.DriverCBIDCuLaunchKernel, cupti.DriverCBIDCuLaunchKernelPtsz,
		cupti.DriverCBIDCuLaunchKernelEx, cupti.DriverCBIDCuLaunchKernelExPtsz,
		cupti.DriverCBIDCuLaunchCooperativeKernel, cupti.DriverCBIDCuLaunchCooperativeKernelPtsz:
	default:
		return
	}
	if a.SymbolName == "" {
		k.failure = errors.New("cuptirange: kernel launch callback has no symbol name")
		return
	}
	if uint(len(k.current)) >= k.limit {
		k.failure = errors.New("cuptirange: kernel launch identity capacity exceeded")
		return
	}
	k.current = append(k.current, kernelLaunch{API: a.FunctionName, Symbol: a.SymbolName})
}

// checkCount also protects captures that do not request enriched range records.
func (k *kernelNames) checkCount(count uint) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.checkCountLocked(count)
}

func (k *kernelNames) checkCountLocked(count uint) error {
	if k.active || k.passes == 0 || k.failure != nil {
		return errors.New("cuptirange: incomplete kernel identity trace")
	}
	if err := k.checkGraphs(); err != nil {
		return err
	}
	if uint(len(k.expected)) != count {
		return fmt.Errorf("cuptirange: kernel identity count %d differs from native range count %d", len(k.expected), count)
	}
	return nil
}

func (k *kernelNames) resolve(records []RangeInfo) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if err := k.checkCountLocked(uint(len(records))); err != nil {
		return err
	}
	seen := make(map[uint64]bool)
	for i := range records {
		ordinal, err := strconv.ParseUint(records[i].Name, 10, 64)
		if err != nil || ordinal >= uint64(len(k.expected)) || seen[ordinal] {
			return fmt.Errorf("cuptirange: invalid native kernel launch ordinal %q", records[i].Name)
		}
		seen[ordinal] = true
		records[i].NativeName = records[i].Name
		records[i].Name = k.expected[ordinal].Symbol
	}
	return nil
}

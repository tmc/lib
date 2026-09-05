package cuptirange

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

// metricNames owns the native char** table and every NUL-terminated name it
// references. Keep it pinned for the whole host-profiler object lifetime, not
// merely the ConfigAddMetrics call, so no native retention assumption is needed.
type metricNames struct {
	data     [][]byte
	pointers []uintptr
	pin      runtime.Pinner
}

func newMetricNames(names []string) (*metricNames, error) {
	if len(names) == 0 {
		return nil, errors.New("cuptirange: no metrics")
	}
	seen := make(map[string]bool)
	for _, name := range names {
		if name == "" || strings.ContainsRune(name, 0) {
			return nil, errors.New("cuptirange: invalid metric name")
		}
		if seen[name] {
			return nil, fmt.Errorf("cuptirange: duplicate metric %q", name)
		}
		seen[name] = true
	}
	n := &metricNames{data: make([][]byte, len(names)), pointers: make([]uintptr, len(names))}
	for i, name := range names {
		n.data[i] = append([]byte(name), 0)
		n.pin.Pin(&n.data[i][0])
		n.pointers[i] = uintptr(unsafe.Pointer(&n.data[i][0]))
	}
	n.pin.Pin(&n.pointers[0])
	return n, nil
}

func (n *metricNames) close() {
	n.pin.Unpin()
	n.data, n.pointers = nil, nil
}

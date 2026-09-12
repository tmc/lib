package cuptiactivity

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
)

// Config bounds collection memory. Kinds must be nonempty and contain only
// driver/runtime API, concurrent kernel, memcpy, memset, memory operation,
// synchronization, or external-correlation
// activities. External-correlation records require an instrumented producer
// that pushes and pops external IDs on the thread issuing CUDA calls.
type Config struct {
	Kinds       []cupti.ActivityKind
	BufferBytes int
	MaxBuffers  int
	MaxRecords  int
}

var nativeSessions struct {
	sync.Mutex
	active            *collection
	request, complete uintptr
	registered        bool
	callbacks         int
}

// Start starts process-wide CUPTI activity collection. The caller must own
// exclusive use of CUPTI activity registration and enable/disable APIs in this
// process, including other profiling libraries. This package cannot detect an
// external registrant. Only one Session can be active at a time.
//
// All Config bounds must be positive; BufferBytes must be a multiple of eight.
// A nonnil Session returned with an error retains cleanup ownership: call Close
// again. Only the locally qualified CUPTI record version is accepted.
func Start(config Config) (*Session, error) {
	if config.MaxRecords <= 0 {
		return nil, errors.New("cuptiactivity: record limit must be positive")
	}
	pool, err := newBufferPool(config.BufferBytes, config.MaxBuffers)
	if err != nil {
		return nil, err
	}
	version, err := cupti.GetVersion()
	if err != nil {
		return nil, err
	}
	layouts, err := recordLayouts(version, cupti.ActivityGetStructSize)
	if err != nil {
		return nil, err
	}
	if err := validateKinds(config.Kinds, layouts); err != nil {
		return nil, err
	}
	c := &collection{pool: pool, layouts: layouts, next: cupti.ActivityGetNextRecord,
		name: nativeName, dropped: cupti.ActivityGetNumDroppedRecords, limit: config.MaxRecords}
	nativeSessions.Lock()
	if nativeSessions.active != nil {
		nativeSessions.Unlock()
		return nil, errors.New("cuptiactivity: another session owns activity collection")
	}
	if nativeSessions.request == 0 {
		nativeSessions.request = purego.NewCallback(requestBuffer)
		nativeSessions.complete = purego.NewCallback(completeBuffer)
	}
	// Registration itself must not run under the callback lock.
	nativeSessions.active = c
	registered := nativeSessions.registered
	nativeSessions.Unlock()
	release := func() {
		nativeSessions.Lock()
		if nativeSessions.active == c {
			nativeSessions.active = nil
		}
		nativeSessions.Unlock()
	}
	if !registered {
		if err := cupti.ActivityRegisterCallbacks(nativeSessions.request, nativeSessions.complete); err != nil {
			release()
			return nil, err
		}
		nativeSessions.Lock()
		nativeSessions.registered = true
		nativeSessions.Unlock()
	}
	o, err := startActivities(config.Kinds, layouts, pool, cupti.ActivityEnable, cupti.ActivityDisable,
		func() error { return cupti.ActivityFlushAll(uint32(cupti.ActivityFlagFlushForced)) }, release, func() error {
			nativeSessions.Lock()
			defer nativeSessions.Unlock()
			pending, _ := pool.status()
			if nativeSessions.callbacks != 0 || pending != 0 {
				return errors.New("cuptiactivity: callbacks or buffers still active")
			}
			if nativeSessions.active == c {
				nativeSessions.active = nil
			}
			return nil
		})
	if o == nil {
		// Validation can fail before startActivities acquires ownership.
		nativeSessions.Lock()
		if nativeSessions.active == c {
			nativeSessions.active = nil
		}
		nativeSessions.Unlock()
		return nil, err
	}
	return &Session{owner: o, collection: c}, err
}

func requestBuffer(buffer *uintptr, size, maxRecords *uint) {
	*buffer, *size, *maxRecords = 0, 0, 0
	nativeSessions.Lock()
	defer nativeSessions.Unlock()
	if c := nativeSessions.active; c != nil {
		*buffer, *size = c.pool.request()
	}
}

func completeBuffer(context cupti.Context, stream uint32, buffer uintptr, size, valid uint) {
	nativeSessions.Lock()
	c := nativeSessions.active
	if c != nil {
		nativeSessions.callbacks++
	}
	nativeSessions.Unlock()
	if c != nil {
		defer func() { nativeSessions.Lock(); nativeSessions.callbacks--; nativeSessions.Unlock() }()
		c.complete(context, stream, buffer, size, valid)
	}
}

// nativeName copies a native NUL-terminated string while its CUPTI record is
// borrowed. The uintptr is a C pointer field, never an address of unpinned Go
// memory. Interpret its pointer representation without uintptr arithmetic.
func nativeName(address uintptr) (string, error) {
	if address == 0 {
		return "", nil
	}
	p := *(*unsafe.Pointer)(unsafe.Pointer(&address))
	var data []byte
	for i := 0; i < 1<<20; i++ {
		b := *(*byte)(unsafe.Add(p, i))
		if b == 0 {
			return string(data), nil
		}
		data = append(data, b)
	}
	return "", errors.New("kernel name exceeds one MiB")
}

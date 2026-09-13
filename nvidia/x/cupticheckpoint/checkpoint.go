package cupticheckpoint

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Optimizations specifies performance optimization modes for checkpoint save and restore.
type Optimizations uint8

const (
	// OptNone is the default behavior.
	OptNone Optimizations = 0
	// OptTransfer caches which memory blocks have changed and only restores modified blocks.
	OptTransfer Optimizations = 1
)

func (o Optimizations) String() string {
	switch o {
	case OptNone:
		return "None"
	case OptTransfer:
		return "Transfer"
	default:
		return fmt.Sprintf("Optimizations(%d)", o)
	}
}

// Options configures device and host memory reservations for a checkpoint.
type Options struct {
	// ReserveDeviceMB reserves device memory in megabytes that will remain free
	// after saving the checkpoint.
	ReserveDeviceMB uint

	// ReserveHostMB reserves host memory in megabytes.
	ReserveHostMB uint

	// AllowOverwrite enables overwriting an existing saved checkpoint on the handle.
	AllowOverwrite bool

	// Optimizations configures optimization flags.
	Optimizations Optimizations
}

// Checkpoint owns a saved CUPTI checkpoint on a CUDA context.
// Callers must call Free when finished with the checkpoint to release device backing memory.
type Checkpoint struct {
	mu    sync.Mutex
	raw   cupti.Checkpoint
	ctx   uintptr
	opts  Options
	saved bool
	freed bool
}

// Save initializes and saves a device state checkpoint for the specified CUDA context.
func Save(ctx uintptr, opts Options) (*Checkpoint, error) {
	if err := CheckVersion(); err != nil {
		return nil, err
	}
	if err := loadLib(); err != nil {
		return nil, err
	}

	c := &Checkpoint{
		ctx:  ctx,
		opts: opts,
	}

	c.raw.StructSize = uint(unsafe.Offsetof(c.raw.PPriv) + unsafe.Sizeof(c.raw.PPriv))
	c.raw.Ctx = cupti.Context(ctx)
	c.raw.ReserveDeviceMB = opts.ReserveDeviceMB
	c.raw.ReserveHostMB = opts.ReserveHostMB
	if opts.AllowOverwrite {
		c.raw.AllowOverwrite = 1
	}
	c.raw.Optimizations = uint8(opts.Optimizations)

	if err := toError(fnCheckpointSave(&c.raw)); err != nil {
		return nil, fmt.Errorf("cupticheckpoint: save: %w", err)
	}
	c.saved = true
	return c, nil
}

// Restore restores the saved checkpoint to the device associated with its CUDA context.
func (c *Checkpoint) Restore() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.freed {
		return errors.New("cupticheckpoint: checkpoint is freed")
	}
	if !c.saved {
		return errors.New("cupticheckpoint: checkpoint is not saved")
	}

	if err := toError(fnCheckpointRestore(&c.raw)); err != nil {
		return fmt.Errorf("cupticheckpoint: restore: %w", err)
	}
	return nil
}

// Free releases backing storage associated with the saved checkpoint.
// Free is idempotent.
func (c *Checkpoint) Free() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.freed {
		return nil
	}
	c.freed = true

	if c.saved {
		if err := toError(fnCheckpointFree(&c.raw)); err != nil {
			return fmt.Errorf("cupticheckpoint: free: %w", err)
		}
		c.saved = false
	}
	return nil
}

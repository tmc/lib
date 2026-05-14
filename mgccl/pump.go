package mgccl

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/gopccl"
)

var (
	ErrInvalidConfig  = errors.New("mgccl: invalid config")
	ErrInvalidGrid    = errors.New("mgccl: invalid grid")
	ErrIncompleteGrid = errors.New("mgccl: incomplete grid")
	ErrStaleRevision  = errors.New("mgccl: stale revision")
	ErrIterationLimit = errors.New("mgccl: iteration limit reached")
)

// Island is the local collective island controlled by the accept pump.
type Island interface {
	PeersPending(context.Context) (bool, error)
	UpdateTopology(context.Context) error
	OptimizeTopology(context.Context) error
	SynchronizeState(context.Context, uint64, []gopccl.StateTensor, gopccl.SyncStrategy) (gopccl.SyncInfo, error)
}

// Rank identifies one cell in the PCCL by FSDP rank matrix.
type Rank struct {
	PCCL int
	FSDP int
}

// Grid describes the expected PCCL by FSDP rank matrix.
type Grid struct {
	PCCLRanks int
	FSDPRanks int
	Ranks     []Rank
}

// Validate checks whether g describes a fully populated rank matrix.
func (g Grid) Validate() error {
	if g.PCCLRanks <= 0 || g.FSDPRanks <= 0 {
		return fmt.Errorf("%w: dimensions must be positive", ErrInvalidGrid)
	}
	want := g.PCCLRanks * g.FSDPRanks
	if len(g.Ranks) != want {
		return fmt.Errorf("%w: got %d ranks, want %d", ErrIncompleteGrid, len(g.Ranks), want)
	}
	seen := make(map[Rank]bool, len(g.Ranks))
	for _, r := range g.Ranks {
		if r.PCCL < 0 || r.PCCL >= g.PCCLRanks {
			return fmt.Errorf("%w: pccl rank %d out of range", ErrInvalidGrid, r.PCCL)
		}
		if r.FSDP < 0 || r.FSDP >= g.FSDPRanks {
			return fmt.Errorf("%w: fsdp rank %d out of range", ErrInvalidGrid, r.FSDP)
		}
		if seen[r] {
			return fmt.Errorf("%w: duplicate rank pccl=%d fsdp=%d", ErrInvalidGrid, r.PCCL, r.FSDP)
		}
		seen[r] = true
	}
	return nil
}

// PumpConfig configures one accept-pump pass.
type PumpConfig struct {
	Grid          Grid
	Revision      uint64
	Tensors       []gopccl.StateTensor
	Strategy      gopccl.SyncStrategy
	PollInterval  time.Duration
	MaxIterations int
	Ready         func(context.Context) error
}

// Pump coordinates topology acceptance and state synchronization.
type Pump struct {
	lastRevision uint64
	haveRevision bool
}

// RunAcceptPump runs one accept-pump pass with a zero-value Pump.
func RunAcceptPump(ctx context.Context, island Island, cfg PumpConfig) error {
	var p Pump
	return p.Run(ctx, island, cfg)
}

// Run runs one accept-pump pass.
func (p *Pump) Run(ctx context.Context, island Island, cfg PumpConfig) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if island == nil {
		return fmt.Errorf("%w: nil island", ErrInvalidConfig)
	}
	if err := cfg.Grid.Validate(); err != nil {
		return fmt.Errorf("run accept pump: %w", err)
	}
	if p != nil && p.haveRevision && cfg.Revision <= p.lastRevision {
		return fmt.Errorf("%w: got %d, last %d", ErrStaleRevision, cfg.Revision, p.lastRevision)
	}

	for iter := 0; ; iter++ {
		if cfg.MaxIterations > 0 && iter >= cfg.MaxIterations {
			return ErrIterationLimit
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		pending, err := island.PeersPending(ctx)
		if err != nil {
			return fmt.Errorf("peers pending: %w", err)
		}
		if !pending {
			info, err := island.SynchronizeState(ctx, cfg.Revision, cfg.Tensors, cfg.Strategy)
			if err != nil {
				return fmt.Errorf("synchronize state: %w", err)
			}
			_ = info
			if cfg.Ready != nil {
				if err := cfg.Ready(ctx); err != nil {
					return fmt.Errorf("ready: %w", err)
				}
			}
			if p != nil {
				p.lastRevision = cfg.Revision
				p.haveRevision = true
			}
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := island.UpdateTopology(ctx); err != nil {
			return fmt.Errorf("update topology: %w", err)
		}
		if err := island.OptimizeTopology(ctx); err != nil {
			return fmt.Errorf("optimize topology: %w", err)
		}
		if cfg.PollInterval > 0 {
			if err := wait(ctx, cfg.PollInterval); err != nil {
				return err
			}
		}
	}
}

func wait(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

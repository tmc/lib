package breakout

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:generate cp $PUFFERLIB/ocean/breakout/breakout.cu breakout.cu
//go:generate go run ../../cmd/cudabind -type refModule -prefix ref -I shim -std c++20 -default-device -macros OBS_SIZE,BREAKOUT_GPU_MAX_BRICKS,BREAKOUT_THREADS_PER_ENV,BREAKOUT_BLOCK -o breakout.cu.gen.go vec.cu
//go:generate go run ../../cmd/cudabind -type optModule -prefix opt -D BREAKOUT_OPT -I shim -std c++20 -default-device -macros OBS_SIZE,BREAKOUT_GPU_MAX_BRICKS,BREAKOUT_BLOCK -o breakout_opt.cu.gen.go vec.cu

const (
	// ObsSize is the number of observations per environment.
	ObsSize = refObsSize

	// NumActions is the number of discrete actions: none, left and right.
	NumActions = 3
)

// breakout.cu and breakout_opt.cu agree on OBS_SIZE and BREAKOUT_BLOCK, the
// threads per block of every launch.
var (
	_ = [1]struct{}{}[refObsSize-optObsSize]
	_ = [1]struct{}{}[refBreakoutBlock-optBreakoutBlock]
)

// Config is the [env] section of PufferLib's config/breakout.ini.
type Config struct {
	Width, Height             int
	PaddleWidth, PaddleHeight int
	BallWidth, BallHeight     int
	BrickWidth, BrickHeight   int
	BrickRows, BrickCols      int
	InitialBallSpeed          float32
	MaxBallSpeed              float32
	PaddleSpeed               float32
	Frameskip                 int
	Continuous                bool
}

// DefaultConfig is the configuration in config/breakout.ini.
var DefaultConfig = Config{
	Width: 576, Height: 330,
	PaddleWidth: 62, PaddleHeight: 8,
	BallWidth: 32, BallHeight: 32,
	BrickWidth: 32, BrickHeight: 12,
	BrickRows: 6, BrickCols: 18,
	InitialBallSpeed: 256,
	MaxBallSpeed:     448,
	PaddleSpeed:      620,
	Frameskip:        3,
}

// kwargs returns c as the keys and values of breakout.ini, which the kernel
// breakout_config reads.
func (c Config) kwargs() []refDictItem {
	continuous := 0
	if c.Continuous {
		continuous = 1
	}
	return []refDictItem{
		item("frameskip", c.Frameskip),
		item("width", c.Width),
		item("height", c.Height),
		item("paddle_width", c.PaddleWidth),
		item("paddle_height", c.PaddleHeight),
		item("ball_width", c.BallWidth),
		item("ball_height", c.BallHeight),
		item("brick_width", c.BrickWidth),
		item("brick_height", c.BrickHeight),
		item("brick_rows", c.BrickRows),
		item("brick_cols", c.BrickCols),
		item("initial_ball_speed", c.InitialBallSpeed),
		item("max_ball_speed", c.MaxBallSpeed),
		item("paddle_speed", c.PaddleSpeed),
		item("continuous", continuous),
	}
}

func item[T int | float32](key string, value T) refDictItem {
	d := refDictItem{Value: float64(value)}
	copy(d.Key[:], key)
	return d
}

// Log is the episode statistics breakout.cu accumulates, summed over
// environments. Divide by N for means.
type Log struct {
	Perf          float32
	Score         float32
	EpisodeReturn float32
	EpisodeLength float32
	N             float32
}

// Log is both C types, which the kernels add up in device memory.
var _, _ = Log(refLog{}), Log(optLog{})

// Vec is a vector of Breakout environments on a CUDA device, run by the
// kernels of PufferLib's breakout.cu or of breakout_opt.cu.
type Vec struct {
	n       int
	ref     *refModule // breakout.cu, which keeps all of an env's state in its Env
	opt     *optModule // breakout_opt.cu
	arena   *cudamem.Arena
	refEnvs cudamem.Slice[refEnv]
	optEnvs cudamem.Slice[optEnv]
	state   cudamem.Slice[[4]uint32] // breakout_opt.cu's physics state
	sum     cudamem.Slice[refLog]
}

// New compiles breakout.cu for dev and creates n environments, seeded as
// PufferLib seeds them. Call [Vec.Reset] before the first [Vec.Step].
//
// The constant configuration belongs to the compiled module, so each Vec
// compiles and loads its own.
func New(dev *cuda.Device, cfg Config, n int) (*Vec, error) {
	return newVec(dev, cfg, n, false)
}

// NewOpt is like [New] but runs breakout_opt.cu, which steps the
// environments exactly as breakout.cu does, faster.
func NewOpt(dev *cuda.Device, cfg Config, n int) (*Vec, error) {
	return newVec(dev, cfg, n, true)
}

func newVec(dev *cuda.Device, cfg Config, n int, opt bool) (_ *Vec, err error) {
	if n <= 0 || n > (1<<31-1)/refBreakoutThreadsPerEnv {
		return nil, fmt.Errorf("breakout: invalid number of environments %d", n)
	}
	v := &Vec{n: n, arena: cudamem.NewArena(dev)}
	defer func() {
		if err != nil {
			v.Close()
			err = fmt.Errorf("breakout: %w", err)
		}
	}()
	v.sum = cudamem.AllocIn[refLog](v.arena, "log", 1)
	if opt {
		v.optEnvs = cudamem.AllocIn[optEnv](v.arena, "envs", n)
		v.state = cudamem.AllocIn[[4]uint32](v.arena, "state", n*optBreakoutStateGroups)
		v.opt, err = loadOptModule(dev)
	} else {
		v.refEnvs = cudamem.AllocIn[refEnv](v.arena, "envs", n)
		v.ref, err = loadRefModule(dev)
	}
	if err != nil {
		return nil, err
	}
	stream := dev.DefaultStream()
	if err := v.configure(stream, cfg); err != nil {
		return nil, err
	}
	if opt {
		err = v.opt.breakoutInit(stream, grid(n), block, v.optEnvs, v.state, int32(n))
	} else {
		err = v.ref.breakoutInit(stream, grid(n), block, v.refEnvs, v.state, int32(n))
	}
	if err != nil {
		return nil, err
	}
	if err := v.sum.MemsetAsync(0, stream); err != nil {
		return nil, err
	}
	return v, nil
}

// configure sets the module's d_bcfg from cfg as puf_vec_create does, by
// running gpu_breakout_host_fill_config on the device.
func (v *Vec) configure(stream *cuda.Stream, cfg Config) error {
	items := cfg.kwargs()
	kwargs := cudamem.AllocIn[refDictItem](v.arena, "kwargs", len(items))
	out := cudamem.AllocIn[refGpuBreakoutConfig](v.arena, "config", 1)
	if err := v.arena.Err(); err != nil {
		return err
	}
	if err := kwargs.CopyFromHost(items); err != nil {
		return err
	}
	one := [3]uint32{1, 1, 1}
	var err error
	if v.opt != nil {
		// Reinterpret cannot fail: the types are the same C types.
		kw, _ := cudamem.Reinterpret[optDictItem](kwargs)
		o, _ := cudamem.Reinterpret[optGpuBreakoutConfig](out)
		err = v.opt.breakoutConfig(stream, one, one, kw, int32(len(items)), o)
	} else {
		err = v.ref.breakoutConfig(stream, one, one, kwargs, int32(len(items)), out)
	}
	if err != nil {
		return err
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return err
	}
	c := make([]refGpuBreakoutConfig, 1)
	if err := out.CopyToHost(c); err != nil {
		return err
	}
	var code int32
	if v.opt != nil {
		code, err = v.opt.exitCode()
	} else {
		code, err = v.ref.exitCode()
	}
	switch {
	case err != nil:
		return err
	case code != 0:
		return fmt.Errorf("breakout.cu rejects %d×%d bricks: want 1 to %d", cfg.BrickRows, cfg.BrickCols, min(refBreakoutGpuMaxBricks, ObsSize-10))
	case v.opt != nil:
		return v.opt.setDBcfg(optGpuBreakoutConfig(c[0]))
	default:
		return v.ref.setDBcfg(c[0])
	}
}

// Close frees the environments and unloads the kernels. Work the Vec
// enqueued must be complete.
func (v *Vec) Close() error {
	var errs []error
	if v.arena != nil {
		errs = append(errs, v.arena.Free())
	}
	if v.ref != nil {
		errs = append(errs, v.ref.Close())
	}
	if v.opt != nil {
		errs = append(errs, v.opt.Close())
	}
	v.arena, v.ref, v.opt = nil, nil, nil
	return errors.Join(errs...)
}

// Len returns the number of environments.
func (v *Vec) Len() int {
	return v.n
}

// Reset enqueues on stream a reset of every environment. It writes the
// first observations, n×[ObsSize] bfloat16 values, to obs, and zeroes
// rewards and terminals.
func (v *Vec) Reset(stream *cuda.Stream, obs cudamem.Slice[uint16], rewards, terminals cudamem.Slice[float32]) error {
	n := int32(v.n)
	err := v.check(v.n, obs, rewards, terminals)
	if err == nil {
		if v.opt != nil {
			err = v.opt.gpuBreakoutResetKernel(stream, grid(v.n), block, v.optEnvs, v.state, obs, rewards, terminals, n)
		} else {
			err = v.ref.gpuBreakoutResetKernel(stream, grid(v.n*refBreakoutThreadsPerEnv), block, v.refEnvs, obs, rewards, terminals, n)
		}
	}
	if err != nil {
		return fmt.Errorf("breakout: reset: %w", err)
	}
	return nil
}

// Step enqueues on stream one step of every environment. actions holds one
// action per environment: 0, 1 (left) or 2 (right), or a value in [-1, 1]
// if the configuration is continuous. An environment whose episode ends
// resets itself and sets its terminal to 1.
func (v *Vec) Step(stream *cuda.Stream, actions cudamem.Slice[float32], obs cudamem.Slice[uint16], rewards, terminals cudamem.Slice[float32]) error {
	n := int32(v.n)
	err := v.check(actions.Len(), obs, rewards, terminals)
	if err == nil {
		if v.opt != nil {
			err = v.opt.gpuBreakoutStepKernel(stream, grid(v.n), block, v.optEnvs, v.state, actions, obs, rewards, terminals, n)
		} else {
			err = v.ref.gpuBreakoutStepKernel(stream, grid(v.n*refBreakoutThreadsPerEnv), block, v.refEnvs, actions, obs, rewards, terminals, 0, n)
		}
	}
	if err != nil {
		return fmt.Errorf("breakout: step: %w", err)
	}
	return nil
}

func (v *Vec) check(actions int, obs cudamem.Slice[uint16], rewards, terminals cudamem.Slice[float32]) error {
	if actions != v.n || obs.Len() != v.n*ObsSize || rewards.Len() != v.n || terminals.Len() != v.n {
		return fmt.Errorf("have actions %d, obs %d, rewards %d, terminals %d for %d environments",
			actions, obs.Len(), rewards.Len(), terminals.Len(), v.n)
	}
	return nil
}

// Log returns the statistics of the episodes that ended since the last
// call, waiting for the work enqueued on stream.
func (v *Vec) Log(ctx context.Context, stream *cuda.Stream) (Log, error) {
	l, err := v.log(ctx, stream)
	if err != nil {
		return Log{}, fmt.Errorf("breakout: log: %w", err)
	}
	return l, nil
}

func (v *Vec) log(ctx context.Context, stream *cuda.Stream) (Log, error) {
	var err error
	if v.opt != nil {
		// Reinterpret cannot fail: optLog and refLog are both Log.
		sum, _ := cudamem.Reinterpret[optLog](v.sum)
		err = v.opt.breakoutLog(stream, grid(v.n), block, v.optEnvs, int32(v.n), sum)
	} else {
		err = v.ref.breakoutLog(stream, grid(v.n), block, v.refEnvs, int32(v.n), v.sum)
	}
	if err != nil {
		return Log{}, err
	}
	if err := stream.Synchronize(ctx); err != nil {
		return Log{}, err
	}
	l := make([]refLog, 1)
	if err := v.sum.CopyToHost(l); err != nil {
		return Log{}, err
	}
	if err := v.sum.MemsetAsync(0, stream); err != nil {
		return Log{}, err
	}
	return Log(l[0]), nil
}

// block is the block of every launch.
var block = [3]uint32{refBreakoutBlock, 1, 1}

// grid returns the grid of blocks that covers threads.
func grid(threads int) [3]uint32 {
	return [3]uint32{uint32((threads + refBreakoutBlock - 1) / refBreakoutBlock), 1, 1}
}

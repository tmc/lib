package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"text/scanner"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

type sidecarPending interface{ Wait(context.Context) error }

// Validate every watched candidate before compilation or CUDA admission.
type sidecarLoader struct {
	cudahotswap.Loader
	ref   *sidecarReference
	entry string
}

func (l *sidecarLoader) Compile(ctx context.Context, source cudahotswap.Source) (*cudahotswap.Compilation, error) {
	if source.Entry != l.entry || len(source.Code) > 8<<20 || len(source.Flags) != 0 {
		return nil, errors.New("sidecar candidate entry, source bound or compiler flags differ")
	}
	copy := *l.ref
	copy.source = l.ref.source
	copy.source.CUDA = source.Code
	if err := sidecarABI(&copy, l.entry); err != nil {
		return nil, err
	}
	return l.Loader.Compile(ctx, source)
}

// The application owns the expression reference and actual device buffers.
// A failed wait or lease release retains its owner for cleanup.
type sidecarApplication struct {
	owners    *sessionOwners
	ref       *sidecarReference
	mu        sync.Mutex
	pending   sidecarPending
	readLease io.Closer
	submit    func(context.Context) (sidecarPending, error)
	write     func([]float32) error
	restore   func() error
	read      func([]float32) error
	observe   func(cudahotswap.State, []float32) error
}

func sidecarABI(ref *sidecarReference, entry string) error {
	if !regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`).MatchString(entry) {
		return errors.New("sidecar entry must be a CUDA identifier")
	}
	var lex scanner.Scanner
	code := ref.source.CUDA
	lines := strings.Split(code, "\n")
	for _, line := range lines {
		// C++ splices lines before removing comments. Inspect raw lines
		// so a continuation cannot hide a declaration from this scanner.
		if strings.HasSuffix(strings.TrimRight(line, " \t\r\v\f"), "\\") {
			return errors.New("sidecar source line continuations are unsupported")
		}
	}
	lex.Init(strings.NewReader(strings.Join(lines, "\n")))
	var lexicalError error
	lex.Error = func(_ *scanner.Scanner, msg string) { lexicalError = errors.New(msg) }
	var tokens []string
	for lex.Scan() != scanner.EOF {
		if len(tokens) >= 262144 {
			return errors.New("sidecar source token bound exceeded")
		}
		tokens = append(tokens, lex.TokenText())
	}
	if lexicalError != nil {
		return fmt.Errorf("sidecar source tokens: %w", lexicalError)
	}
	declarations := 0
	for i := 0; i < len(tokens); i++ {
		if tokens[i] == "#" {
			return errors.New("sidecar source preprocessor directives are unsupported")
		}
		if tokens[i] == "\\" {
			return errors.New("sidecar source line continuations are unsupported")
		}
		if i+5 >= len(tokens) || tokens[i] != "extern" || tokens[i+1] != `"C"` || tokens[i+2] != "__global__" || tokens[i+3] != "void" || tokens[i+4] != entry {
			continue
		}
		declarations++
		want := []string{"("}
		for _, p := range ref.source.Parameters {
			want = append(want, "const", "float", "*", p.Name, ",")
		}
		want = append(want, "float", "*", "eg_output", ")", "{")
		if i+5+len(want) > len(tokens) {
			return errors.New("sidecar entry pointer ABI incomplete")
		}
		for j, v := range want {
			if tokens[i+5+j] != v {
				return errors.New("sidecar entry must match ordered input pointers followed by eg_output")
			}
		}
	}
	if declarations != 1 {
		return errors.New("sidecar entry requires one emitted pointer ABI declaration")
	}
	return nil
}

func newSidecarSession(ctx context.Context, c config) (*cudahotswap.Service, func() error, error) {
	return newSidecarSessionObserved(ctx, c, nil)
}

func newSidecarSessionObserved(ctx context.Context, c config, observe func(cudahotswap.State, []float32) error) (*cudahotswap.Service, func() error, error) {
	return newSidecarSessionReady(ctx, c, observe, nil)
}

func newSidecarSessionReady(ctx context.Context, c config, observe func(cudahotswap.State, []float32) error, ready func(*sidecarApplication)) (*cudahotswap.Service, func() error, error) {
	return newSidecarSessionLoader(ctx, c, observe, ready, nil)
}

func newSidecarSessionLoader(ctx context.Context, c config, observe func(cudahotswap.State, []float32) error, ready func(*sidecarApplication), wrap func(cudahotswap.Loader) cudahotswap.Loader) (*cudahotswap.Service, func() error, error) {
	if ctx == nil || c.sidecar == "" || c.inputs == "" {
		return nil, nil, errors.New("sidecar requires context, metadata and explicit inputs")
	}
	ref, err := loadSidecarReference(c.path, c.sidecar, c.inputs)
	if err != nil {
		return nil, nil, err
	}
	if err := sidecarABI(ref, c.entry); err != nil {
		return nil, nil, err
	}
	a := &sidecarApplication{owners: &sessionOwners{acquire: acquireSessionLock}, ref: ref, observe: observe}
	o := a.owners
	cleanup := func() error {
		closeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return a.close(closeCtx)
	}
	o.lease, err = o.acquire(ctx, "cudahotswap: sidecar setup")
	if err != nil {
		return nil, cleanup, err
	}
	if o.lease == nil {
		return nil, cleanup, errors.New("sidecar setup returned no lease")
	}
	o.dev, err = cuda.NewDevice(c.device)
	if err != nil {
		return nil, cleanup, err
	}
	o.stream, err = o.dev.NewStream()
	if err != nil {
		return nil, cleanup, err
	}
	values := append(append([][]float32(nil), ref.inputs...), make([]float32, len(ref.want)))
	for _, data := range values {
		buffer, e := cudamem.Alloc[byte](o.dev, 4*len(data))
		if buffer.Ptr() != 0 {
			o.buffers = append(o.buffers, buffer)
		}
		if e != nil {
			return nil, cleanup, e
		}
		view, e := cudamem.Reinterpret[float32](buffer)
		if e != nil {
			return nil, cleanup, e
		}
		if e = view.CopyFromHost(data); e != nil {
			return nil, cleanup, e
		}
	}
	if err = o.lease.Close(); err != nil {
		return nil, cleanup, err
	}
	o.lease = nil
	loader, err := cudahotswap.NewNVRTCLoader(o.dev, append([]string{filepath.Dir(c.path)}, c.includes...)...)
	if err != nil {
		return nil, cleanup, err
	}
	var selectedLoader cudahotswap.Loader = &sidecarLoader{Loader: loader, ref: ref, entry: c.entry}
	if wrap != nil {
		selectedLoader = wrap(selectedLoader)
	}
	o.service, err = cudahotswap.NewService(selectedLoader)
	if o.service != nil {
		o.serviceClose = func() error { return o.service.Close(context.Background()) }
	}
	if err != nil {
		return nil, cleanup, err
	}
	if _, err = o.service.Register(ctx, "kernel", cudahotswap.Source{Code: ref.source.CUDA, Filename: c.path, Entry: c.entry}, "sidecar expression source"); err != nil {
		return nil, cleanup, err
	}
	args := make([]any, len(o.buffers))
	for i, b := range o.buffers {
		args[i] = b
	}
	launch := cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(1, 1, 1), Stream: o.stream}
	a.submit = func(ctx context.Context) (sidecarPending, error) {
		p, e := o.service.Launch(ctx, "kernel", launch, args...)
		if p == nil {
			return nil, e
		}
		return p, e
	}
	output := o.buffers[len(o.buffers)-1]
	a.restore = func() error {
		for i, data := range ref.inputs {
			v, e := cudamem.Reinterpret[float32](o.buffers[i])
			if e != nil {
				return e
			}
			if e = v.CopyFromHost(data); e != nil {
				return e
			}
		}
		return nil
	}
	a.write = func(data []float32) error {
		v, e := cudamem.Reinterpret[float32](output)
		if e != nil {
			return e
		}
		return v.CopyFromHost(data)
	}
	a.read = func(data []float32) error {
		v, e := cudamem.Reinterpret[float32](output)
		if e != nil {
			return e
		}
		return v.CopyToHost(data)
	}
	if _, err = o.service.SetCheck("kernel", cudahotswap.CheckConfig{Checker: a.check}, "frozen egraph expression reference"); err != nil {
		return nil, cleanup, err
	}
	if err = a.check(ctx, cudahotswap.State{}); err != nil {
		return nil, cleanup, err
	}
	if ready != nil {
		ready(a)
	}
	return o.service, cleanup, nil
}

func (a *sidecarApplication) releaseRead() error {
	if a.readLease == nil {
		return nil
	}
	if err := a.readLease.Close(); err != nil {
		return err
	}
	a.readLease = nil
	return nil
}
func (a *sidecarApplication) finish(ctx context.Context) error {
	if err := a.releaseRead(); err != nil {
		return err
	}
	if a.pending != nil {
		if err := a.pending.Wait(ctx); err != nil {
			return err
		}
		a.pending = nil
	}
	return nil
}
func (a *sidecarApplication) check(ctx context.Context, state cudahotswap.State) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.finish(ctx); err != nil {
		return err
	}
	lease, err := a.owners.acquire(ctx, "cudahotswap: sidecar prefill")
	if lease != nil {
		a.readLease = lease
	}
	if err != nil {
		return err
	}
	if lease == nil {
		return errors.New("sidecar prefill returned no lease")
	}
	if a.restore != nil {
		if err := a.restore(); err != nil {
			return errors.Join(err, a.releaseRead())
		}
	}
	prefill := make([]float32, len(a.ref.want))
	for i := range prefill {
		prefill[i] = math.Float32frombits(0x7fc00001)
	}
	if err := errors.Join(a.write(prefill), a.releaseRead()); err != nil {
		return err
	}
	p, err := a.submit(ctx)
	if p != nil {
		a.pending = p
	}
	if err != nil {
		return err
	}
	if p == nil {
		return errors.New("sidecar launch returned no completion owner")
	}
	if err = p.Wait(ctx); err != nil {
		return err
	}
	a.pending = nil
	lease, err = a.owners.acquire(ctx, "cudahotswap: sidecar readback")
	if lease != nil {
		a.readLease = lease
	}
	if err != nil {
		return err
	}
	if lease == nil {
		return errors.New("sidecar readback returned no lease")
	}
	out := make([]float32, len(a.ref.want))
	err = errors.Join(a.read(out), a.releaseRead())
	if err != nil {
		return err
	}
	if a.observe != nil {
		if err := a.observe(state, append([]float32(nil), out...)); err != nil {
			return err
		}
	}
	return compareSidecar(a.ref.want, out)
}
func compareSidecar(want, out []float32) error {
	if len(want) == 0 || len(want) != len(out) {
		return errors.New("sidecar reference: extent differs")
	}
	for i, v := range out {
		w := want[i]
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-float64(w)) > 1e-5+1e-5*math.Abs(float64(w)) {
			return fmt.Errorf("sidecar reference: element %d got %g want %g", i, v, w)
		}
	}
	return nil
}
func (a *sidecarApplication) close(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.finish(ctx); err != nil {
		return err
	}
	return a.owners.close()
}

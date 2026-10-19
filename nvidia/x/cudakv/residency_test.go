package cudakv

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"slices"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

var residencyFault = errors.New("residency fake fault")

type fakeResidencyCompletion struct {
	apply                 func()
	failWait, failCleanup bool
	done, closed          bool
}

func (c *fakeResidencyCompletion) Wait(ctx context.Context) error {
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if c.failWait {
		c.failWait = false
		return residencyFault
	}
	if !c.done {
		c.apply()
		c.done = true
	}
	if c.failCleanup {
		c.failCleanup = false
		return residencyFault
	}
	c.closed = true
	return nil
}
func (c *fakeResidencyCompletion) Close() error { return c.Wait(nil) }

type fakeResidency struct {
	pc                            *PagedCache
	buffers                       map[*residencyHost][]float32
	freed                         map[*residencyHost]int
	borrows                       int
	allocs, copies                int
	failAlloc, failCopy, failFree int
	waitFault, cleanupFault       bool
	completions                   []*fakeResidencyCompletion
}

func newFakeResidency(t *testing.T, total int) (*fakeResidency, *Residency) {
	t.Helper()
	pc, err := NewTestPagedCache(2, 1, 2, 2, total)
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.AllocSequence(7, 3); err != nil {
		t.Fatal(err)
	}
	f := &fakeResidency{pc: pc, buffers: make(map[*residencyHost][]float32), freed: make(map[*residencyHost]int)}
	for layer := range pc.hostK {
		for i := range pc.hostK[layer] {
			pc.hostK[layer][i] = float32(1000*layer + 3*i + 1)
			pc.hostV[layer][i] = float32(1000*layer + 7*i + 2)
		}
	}
	gen, _ := pc.AllocationGeneration(7)
	r, err := newResidency(pc, 7, gen, f.ops())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	return f, r
}
func (f *fakeResidency) ops() residencyOps {
	return residencyOps{
		page: func(pc *PagedCache, layer, slot int) (Page, error) {
			if pc != f.pc || layer < 0 || layer >= pc.numLayers || slot < 0 || slot >= pc.totalBlocks {
				return Page{}, residencyFault
			}
			return Page{}, nil
		},
		borrow: func(v residencyView, _ *cuda.Device) (func(), error) {
			f.borrows++
			var once sync.Once
			return func() { once.Do(func() { f.borrows-- }) }, nil
		},
		allocate: func(_ *cuda.Device, n int) (*residencyHost, error) {
			f.allocs++
			h := &residencyHost{}
			f.buffers[h] = make([]float32, n)
			h.free = func() error {
				for _, c := range f.completions {
					if !c.closed {
						return errors.New("fake free before actual completion")
					}
				}
				if f.failFree > 0 {
					f.failFree--
					return residencyFault
				}
				f.freed[h]++
				if f.freed[h] > 1 {
					return errors.New("fake double free")
				}
				delete(f.buffers, h)
				return nil
			}
			if f.failAlloc == f.allocs {
				return h, residencyFault
			}
			return h, nil
		},
		copy: func(v residencyView, h *residencyHost, _ *cuda.Stream, toHost bool) (residencyCompletion, error) {
			f.copies++
			c := &fakeResidencyCompletion{failWait: f.waitFault, failCleanup: f.cleanupFault}
			f.waitFault = false
			f.cleanupFault = false
			c.apply = func() {
				data := f.pc.hostK[v.layer]
				if v.v {
					data = f.pc.hostV[v.layer]
				}
				data = data[v.slot*v.elements : (v.slot+1)*v.elements]
				if toHost {
					copy(f.buffers[h], data)
				} else {
					copy(data, f.buffers[h])
				}
			}
			f.completions = append(f.completions, c)
			if f.failCopy == f.copies {
				return c, residencyFault
			}
			return c, nil
		},
		stream: func(stream *cuda.Stream, _ *cuda.Device) error {
			if stream == nil {
				return residencyFault
			}
			return nil
		},
	}
}
func residencyReadAll(t *testing.T, pc *PagedCache) [][]float32 {
	t.Helper()
	var out [][]float32
	for layer := 0; layer < 2; layer++ {
		for token := 0; token < 3; token++ {
			k, v, err := pc.ReadToken(7, token, layer)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, k, v)
		}
	}
	return out
}

func TestResidencyRoundTrip(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	pc := f.pc
	want := residencyReadAll(t, pc)
	old, err := r.Binding()
	if err != nil {
		t.Fatal(err)
	}
	if err := pc.ValidateSequence(old); err != nil {
		t.Fatal(err)
	}
	if err := r.Evict(&cuda.Stream{}); err != nil {
		t.Fatal(err)
	}
	if r.Tier() != ResidencyEvicting || pc.NumFreeBlocks() != 2 {
		t.Fatal("prediction became completed eviction", r.Tier(), pc.NumFreeBlocks())
	}
	if _, err := r.Binding(); err == nil {
		t.Fatal("in-flight binding admitted")
	}
	if err := pc.ValidateSequence(old); err == nil {
		t.Fatal("old binding admitted during eviction")
	}
	if _, _, err := pc.ReadToken(7, 0, 0); err == nil {
		t.Fatal("suspended token read admitted")
	}
	if err := pc.WriteToken(7, 0, 0, []float32{1, 2}, []float32{3, 4}); err == nil {
		t.Fatal("suspended token write admitted")
	}
	if err := pc.Retain(old.Blocks); err == nil {
		t.Fatal("in-flight prefix retain admitted")
	}
	if err := pc.AllocSequenceWithPrefix(9, old.Blocks, 4); err == nil {
		t.Fatal("in-flight prefix share admitted")
	}
	if table := pc.BlockTable(7); table != nil {
		t.Fatal("in-flight table exposed", table)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := r.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if r.Tier() != ResidencyEvicting || pc.NumFreeBlocks() != 2 || f.borrows != 8 {
		t.Fatal("cancellation released native ownership")
	}
	allocator := pc.blocks
	if err := pc.Close(); !errors.Is(err, cudamem.ErrInUse) || pc.closed || pc.blocks != allocator {
		t.Fatal("Close reset an in-use allocator", err)
	}
	if err := r.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r.Tier() != ResidencyHost || pc.NumFreeBlocks() != 4 {
		t.Fatal("completed eviction did not release pool slots", r.Tier(), pc.NumFreeBlocks())
	}
	if err := pc.AllocSequence(99, 8); err != nil {
		t.Fatal("evicted capacity not reusable", err)
	}
	if err := r.Prefetch(&cuda.Stream{}); err == nil || r.Tier() != ResidencyHost {
		t.Fatal("exhaustion fabricated prefetch", err)
	}
	pc.FreeSequence(99)
	if err := pc.AllocSequence(99, 2); err != nil {
		t.Fatal(err)
	}
	// The pool can overwrite a freed old slot while the host copy stays valid.
	if err := pc.WriteToken(99, 0, 0, []float32{-1, -2}, []float32{-3, -4}); err != nil {
		t.Fatal(err)
	}
	if err := r.Prefetch(&cuda.Stream{}); err != nil {
		t.Fatal(err)
	}
	if r.Tier() != ResidencyPrefetching {
		t.Fatal(r.Tier())
	}
	if _, err := r.Binding(); err == nil {
		t.Fatal("predicted restore became resident")
	}
	if err := r.Wait(nil); err != nil {
		t.Fatal(err)
	}
	binding, err := r.Binding()
	if err != nil {
		t.Fatal(err)
	}
	if binding.Generation == old.Generation || pc.ValidateSequence(old) == nil || pc.ValidateSequence(binding) != nil {
		t.Fatal("restore did not publish new allocation identity")
	}
	got := residencyReadAll(t, pc)
	for i := range want {
		if !slices.Equal(want[i], got[i]) {
			t.Fatalf("layer/token %d: got %v want %v", i, got[i], want[i])
		}
	}
	binding.Blocks[0] = -1
	if _, err := r.Binding(); err != nil {
		t.Fatal("binding aliases allocator", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if f.borrows != 0 || len(f.buffers) != 0 {
		t.Fatal("residency leaked owners", f.borrows, len(f.buffers))
	}
	if err := pc.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestResidencySharedAndStaleAdmission(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	pc := f.pc
	old, _ := r.Binding()
	if err := pc.AllocSequenceWithPrefix(9, old.Blocks, 4); err != nil {
		t.Fatal(err)
	}
	if err := r.Evict(&cuda.Stream{}); err == nil || f.copies != 0 || f.allocs != 0 {
		t.Fatal("shared eviction touched live prefix", err)
	}
	pc.FreeSequence(9)
	if _, err := newResidency(pc, 7, old.Generation, f.ops()); err == nil {
		t.Fatal("duplicate residency owner admitted")
	}
	pc.FreeSequence(7)
	if err := pc.AllocSequence(7, 3); err != nil {
		t.Fatal(err)
	}
	if err := r.Evict(&cuda.Stream{}); err == nil || f.copies != 0 {
		t.Fatal("stale allocation enqueued work", err)
	}
}

func TestResidencyLateCompletion(t *testing.T) {
	for _, phase := range []string{"evict", "prefetch"} {
		t.Run(phase, func(t *testing.T) {
			f, r := newFakeResidency(t, 4)
			pc := f.pc
			if err := r.Evict(&cuda.Stream{}); err != nil {
				t.Fatal(err)
			}
			if phase == "prefetch" {
				if err := r.Wait(nil); err != nil {
					t.Fatal(err)
				}
				if err := r.Prefetch(&cuda.Stream{}); err != nil {
					t.Fatal(err)
				}
			}
			pc.FreeSequence(7)
			if err := pc.AllocSequence(7, 3); err != nil {
				t.Fatal(err)
			}
			gen, _ := pc.AllocationGeneration(7)
			blocks := pc.BlockTable(7)
			if err := r.Wait(nil); err == nil || r.Tier() != ResidencyLost {
				t.Fatal("late completion attached to replacement", err)
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			if now, _ := pc.AllocationGeneration(7); now != gen || !slices.Equal(blocks, pc.BlockTable(7)) {
				t.Fatal("late cleanup destroyed replacement")
			}
		})
	}
}

func TestResidencyWaitCleanupAndEnqueueFailure(t *testing.T) {
	for _, phase := range []string{"evict", "prefetch"} {
		for _, fault := range []string{"wait", "cleanup", "enqueue"} {
			t.Run(phase+"/"+fault, func(t *testing.T) {
				f, r := newFakeResidency(t, 4)
				if phase == "prefetch" {
					if err := r.Evict(&cuda.Stream{}); err != nil {
						t.Fatal(err)
					}
					if err := r.Wait(nil); err != nil {
						t.Fatal(err)
					}
				}
				switch fault {
				case "wait":
					f.waitFault = true
				case "cleanup":
					f.cleanupFault = true
				case "enqueue":
					f.failCopy = f.copies + 2
				}
				var err error
				if phase == "evict" {
					err = r.Evict(&cuda.Stream{})
				} else {
					err = r.Prefetch(&cuda.Stream{})
				}
				if (fault == "enqueue") != errors.Is(err, residencyFault) {
					t.Fatal("unexpected enqueue result", err)
				}
				if fault != "enqueue" {
					if err := r.Wait(nil); !errors.Is(err, residencyFault) {
						t.Fatal(err)
					}
					if _, err := r.Binding(); err == nil {
						t.Fatal("failed wait published binding")
					}
					if f.borrows != 8 || len(f.buffers) != 8 {
						t.Fatal("failed wait dropped owners")
					}
				}
				err = r.Wait(nil)
				if fault == "enqueue" {
					if !errors.Is(err, residencyFault) {
						t.Fatal(err)
					}
					want := ResidencyHBM
					if phase == "prefetch" {
						want = ResidencyHost
					}
					if r.Tier() != want {
						t.Fatal("failed enqueue did not retain prior tier", r.Tier())
					}
				} else if err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func TestResidencyAllocationAndCloseRetry(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	f.failAlloc = 2
	if err := r.Evict(&cuda.Stream{}); !errors.Is(err, residencyFault) || f.copies != 0 || r.Tier() != ResidencyHBM {
		t.Fatal("partial pinned failure changed residency", err)
	}
	if _, err := r.Binding(); err != nil {
		t.Fatal("HBM fallback unavailable", err)
	}
	f.failFree = 1
	if err := r.Close(); !errors.Is(err, residencyFault) {
		t.Fatal(err)
	}
	if f.borrows != 8 || f.pc.residencies != 1 || len(f.buffers) != 1 {
		t.Fatal("failed free lost retry owner", f.borrows, len(f.buffers))
	}
	if err := f.pc.Close(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if f.borrows != 0 || len(f.buffers) != 0 || f.pc.residencies != 0 {
		t.Fatal("retry did not finish")
	}
}

func TestResidencyGenerationExhaustion(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	if err := r.Evict(&cuda.Stream{}); err != nil {
		t.Fatal(err)
	}
	if err := r.Wait(nil); err != nil {
		t.Fatal(err)
	}
	before := f.copies
	f.pc.blocks.nextAllocation = ^uint64(0)
	if err := r.Prefetch(&cuda.Stream{}); err == nil || f.copies != before || f.pc.NumFreeBlocks() != 4 || r.Tier() != ResidencyHost {
		t.Fatal("generation exhaustion wrapped or enqueued", err)
	}
}

type residencyCountingCopier struct{ calls int }

func (c *residencyCountingCopier) CopyBlock(int, int) error { c.calls++; return nil }
func TestResidencyCompactionAndGraphOwners(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	pc := f.pc
	copier := &residencyCountingCopier{}
	compactor := NewCompactor(DefaultDefragPolicy()).WithCopier(copier)
	if _, err := compactor.Compact(pc); !errors.Is(err, cudamem.ErrInUse) || copier.calls != 0 {
		t.Fatal("compaction effected copies before refusal", err)
	}
	plan := CompactionPlan{Moves: []BlockMove{{SrcBlock: 3, DstBlock: 0}}, Remappings: map[int]int{3: 0}}
	before := pc.BlockTable(7)
	pc.blocks.ApplyCompaction(plan)
	if !slices.Equal(before, pc.BlockTable(7)) {
		t.Fatal("stale raw plan changed live residency")
	}
	pc.mu.Lock()
	err := pc.copyBlockAsync(3, 0, nil)
	pc.mu.Unlock()
	if !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("stale direct copy admitted", err)
	}
	if _, err := compactor.CaptureGraph(pc, &cuda.Stream{}, plan); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("graph capture reached vendor before refusal", err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	// A previously captured owner blocks residency and pool cleanup until all
	// graph resources actually close. Fake destructors touch no CUDA API.
	pc.compactionOwners = 1
	cg := &CompactionGraph{cache: pc, exec: &cudagraph.Exec{}, graph: &cudagraph.Graph{}}
	gen, _ := pc.AllocationGeneration(7)
	if _, err := newResidency(pc, 7, gen, f.ops()); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("captured graph owner bypassed", err)
	}
	if err := pc.Close(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("pool closed under captured graph", err)
	}
	if err := cg.destroyWith(func(*cudagraph.Exec) error { return residencyFault }, func(*cudagraph.Graph) error { return nil }); !errors.Is(err, residencyFault) || pc.compactionOwners != 1 {
		t.Fatal("failed graph destroy released pool", err)
	}
	if err := cg.destroyWith(func(*cudagraph.Exec) error { return nil }, func(*cudagraph.Graph) error { return nil }); err != nil || pc.compactionOwners != 0 {
		t.Fatal("graph retry lost owner", err)
	}
}

func TestResidencyCloseWaitRetry(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	f.waitFault = true
	if err := r.Evict(&cuda.Stream{}); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); !errors.Is(err, residencyFault) || f.borrows != 8 || f.pc.residencies != 1 || len(f.buffers) != 8 {
		t.Fatal("failed Close lost retry ownership", err)
	}
	if err := r.Evict(&cuda.Stream{}); err == nil {
		t.Fatal("closing owner enqueued new work")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.pc.AllocationGeneration(7); ok {
		t.Fatal("closed host sequence stayed bindable")
	}
	if f.pc.NumFreeBlocks() != 4 || f.borrows != 0 || len(f.buffers) != 0 {
		t.Fatal("Close did not finish cleanup")
	}
}

func TestResidencyMissingCompletionRefused(t *testing.T) {
	f, r := newFakeResidency(t, 4)
	r.ops.copy = func(residencyView, *residencyHost, *cuda.Stream, bool) (residencyCompletion, error) { return nil, nil }
	if err := r.Evict(&cuda.Stream{}); err == nil {
		t.Fatal("missing completion accepted")
	}
	if err := r.Wait(nil); err == nil || r.Tier() != ResidencyHBM || f.pc.NumFreeBlocks() != 2 {
		t.Fatal("missing completion manufactured host residency", err)
	}
}

func TestResidencyConcurrentStatus(t *testing.T) {
	_, r := newFakeResidency(t, 4)
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for range 3 {
		wg.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
					r.Tier()
					r.Binding()
				}
			}
		})
	}
	for range 10 {
		if err := r.Evict(&cuda.Stream{}); err != nil {
			t.Fatal(err)
		}
		if err := r.Wait(nil); err != nil {
			t.Fatal(err)
		}
		if err := r.Prefetch(&cuda.Stream{}); err != nil {
			t.Fatal(err)
		}
		if err := r.Wait(nil); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)
	wg.Wait()
}

// TestResidencyNativeAdmission checks source ordering without executing a native
// fixture, querying a device, acquiring a GPU lock or calling vendor APIs.
func TestResidencyNativeAdmission(t *testing.T) {
	parse := func() *ast.File {
		t.Helper()
		file, err := parser.ParseFile(token.NewFileSet(), "residency_native_test.go", nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		return file
	}
	if !residencyNativeGuarded(parse()) {
		t.Fatal("native fixture does not gate before Require")
	}
	for _, index := range []int{0, 1, 2, 3} {
		file := parse()
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if ok && fn.Name.Name == "TestResidencyNativeRoundTrip" {
				fn.Body.List = append(fn.Body.List[:index], fn.Body.List[index+1:]...)
			}
		}
		if residencyNativeGuarded(file) {
			t.Fatalf("missing admission statement %d accepted", index)
		}
	}
}

func residencyNativeCall(expr ast.Expr, pkg, method string) *ast.CallExpr {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return nil
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != method {
		return nil
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || id.Name != pkg {
		return nil
	}
	return call
}
func residencyNativeFatal(body *ast.BlockStmt) bool {
	if body == nil || len(body.List) != 1 {
		return false
	}
	expr, ok := body.List[0].(*ast.ExprStmt)
	return ok && (residencyNativeCall(expr.X, "t", "Fatal") != nil || residencyNativeCall(expr.X, "t", "Fatalf") != nil)
}
func residencyNativeGuarded(file *ast.File) bool {
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "TestResidencyNativeRoundTrip" {
			continue
		}
		body := fn.Body.List
		if len(body) < 5 {
			return false
		}
		assign, ok := body[0].(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 || len(assign.Lhs) != 2 {
			return false
		}
		owner, ok := assign.Lhs[0].(*ast.Ident)
		if !ok || owner.Name != "lockOwner" {
			return false
		}
		failure, ok := assign.Lhs[1].(*ast.Ident)
		if !ok || failure.Name != "err" {
			return false
		}
		call := residencyNativeCall(assign.Rhs[0], "cudalock", "Inherited")
		if call == nil || len(call.Args) != 1 {
			return false
		}
		env, ok := call.Args[0].(*ast.SelectorExpr)
		if !ok || env.Sel.Name != "Getenv" {
			return false
		}
		id, ok := env.X.(*ast.Ident)
		if !ok || id.Name != "os" {
			return false
		}
		guard, ok := body[1].(*ast.IfStmt)
		if !ok || !residencyNativeFatal(guard.Body) {
			return false
		}
		cond, ok := guard.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return false
		}
		left, lok := cond.X.(*ast.Ident)
		right, rok := cond.Y.(*ast.Ident)
		if !lok || !rok || left.Name != "err" || right.Name != "nil" {
			return false
		}
		flags, ok := body[2].(*ast.IfStmt)
		if !ok || !residencyNativeFatal(flags.Body) {
			return false
		}
		or, ok := flags.Cond.(*ast.BinaryExpr)
		if !ok || or.Op != token.LOR {
			return false
		}
		for i, expr := range []ast.Expr{or.X, or.Y} {
			neq, ok := expr.(*ast.BinaryExpr)
			if !ok || neq.Op != token.NEQ {
				return false
			}
			get := residencyNativeCall(neq.X, "os", "Getenv")
			if get == nil || len(get.Args) != 1 {
				return false
			}
			name, ok := get.Args[0].(*ast.BasicLit)
			if !ok {
				return false
			}
			want := []string{`"GPU_LOCK_DISABLED"`, `"CUDATEST_NO_LOCK"`}[i]
			empty, ok := neq.Y.(*ast.BasicLit)
			if !ok || empty.Value != `""` || name.Value != want {
				return false
			}
		}
		cleanup, ok := body[3].(*ast.ExprStmt)
		if !ok {
			return false
		}
		keep := residencyNativeCall(cleanup.X, "t", "Cleanup")
		if keep == nil || len(keep.Args) != 1 {
			return false
		}
		closure, ok := keep.Args[0].(*ast.FuncLit)
		if !ok {
			return false
		}
		var inherited, retained bool
		ast.Inspect(closure, func(n ast.Node) bool {
			if expr, ok := n.(ast.Expr); ok && residencyNativeCall(expr, "cudalock", "Inherited") != nil {
				inherited = true
			}
			if id, ok := n.(*ast.Ident); ok && id.Name == "lockOwner" {
				retained = true
			}
			return true
		})
		if !inherited || !retained {
			return false
		}
		require, ok := body[4].(*ast.AssignStmt)
		return ok && len(require.Rhs) == 1 && residencyNativeCall(require.Rhs[0], "cudatest", "Require") != nil
	}
	return false
}

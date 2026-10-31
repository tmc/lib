package egraph_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This fixture has two ranks and at most two calls per rank. Canonical JSON is
// test-local immutable Data, not a new wire contract or identity scheme. It
// lets the existing accounted graph APIs retain strings without admitting the
// unsupported arbitrary-payload insertion path. No runtime owner enters Data.
const collectiveGraphDataLimit = 4096

func collectiveGraphCallData(c egraph.CollectiveCall) (string, error) {
	for _, s := range []string{c.Communicator, c.Operation, c.DType, c.Reduction, c.Layout} {
		if len(s) > 256 {
			return "", errors.New("collective graph: oversized field")
		}
	}
	if err := c.Validate(); err != nil {
		return "", err
	}
	b, err := json.Marshal(c)
	if err != nil {
		return "", err
	}
	got, err := collectiveGraphReadCall(string(b))
	if err != nil || got != c {
		return "", errors.New("collective graph: value cannot be preserved")
	}
	return string(b), nil
}

func collectiveGraphReadCall(data string) (egraph.CollectiveCall, error) {
	var c egraph.CollectiveCall
	if len(data) == 0 || len(data) > collectiveGraphDataLimit {
		return c, errors.New("collective graph: payload bound")
	}
	d := json.NewDecoder(strings.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return egraph.CollectiveCall{}, err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return egraph.CollectiveCall{}, errors.New("collective graph: trailing data")
	}
	if err := c.Validate(); err != nil {
		return egraph.CollectiveCall{}, err
	}
	canonical, err := json.Marshal(c)
	if err != nil || string(canonical) != data {
		return egraph.CollectiveCall{}, errors.New("collective graph: noncanonical value")
	}
	return c, nil
}

func collectiveGraphBuild(calls [][]egraph.CollectiveCall) (*egraph.EGraph, egraph.Id, error) {
	if len(calls) == 0 || len(calls) > 2 {
		return nil, 0, errors.New("collective graph: rank bound")
	}
	g := egraph.New()
	legal := egraph.NewLegalAdapter(g)
	shape, err := cudashape.New(4)
	if err != nil {
		return nil, 0, err
	}
	meta := egraph.TypedMetadata{Shape: shape, DType: cudair.Float32, Effect: cudair.EffectMutating, NumericalPolicy: "exact"}
	add := func(node egraph.ENode) (egraph.Id, error) {
		id, err := g.AddChecked(node)
		if err == nil {
			err = legal.SetMetadata(id, meta)
		}
		return id, err
	}
	var ranks []egraph.Id
	for rank, sequence := range calls {
		if len(sequence) == 0 || len(sequence) > 2 {
			return nil, 0, errors.New("collective graph: sequence bound")
		}
		var children []egraph.Id
		for _, c := range sequence {
			data, err := collectiveGraphCallData(c)
			if err != nil {
				return nil, 0, err
			}
			id, err := add(egraph.ENode{Op: "collective-call", Data: data})
			if err != nil {
				return nil, 0, err
			}
			children = append(children, id)
		}
		id, err := add(egraph.ENode{Op: "collective-rank", Data: strconv.Itoa(rank), Children: children})
		if err != nil {
			return nil, 0, err
		}
		ranks = append(ranks, id)
	}
	root, err := add(egraph.ENode{Op: "collective-world", Data: strconv.Itoa(len(ranks)), Children: ranks})
	return g, root, err
}

func collectiveGraphRead(root *egraph.Expr) ([][]egraph.CollectiveCall, error) {
	if root == nil || root.Op != "collective-world" || len(root.Children) == 0 || len(root.Children) > 2 || root.Data != strconv.Itoa(len(root.Children)) {
		return nil, errors.New("collective graph: invalid world")
	}
	var calls [][]egraph.CollectiveCall
	for rank, sequence := range root.Children {
		if sequence == nil || sequence.Op != "collective-rank" || sequence.Data != strconv.Itoa(rank) || len(sequence.Children) == 0 || len(sequence.Children) > 2 {
			return nil, errors.New("collective graph: invalid rank sequence")
		}
		var row []egraph.CollectiveCall
		for _, term := range sequence.Children {
			if term == nil || term.Op != "collective-call" || len(term.Children) != 0 {
				return nil, errors.New("collective graph: invalid call term")
			}
			data, ok := term.Data.(string)
			if !ok {
				return nil, errors.New("collective graph: call data is not a string")
			}
			c, err := collectiveGraphReadCall(data)
			if err != nil {
				return nil, err
			}
			if c.Rank != rank || c.WorldSize != len(root.Children) {
				return nil, errors.New("collective graph: term participation mismatch")
			}
			row = append(row, c)
		}
		calls = append(calls, row)
	}
	return calls, nil
}

var errCollectiveGraphMemory = errors.New("collective graph: reservation limit")

// These are the public Memory API's reservation units, not an estimate of RAM.
type collectiveGraphMemory struct {
	limit, live, peak int64
	badRelease        bool
}

func (m *collectiveGraphMemory) Reserve(n int64) error {
	if n < 0 || n > m.limit-m.live {
		return errCollectiveGraphMemory
	}
	m.live += n
	if m.live > m.peak {
		m.peak = m.live
	}
	return nil
}

func (m *collectiveGraphMemory) Release(n int64) {
	if n < 0 || n > m.live {
		m.badRelease = true
		return
	}
	m.live -= n
}

func collectiveGraphExtract(g *egraph.EGraph, root egraph.Id, m *collectiveGraphMemory) (*egraph.Extraction, func(), error) {
	scope, err := egraphcore.NewMemoryScope(m)
	if err != nil {
		return nil, func() {}, err
	}
	budget, err := egraphcore.NewWorkBudgetWithMemory(4096, scope)
	if err != nil {
		scope.Close()
		return nil, func() {}, err
	}
	copy, err := g.CloneWithBudget(context.Background(), budget)
	if err != nil {
		scope.Close()
		return nil, func() {}, err
	}
	x, err := egraph.ExtractBoundedWithMemory(context.Background(), copy, root, nil, egraph.ExtractionLimits{MaxWork: 4096, MaxNodes: 16, MaxDepth: 4}, m)
	if err != nil {
		x.Close()
		scope.Close()
		return nil, func() {}, err
	}
	return x, func() { x.Close(); scope.Close() }, nil
}

type collectiveGraphExecution struct {
	rank       int
	op         cudacollective.OperationDescriptor
	send, recv driver.DevicePtr
}

type collectiveGraphTransport struct {
	*cudacollective.DeterministicTestTransport
	executions []collectiveGraphExecution
}

func (t *collectiveGraphTransport) Execute(ctx context.Context, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
	if err := t.DeterministicTestTransport.Execute(ctx, comm, op, send, recv); err != nil {
		return err
	}
	t.executions = append(t.executions, collectiveGraphExecution{comm.Rank(), op, send, recv})
	return nil
}

func collectiveGraphReplay(root *egraph.Expr, before [][]egraph.CollectiveCall, comms []*cudacollective.Communicator, inputs [][][]float32) (*egraph.CollectiveReceipt, error) {
	decoded, err := collectiveGraphRead(root)
	if err != nil {
		return nil, err
	}
	if len(decoded) != len(before) {
		return nil, errors.New("collective graph: changed rank count")
	}
	// The extracted terms, not the admitted snapshot, are the lowering input.
	for rank := range decoded {
		if err := egraph.CheckCollectiveRewrite(before[rank], decoded[rank]); err != nil {
			return nil, err
		}
	}
	p, err := (egraph.CollectiveSchedule{Communicators: comms, Calls: decoded, Quarantine: new(egraph.CollectiveQuarantine)}).Plan()
	if err != nil {
		return nil, err
	}
	receipt, runErr := p.Replay(context.Background(), inputs)
	return receipt, errors.Join(runErr, p.Close())
}

func TestCollectiveGraphReplay(t *testing.T) {
	for _, distinct := range []bool{false, true} {
		t.Run(strconv.FormatBool(distinct), func(t *testing.T) {
			calls := collectiveCalls(2, 7)
			if distinct {
				for rank := range calls {
					for i := range calls[rank] {
						c := &calls[rank][i]
						c.Communicator, c.Generation, c.Stream = "other-epoch", 11, 19
						c.Send.Address += 0x100000
						c.Recv.Address += 0x100000
						c.EffectToken += 50
					}
				}
			}
			before := cloneCalls(calls)
			g, root, err := collectiveGraphBuild(calls)
			if err != nil {
				t.Fatal(err)
			}
			// Mutation of the construction input cannot change graph Data.
			calls[0][0].EffectToken = 99999
			memory := &collectiveGraphMemory{limit: 1 << 20}
			x, close, err := collectiveGraphExtract(g, root, memory)
			if err != nil {
				t.Fatal(err)
			}
			defer close()
			tr := &collectiveGraphTransport{DeterministicTestTransport: cudacollective.NewDeterministicTestTransport(2)}
			comms := collectiveComms(t, cudacollective.Generation(before[0][0].Generation), tr)
			inputs := collectiveInputs(2)
			receipt, err := collectiveGraphReplay(x.Expr, before, comms, cloneInputs(inputs))
			if err != nil {
				t.Fatal(err)
			}
			if len(tr.executions) != 4 || len(receipt.Steps) != 2 {
				t.Fatalf("execution count: %d, receipt %+v", len(tr.executions), receipt)
			}
			for step := range 2 {
				want := sumOracle(inputs[step])
				if receipt.Steps[step].EffectToken != before[0][step].EffectToken {
					t.Fatal("extracted effect token was lost")
				}
				for rank := range 2 {
					c := before[rank][step]
					exec := tr.executions[2*step+rank]
					desc, err := c.OperationDescriptor()
					if err != nil || exec.rank != rank || exec.op != desc || exec.send != driver.DevicePtr(c.Send.Address+c.Send.Offset) || exec.recv != driver.DevicePtr(c.Recv.Address+c.Recv.Offset) {
						t.Fatalf("extracted term not executed: %+v, call %+v, %v", exec, c, err)
					}
					if !slices.Equal(receipt.Steps[step].Outputs[rank], want) || receipt.Steps[step].Completions[rank] != c.Completion {
						t.Fatalf("rank %d step %d: %+v, want %v", rank, step, receipt.Steps[step], want)
					}
					if slices.Equal(want, inputs[step][rank]) {
						t.Fatal("oracle cannot detect a missing peer")
					}
				}
			}
			close()
			if memory.live != 0 || memory.badRelease || memory.peak == 0 {
				t.Fatalf("reservations not released: %+v", memory)
			}
		})
	}
}

func TestCollectiveGraphHashConsing(t *testing.T) {
	base := collectiveCalls(1, 7)[0][0]
	data, err := collectiveGraphCallData(base)
	if err != nil {
		t.Fatal(err)
	}
	g := egraph.New()
	id, _ := g.AddChecked(egraph.ENode{Op: "collective-call", Data: data})
	same, err := g.AddChecked(egraph.ENode{Op: "collective-call", Data: data})
	if err != nil || same != id {
		t.Fatal("same exact term did not hash-cons", same, id, err)
	}
	for _, tt := range []struct {
		name   string
		change func(*egraph.CollectiveCall)
	}{
		{"communicator", func(c *egraph.CollectiveCall) { c.Communicator = "other" }},
		{"generation", func(c *egraph.CollectiveCall) { c.Generation++ }},
		{"rank", func(c *egraph.CollectiveCall) { c.Rank = 1 }},
		{"world", func(c *egraph.CollectiveCall) { c.WorldSize = 3 }},
		{"operation", func(c *egraph.CollectiveCall) { c.Operation = "allgather" }},
		{"dtype", func(c *egraph.CollectiveCall) { c.DType = "i32" }},
		{"count", func(c *egraph.CollectiveCall) { c.Count, c.Send.Bytes, c.Recv.Bytes = 2, 8, 8 }},
		{"in-place", func(c *egraph.CollectiveCall) { c.InPlace = true; c.Recv = c.Send }},
		{"stream", func(c *egraph.CollectiveCall) { c.Stream++ }},
		{"effect", func(c *egraph.CollectiveCall) { c.EffectToken++ }},
		{"root", func(c *egraph.CollectiveCall) { c.Root = 1 }},
		{"reduction", func(c *egraph.CollectiveCall) { c.Reduction = "max" }},
		{"layout", func(c *egraph.CollectiveCall) { c.Layout = "strided" }},
		{"partition", func(c *egraph.CollectiveCall) { c.Partition = 1 }},
		{"send", func(c *egraph.CollectiveCall) { c.Send.Address += 0x10000 }},
		{"recv", func(c *egraph.CollectiveCall) { c.Recv.Offset += 64 }},
		{"predecessor", func(c *egraph.CollectiveCall) { c.Predecessor = 99 }},
		{"completion", func(c *egraph.CollectiveCall) { c.Completion++ }},
		{"capture", func(c *egraph.CollectiveCall) { c.StreamState = egraph.CollectiveStreamCapturing }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := base
			tt.change(&c)
			changed, err := collectiveGraphCallData(c)
			if err != nil {
				if c.Validate() == nil {
					t.Fatal("valid bounded identity was not representable", err)
				}
				return
			}
			other, err := g.AddChecked(egraph.ENode{Op: "collective-call", Data: changed})
			if err != nil || other == id || changed == data {
				t.Fatal("changed collective identity hash-consed", other, id, err)
			}
		})
	}
}

func TestCollectiveGraphDecoding(t *testing.T) {
	c := collectiveCalls(1, 7)[0][0]
	data, err := collectiveGraphCallData(c)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := collectiveGraphReadCall(data); err != nil || got != c {
		t.Fatal("roundtrip lost actual call", got, err)
	}
	for _, tt := range []struct{ name, data string }{
		{"unknown", strings.TrimSuffix(data, "}") + ",\"Extra\":0}"},
		{"duplicate", strings.TrimSuffix(data, "}") + ",\"Generation\":7}"},
		{"missing", strings.Replace(data, "\"Rank\":0,", "", 1)},
		{"space", data + " "},
		{"trailing", data + "{}"},
		{"null", "null"},
		{"invalid", "{"},
		{"large", strings.Repeat("x", collectiveGraphDataLimit+1)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := collectiveGraphReadCall(tt.data); err == nil || got != (egraph.CollectiveCall{}) {
				t.Fatalf("noncanonical term accepted: %+v, %v", got, err)
			}
		})
	}
}

func collectiveGraphChange(t *testing.T, term *egraph.Expr, change func(*egraph.CollectiveCall)) {
	t.Helper()
	c, err := collectiveGraphReadCall(term.Data.(string))
	if err != nil {
		t.Fatal(err)
	}
	change(&c)
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	term.Data = string(b)
}

func TestCollectiveGraphRefusals(t *testing.T) {
	for _, name := range []string{"generation", "rank", "token", "completion", "alias", "count", "dtype", "drop", "reorder", "duplicate", "rank-drop", "wrong-data"} {
		t.Run(name, func(t *testing.T) {
			before := collectiveCalls(2, 7)
			g, root, err := collectiveGraphBuild(before)
			if err != nil {
				t.Fatal(err)
			}
			memory := &collectiveGraphMemory{limit: 1 << 20}
			x, close, err := collectiveGraphExtract(g, root, memory)
			if err != nil {
				t.Fatal(err)
			}
			defer close()
			switch name {
			case "rank-drop":
				x.Expr.Children = x.Expr.Children[:1]
			case "wrong-data":
				x.Expr.Children[0].Children[0].Data = before[0][0]
			default:
				for rank, sequence := range x.Expr.Children {
					switch name {
					case "drop":
						sequence.Children = sequence.Children[:1]
					case "reorder":
						sequence.Children[0], sequence.Children[1] = sequence.Children[1], sequence.Children[0]
					case "duplicate":
						sequence.Children = append(sequence.Children, sequence.Children[0])
					default:
						collectiveGraphChange(t, sequence.Children[0], func(c *egraph.CollectiveCall) {
							switch name {
							case "generation":
								c.Generation++
							case "rank":
								c.Rank = 1 - rank
							case "token":
								c.EffectToken += 500
							case "completion":
								c.Completion += 17
							case "alias":
								c.Recv.Offset += 64
							case "count":
								c.Count, c.Send.Bytes, c.Recv.Bytes = 2, 8, 8
							case "dtype":
								c.DType = "i32"
							}
						})
						if name == "completion" {
							collectiveGraphChange(t, sequence.Children[1], func(c *egraph.CollectiveCall) { c.Predecessor += 17 })
						}
					}
				}
			}
			tr := &collectiveGraphTransport{DeterministicTestTransport: cudacollective.NewDeterministicTestTransport(2)}
			comms := collectiveComms(t, 7, tr)
			receipt, err := collectiveGraphReplay(x.Expr, before, comms, collectiveInputs(2))
			if err == nil || receipt != nil || len(tr.executions) != 0 {
				t.Fatalf("changed extracted terms reached execution: %s, receipt %+v, executions %d, err %v", name, receipt, len(tr.executions), err)
			}
		})
	}
}

func TestCollectiveGraphLegalEffects(t *testing.T) {
	c := collectiveCalls(1, 7)[0][0]
	data, _ := collectiveGraphCallData(c)
	g := egraph.New()
	a, _ := g.AddChecked(egraph.ENode{Op: "collective-call", Data: data})
	c.EffectToken++
	changed, _ := collectiveGraphCallData(c)
	b, _ := g.AddChecked(egraph.ENode{Op: "collective-call", Data: changed})
	pure, _ := g.AddChecked(egraph.ENode{Op: "pure"})
	legal := egraph.NewLegalAdapter(g)
	shape, _ := cudashape.New(4)
	meta := egraph.TypedMetadata{Shape: shape, DType: cudair.Float32, Effect: cudair.EffectMutating, NumericalPolicy: "exact"}
	for _, id := range []egraph.Id{a, b} {
		if err := legal.SetMetadata(id, meta); err != nil {
			t.Fatal(err)
		}
	}
	meta.Effect = cudair.EffectReadOnly
	if err := legal.SetMetadata(pure, meta); err != nil {
		t.Fatal(err)
	}
	for _, other := range []egraph.Id{b, pure} {
		if _, merged, err := legal.LegalUnion(a, other); !errors.Is(err, egraph.ErrEffectBoundaryViolation) || merged || g.Find(a) == g.Find(other) {
			t.Fatal("collective effect dropped by union", merged, err)
		}
	}
}

func TestCollectiveGraphMemory(t *testing.T) {
	g, root, err := collectiveGraphBuild(collectiveCalls(2, 7))
	if err != nil {
		t.Fatal(err)
	}
	m := &collectiveGraphMemory{limit: 1 << 20}
	x, close, err := collectiveGraphExtract(g, root, m)
	if err != nil || x.Expr == nil || m.live == 0 {
		t.Fatal("accounted path unavailable", err)
	}
	close()
	if m.live != 0 || m.badRelease {
		t.Fatal("accounted graph leaked reservations", m)
	}
	small := &collectiveGraphMemory{limit: m.peak / 2}
	x, close, err = collectiveGraphExtract(g, root, small)
	close()
	if !errors.Is(err, errCollectiveGraphMemory) || x != nil || small.live != 0 || small.badRelease {
		t.Fatal("required memory bound bypassed", x, small, err)
	}

	// Reference-valued matching remains unsupported by the accounted matcher.
	// The public unlimited matcher can discover this action, so refusal is a
	// memory-capability boundary, not an invalid pattern or empty fixture.
	input := egraph.New()
	payload := []egraph.CollectiveCall{collectiveCalls(1, 7)[0][0]}
	_, err = input.AddChecked(egraph.ENode{Op: "payload-seed", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	rule := &egraph.Rewrite{Name: "unsupported-payload-match", LHS: &egraph.Pattern{Op: "payload-seed", HasData: true, Data: payload}, RHS: egraph.Pat("payload-result")}
	actions, err := egraph.EnumerateActionsChecked(input, []*egraph.Rewrite{rule})
	if err != nil || len(actions) != 1 {
		t.Fatal("capability negative not reachable", actions, err)
	}
	memory := &collectiveGraphMemory{limit: 1 << 20}
	owner, err := egraph.EnumerateActionsWithMemory(input, []*egraph.Rewrite{rule}, 4096, memory)
	if !errors.Is(err, egraph.ErrMemoryOperation) || owner != nil || memory.live != 0 || memory.badRelease {
		t.Fatal("reference payload matching unexpectedly admitted", owner, memory, err)
	}
	classes, nodes := input.Size()
	if classes != 1 || nodes != 1 {
		t.Fatal("refused matching mutated input", classes, nodes)
	}
	t.Logf("accounted graph/extraction peak %d reservation bytes; refusal cap %d", m.peak, small.limit)
}

// This assertion names the actual consumed interface without another runtime.
var _ cudacollective.Transport = (*collectiveGraphTransport)(nil)

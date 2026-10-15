package egraphcore

import (
	"errors"
	"reflect"
	"testing"
)

var errMemoryFixture = errors.New("fixture memory limit")

func TestMemoryCanonicalCopyWorkLimit(t *testing.T) {
	source := NewFactStore[uint32]()
	_, err := source.Register(Schema{Name: "facts", Columns: []Column{{Kind: ClassValue}}})
	if err != nil {
		t.Fatal(err)
	}
	memory := new(fixtureMemory)
	scope, err := NewMemoryScope(memory)
	if err != nil {
		t.Fatal(err)
	}
	budget, err := NewWorkBudgetWithMemory(1, scope)
	if err != nil {
		t.Fatal(err)
	}
	copy, err := source.CanonicalCopyWithBudget(func(id uint32) uint32 { return id }, budget)
	if copy != nil || !errors.Is(err, ErrMatchWorkLimit) || budget.Used() > 1 {
		t.Fatal(copy, err, budget.Used())
	}
	scope.Close()
	if memory.used != 0 {
		t.Fatal(memory.used)
	}
}

func TestMemoryCanonicalFactsEveryReservation(t *testing.T) {
	source := NewFactStore[uint32]()
	relation, err := source.Register(Schema{Name: "facts", Columns: []Column{{Kind: ClassValue}, {Kind: StringValue}}})
	if err != nil {
		t.Fatal(err)
	}
	identity := func(id uint32) uint32 { return id }
	for _, id := range []uint32{1, 2} {
		if _, err := relation.Add([]Value[uint32]{Class(id), String[uint32]("fact")}, identity); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := relation.MarkSubsumed([]Value[uint32]{Class(uint32(2)), String[uint32]("fact")}, identity); err != nil {
		t.Fatal(err)
	}
	cursor := relation.Snapshot()
	canonical := func(uint32) uint32 { return 1 }
	run := func(memory *fixtureMemory) error {
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		copy, err := source.CanonicalCopyWithBudget(canonical, budget)
		if err != nil {
			return err
		}
		if copy == source || copy.Get("facts") == relation || copy.Get("facts").Len() != 1 {
			t.Fatal("not a detached canonical copy")
		}
		rows, complete, err := copy.Get("facts").Query(nil, canonical, budget)
		if err != nil {
			return err
		}
		if !complete || len(rows) != 0 {
			t.Fatal("canonical collision lost subsumption", rows, complete)
		}
		schema, err := copy.Get("facts").SchemaWithBudget(budget)
		if err != nil {
			return err
		}
		schema.Columns[0].Sort = "detached"
		if copy.Get("facts").schema.Columns[0].Sort != "" {
			t.Fatal("schema aliases relation")
		}
		return nil
	}
	baseline := new(fixtureMemory)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &fixtureMemory{failAt: fail}
		if err := run(memory); !errors.Is(err, errMemoryFixture) {
			t.Fatalf("reservation %d: %v", fail, err)
		}
		if memory.used != 0 {
			t.Fatal("reservation leaked", fail, memory.used)
		}
		if relation.Len() != 2 || relation.Snapshot() != cursor {
			t.Fatal("source mutated")
		}
	}
}

type fixtureMemory struct {
	used, peak int64
	calls      int
	failAt     int
}

func (m *fixtureMemory) Reserve(bytes int64) error {
	m.calls++
	if m.calls == m.failAt {
		return errMemoryFixture
	}
	m.used += bytes
	if m.used > m.peak {
		m.peak = m.used
	}
	return nil
}

func (m *fixtureMemory) Release(bytes int64) { m.used -= bytes }

func fixtureWork(t *testing.T, memory *fixtureMemory) (*WorkBudget, *MemoryScope, error) {
	t.Helper()
	scope, err := NewMemoryScope(memory)
	if err != nil {
		return nil, nil, err
	}
	budget, err := NewWorkBudgetWithMemory(100000, scope)
	if err != nil {
		scope.Close()
		return nil, nil, err
	}
	return budget, scope, nil
}

func TestMemoryScopeOwnership(t *testing.T) {
	memory := new(fixtureMemory)
	scope, err := NewMemoryScope(memory)
	if err != nil {
		t.Fatal(err)
	}
	initial := scope.Bytes()
	if initial != int64(reflect.TypeFor[MemoryScope]().Size()) {
		t.Fatal(initial)
	}
	if err := scope.Reserve(32); err != nil {
		t.Fatal(err)
	}
	if scope.Bytes() != initial+32 || memory.used != scope.Bytes() {
		t.Fatal(scope.Bytes(), memory.used)
	}
	scope.Close()
	scope.Close()
	if memory.used != 0 || scope.Bytes() != 0 {
		t.Fatal(memory.used, scope.Bytes())
	}
	if err := scope.Reserve(1); !errors.Is(err, ErrMemoryClosed) {
		t.Fatal(err)
	}
}

func TestMemoryMatcherBeforeCandidateCallback(t *testing.T) {
	memory := &fixtureMemory{failAt: 3}
	budget, scope, err := fixtureWork(t, memory)
	if err != nil {
		t.Fatal(err)
	}
	defer scope.Close()
	calls := 0
	callbacks := MatcherCallbacks[uint32, string, string]{
		Find:            func(id uint32) uint32 { return id },
		ValidPattern:    func(string) bool { return true },
		IsVariable:      func(string) (string, bool) { return "", false },
		PatternChildren: func(string) []string { return nil },
		Candidates:      func(uint32, string) []string { calls++; return nil },
		NodeChildren:    func(string) []uint32 { return nil },
		NodeEligible:    func(string, string) bool { return true },
	}
	result, _, err := SearchWithMatcher("leaf", []uint32{0}, budget, 10, callbacks)
	if !errors.Is(err, errMemoryFixture) || result != nil || calls != 0 {
		t.Fatalf("matches=%v calls=%d err=%v", result, calls, err)
	}
}

func TestMemoryMatcherEveryReservation(t *testing.T) {
	run := func(memory *fixtureMemory) error {
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		pattern := &Pattern{Op: "?x", Variable: true}
		matches, limited, err := Search(pattern, []uint32{2, 5}, budget, 10, func(id uint32) uint32 { return id }, func(uint32) []string { return nil }, func(node string) string { return node }, func(string) []uint32 { return nil })
		if err != nil {
			return err
		}
		if limited || len(matches) != 2 || matches[0].Subst["?x"] != 2 || matches[1].Subst["?x"] != 5 {
			t.Fatal(matches, limited)
		}
		if scope.Bytes() <= int64(reflect.TypeFor[MemoryScope]().Size()) || memory.used != scope.Bytes() {
			t.Fatal("returned matches have no owner")
		}
		return nil
	}
	baseline := new(fixtureMemory)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	if baseline.used != 0 {
		t.Fatal("baseline leaked", baseline.used)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &fixtureMemory{failAt: fail}
		if err := run(memory); !errors.Is(err, errMemoryFixture) {
			t.Fatalf("reservation %d: %v", fail, err)
		}
		if memory.used != 0 {
			t.Fatalf("reservation %d leaked %d", fail, memory.used)
		}
	}
}

func TestMemoryJoinEveryReservation(t *testing.T) {
	relation, err := NewRelation[uint32](Schema{Name: "items", Columns: []Column{{Kind: Int64Value}, {Kind: StringValue}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range [][]Value[uint32]{{Int64[uint32](3), String[uint32]("alpha")}, {Int64[uint32](7), String[uint32]("beta")}} {
		if _, err := relation.Add(row, func(id uint32) uint32 { return id }); err != nil {
			t.Fatal(err)
		}
	}
	atoms := []Atom[uint32]{{Relation: relation, Terms: []Term[uint32]{Variable[uint32]("n"), Variable[uint32]("name")}}}
	run := func(memory *fixtureMemory) error {
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		bindings, complete, err := Join(atoms, func(id uint32) uint32 { return id }, budget)
		if err != nil {
			return err
		}
		if !complete || len(bindings) != 2 || bindings[0]["n"].Int64 != 3 || bindings[1]["name"].String != "beta" {
			t.Fatal(bindings, complete)
		}
		if scope.Bytes() == 0 {
			t.Fatal("bindings have no reservation owner")
		}
		return nil
	}
	baseline := new(fixtureMemory)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	if baseline.used != 0 {
		t.Fatal(baseline.used)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &fixtureMemory{failAt: fail}
		if err := run(memory); !errors.Is(err, errMemoryFixture) {
			t.Fatalf("reservation %d: %v", fail, err)
		}
		if memory.used != 0 {
			t.Fatalf("reservation %d leaked %d", fail, memory.used)
		}
	}
	if relation.Len() != 2 {
		t.Fatal("join mutated borrowed relation")
	}
}

func TestMemoryTupleKeyIdentity(t *testing.T) {
	row := []Value[uint32]{Class[uint32](2), Int64[uint32](-3), Bool[uint32](true), Float64[uint32](-0.25), String[uint32]("x\x00y"), Unit[uint32](), BigInt[uint32]("999999999999999999999"), BigRat[uint32]("2/3")}
	memory := new(fixtureMemory)
	budget, scope, err := fixtureWork(t, memory)
	if err != nil {
		t.Fatal(err)
	}
	defer scope.Close()
	got, ok := tupleKeyWithBudget(row, budget)
	if !ok || got != tupleKey(row) {
		t.Fatal("tuple identity changed")
	}
}

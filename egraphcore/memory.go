package egraphcore

import (
	"errors"
	"math"
	"reflect"
)

// Memory reserves storage before allocation. Reserve must leave its live byte
// count unchanged on error. Release relinquishes a previous reservation; it does not
// free or inspect the associated Go values. Implementations may share one limit
// with executable workspace. Calls need not be safe for concurrent use.
type Memory interface {
	Reserve(bytes int64) error
	Release(bytes int64)
}

// ErrMemorySize reports an invalid or overflowing storage request.
var ErrMemorySize = errors.New("egraphcore: invalid memory size")

// ErrMemoryClosed reports use of a released allocation scope.
var ErrMemoryClosed = errors.New("egraphcore: memory scope is closed")

// MemoryScope owns reservations for one operation and its returned values.
// Close must follow the last use of those values. Temporary and replaced
// backing arrays remain charged until Close, giving a conservative upper
// bound on live storage. Units cover Go objects, backing-array slots and map
// entries; allocator and map-bucket overhead are outside these units.
// Callbacks must return borrowed data or reserve their own storage.
// A nil scope is unlimited. A scope is not safe for concurrent use.
type MemoryScope struct {
	memory Memory
	bytes  int64
	err    error
	closed bool
}

// NewMemoryScope reserves and creates a result owner. A nil memory capability
// returns a nil, unlimited scope. A failed reservation creates no owner.
func NewMemoryScope(memory Memory) (*MemoryScope, error) {
	if memory == nil {
		return nil, nil
	}
	bytes := int64(reflect.TypeFor[MemoryScope]().Size())
	if err := memory.Reserve(bytes); err != nil {
		return nil, err
	}
	return &MemoryScope{memory: memory, bytes: bytes}, nil
}

// Reserve charges bytes before creating storage. The first error is sticky.
func (s *MemoryScope) Reserve(bytes int64) error {
	if bytes < 0 {
		if s != nil && s.err == nil {
			s.err = ErrMemorySize
		}
		return ErrMemorySize
	}
	if s == nil {
		return nil
	}
	if s.closed {
		if s.err == nil {
			s.err = ErrMemoryClosed
		}
		return ErrMemoryClosed
	}
	if s.err != nil {
		return s.err
	}
	if s.memory == nil {
		return nil
	}
	if bytes > math.MaxInt64-s.bytes {
		s.err = ErrMemorySize
		return s.err
	}
	if err := s.memory.Reserve(bytes); err != nil {
		s.err = err
		return err
	}
	s.bytes += bytes
	return nil
}

// Err returns the first failed storage reservation.
func (s *MemoryScope) Err() error {
	if s == nil {
		return nil
	}
	return s.err
}

// Bytes reports reservations held by the scope, including its own storage.
func (s *MemoryScope) Bytes() int64 {
	if s == nil {
		return 0
	}
	return s.bytes
}

// Close releases all reservations once. Returned values must no longer be used.
func (s *MemoryScope) Close() {
	if s == nil || s.closed {
		return
	}
	s.closed = true
	if s.memory != nil {
		s.memory.Release(s.bytes)
	}
	s.bytes = 0
}

// NewWorkBudgetWithMemory attaches the caller-owned result scope to a work
// budget. Keep scope open until all returned matches, bindings and expressions
// are no longer used. The work and memory limits are independent.
func NewWorkBudgetWithMemory(limit int, scope *MemoryScope) (*WorkBudget, error) {
	if err := scope.Reserve(int64(reflect.TypeFor[WorkBudget]().Size())); err != nil {
		return nil, err
	}
	return &WorkBudget{limit: limit, memory: scope}, nil
}

// ReserveMemory checks storage before allocation. A nil work budget carries
// no memory capability. A false result is available through MemoryError.
func (b *WorkBudget) ReserveMemory(bytes int64) bool {
	if b == nil {
		return bytes >= 0
	}
	return b.memory.Reserve(bytes) == nil
}

// MemoryError returns the first failed memory reservation, if any.
func (b *WorkBudget) MemoryError() error {
	if b == nil {
		return nil
	}
	return b.memory.Err()
}

// MemoryScope returns the borrowed result owner attached to b. A nil result
// means allocation is unlimited. The caller that created the scope closes it.
func (b *WorkBudget) MemoryScope() *MemoryScope {
	if b == nil {
		return nil
	}
	return b.memory
}

func memorySlots[T any](budget *WorkBudget, count int) bool {
	size := uint64(reflect.TypeFor[T]().Size())
	if count < 0 || size != 0 && uint64(count) > math.MaxInt64/size {
		if budget != nil && budget.memory != nil && budget.memory.err == nil {
			budget.memory.err = ErrMemorySize
		}
		return false
	}
	return budget.ReserveMemory(int64(uint64(count) * size))
}

func memorySlice[T any](budget *WorkBudget, length, capacity int) ([]T, bool) {
	if length < 0 || capacity < length || !memorySlots[T](budget, capacity) {
		return nil, false
	}
	return make([]T, length, capacity), true
}

func memoryAppend[T any](budget *WorkBudget, values []T, added ...T) ([]T, bool) {
	if len(added) > math.MaxInt-len(values) {
		if budget != nil && budget.memory != nil && budget.memory.err == nil {
			budget.memory.err = ErrMemorySize
		}
		return nil, false
	}
	needed := len(values) + len(added)
	if needed > cap(values) {
		capacity := needed
		if cap(values) <= math.MaxInt/2 && 2*cap(values) > capacity {
			capacity = 2 * cap(values)
		}
		grown, ok := memorySlice[T](budget, len(values), capacity)
		if !ok {
			return nil, false
		}
		copy(grown, values)
		values = grown
	}
	return append(values, added...), true
}

func memoryObject[T any](budget *WorkBudget, value T) (*T, bool) {
	if !memorySlots[T](budget, 1) {
		return nil, false
	}
	object := new(T)
	*object = value
	return object, true
}

func memoryMap[K comparable, V any](budget *WorkBudget, hint int) (map[K]V, bool) {
	// Account the map value and hinted slots before make. New entries are
	// charged again by memoryPut, conservatively covering later growth.
	if !memorySlots[map[K]V](budget, 1) || !memorySlots[struct {
		Key   K
		Value V
	}](budget, hint) {
		return nil, false
	}
	return make(map[K]V, hint), true
}

func memoryPut[K comparable, V any](budget *WorkBudget, values map[K]V, key K, value V) bool {
	if _, exists := values[key]; !exists && !memorySlots[struct {
		Key   K
		Value V
	}](budget, 1) {
		return false
	}
	values[key] = value
	return true
}

package egraph

import (
	"fmt"
	"math"
	"reflect"

	"github.com/tmc/lib/egraphcore"
)

func formatterPayload(t reflect.Type) bool {
	return t.Implements(reflect.TypeFor[fmt.Formatter]()) ||
		t.Implements(reflect.TypeFor[fmt.Stringer]()) ||
		t.Implements(reflect.TypeFor[fmt.GoStringer]()) ||
		t.Implements(reflect.TypeFor[error]())
}

func primitivePayload(v reflect.Value) bool {
	if !v.IsValid() || formatterPayload(v.Type()) {
		return false
	}
	switch v.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.String:
		return true
	}
	return false
}

func reflectSlots(work *egraphcore.WorkBudget, t reflect.Type, count int) error {
	size := uint64(t.Size())
	if count < 0 || size != 0 && uint64(count) > math.MaxInt64/size {
		return egraphcore.ErrMemorySize
	}
	if !work.ReserveMemory(int64(uint64(count) * size)) {
		return work.MemoryError()
	}
	return nil
}

func mutablePayloadType(t reflect.Type) bool {
	return mutablePayloadTypeDepth(t, 0)
}

func mutablePayloadTypeDepth(t reflect.Type, depth int) bool {
	if depth > 256 {
		return true
	}
	switch t.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface, reflect.Func, reflect.Chan, reflect.UnsafePointer:
		return true
	case reflect.Array:
		return mutablePayloadTypeDepth(t.Elem(), depth+1)
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if mutablePayloadTypeDepth(t.Field(i).Type, depth+1) {
				return true
			}
		}
	}
	return false
}

// cloneDataWithMemory follows the existing clone's reference memoization, but
// reserves each new object/backing before reflection creates it. Its memo and
// map-iterator copies are scratch; the returned clone is result-owned. Opaque
// functions/channels, cycles, custom formatters, mutable map keys and unexported mutable
// fields are explicitly unsupported. Immutable strings remain shared.
func cloneDataWithMemory(value any, b *extractionBudget, work, output *egraphcore.WorkBudget) (any, error) {
	if value == nil {
		return nil, nil
	}
	type clonedValue struct {
		value  reflect.Value
		active bool
	}
	memo, err := graphMap[cloneVisit, clonedValue](work, 0)
	if err != nil {
		return nil, err
	}
	var clone func(reflect.Value, int) (reflect.Value, error)
	clone = func(v reflect.Value, depth int) (reflect.Value, error) {
		if err := b.take(0); err != nil {
			return reflect.Value{}, err
		}
		if depth > 256 {
			return reflect.Value{}, ErrMemoryOperation
		}
		if !v.IsValid() {
			return v, nil
		}
		if formatterPayload(v.Type()) {
			return reflect.Value{}, ErrMemoryOperation
		}
		switch v.Kind() {
		case reflect.Func, reflect.Chan, reflect.UnsafePointer:
			return reflect.Value{}, ErrMemoryOperation
		case reflect.Interface:
			if v.IsNil() {
				return reflect.Zero(v.Type()), nil
			}
			child, err := clone(v.Elem(), depth+1)
			if err != nil {
				return reflect.Value{}, err
			}
			if err := reflectSlots(output, v.Type(), 1); err != nil {
				return reflect.Value{}, err
			}
			out := reflect.New(v.Type()).Elem()
			out.Set(child)
			return out, nil
		case reflect.Pointer, reflect.Map, reflect.Slice:
			if v.IsNil() {
				return reflect.Zero(v.Type()), nil
			}
			key := cloneVisitFor(v)
			if prior, ok := memo[key]; ok {
				if prior.active {
					return reflect.Value{}, ErrMemoryOperation
				}
				return prior.value, nil
			}
			var out reflect.Value
			switch v.Kind() {
			case reflect.Pointer:
				if err := reflectSlots(output, v.Type().Elem(), 1); err != nil {
					return reflect.Value{}, err
				}
				out = reflect.New(v.Type().Elem())
			case reflect.Slice:
				if err := reflectSlots(output, v.Type(), 1); err != nil {
					return reflect.Value{}, err
				}
				if err := reflectSlots(output, v.Type().Elem(), v.Len()); err != nil {
					return reflect.Value{}, err
				}
				out = reflect.MakeSlice(v.Type(), v.Len(), v.Len())
			case reflect.Map:
				if mutablePayloadType(v.Type().Key()) {
					return reflect.Value{}, ErrMemoryOperation
				}
				if err := reflectSlots(output, v.Type(), 1); err != nil {
					return reflect.Value{}, err
				}
				if err := reflectSlots(output, v.Type().Key(), v.Len()); err != nil {
					return reflect.Value{}, err
				}
				if err := reflectSlots(output, v.Type().Elem(), v.Len()); err != nil {
					return reflect.Value{}, err
				}
				out = reflect.MakeMapWithSize(v.Type(), v.Len())
			}
			if err := graphPut(work, memo, key, clonedValue{value: out, active: true}); err != nil {
				return reflect.Value{}, err
			}
			switch v.Kind() {
			case reflect.Pointer:
				child, err := clone(v.Elem(), depth+1)
				if err != nil {
					return reflect.Value{}, err
				}
				out.Elem().Set(child)
			case reflect.Slice:
				for i := 0; i < v.Len(); i++ {
					child, err := clone(v.Index(i), depth+1)
					if err != nil {
						return reflect.Value{}, err
					}
					out.Index(i).Set(child)
				}
			case reflect.Map:
				if err := graphSlots[reflect.MapIter](work, 1); err != nil {
					return reflect.Value{}, err
				}
				var iter reflect.MapIter
				iter.Reset(v)
				for iter.Next() {
					if err := reflectSlots(work, v.Type().Key(), 1); err != nil {
						return reflect.Value{}, err
					}
					if err := reflectSlots(work, v.Type().Elem(), 1); err != nil {
						return reflect.Value{}, err
					}
					key, err := clone(iter.Key(), depth+1)
					if err != nil {
						return reflect.Value{}, err
					}
					value, err := clone(iter.Value(), depth+1)
					if err != nil {
						return reflect.Value{}, err
					}
					out.SetMapIndex(key, value)
				}
			}
			memo[key] = clonedValue{value: out}
			return out, nil
		case reflect.Array, reflect.Struct:
			if err := reflectSlots(output, v.Type(), 1); err != nil {
				return reflect.Value{}, err
			}
			out := reflect.New(v.Type()).Elem()
			out.Set(v)
			if v.Kind() == reflect.Array {
				for i := 0; i < v.Len(); i++ {
					child, err := clone(v.Index(i), depth+1)
					if err != nil {
						return reflect.Value{}, err
					}
					out.Index(i).Set(child)
				}
			} else {
				for i := 0; i < v.NumField(); i++ {
					if !v.Field(i).CanInterface() || !out.Field(i).CanSet() {
						if mutablePayloadType(v.Type().Field(i).Type) {
							return reflect.Value{}, ErrMemoryOperation
						}
						continue
					}
					child, err := clone(v.Field(i), depth+1)
					if err != nil {
						return reflect.Value{}, err
					}
					out.Field(i).Set(child)
				}
			}
			return out, nil
		default:
			return v, nil
		}
	}
	copy, err := clone(reflect.ValueOf(value), 1)
	if err != nil {
		return nil, err
	}
	if !copy.IsValid() {
		return nil, nil
	}
	// Interface may box an addressable scalar/header value independently.
	if err := reflectSlots(output, copy.Type(), 1); err != nil {
		return nil, err
	}
	return copy.Interface(), nil
}

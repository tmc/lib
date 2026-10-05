package cudahotswap

import (
	"errors"
	"fmt"
	"io"
	"math"
	"testing"
)

// ds4RetryOwners is used by the native producer before every fatal compile or
// load return. A failed close retains that owner and blocks dependent cleanup.
type ds4RetryOwners struct {
	owners []io.Closer
}

func (r *ds4RetryOwners) adoptError(err error) error {
	var owner io.Closer
	if errors.As(err, &owner) {
		r.owners = append(r.owners, owner)
	}
	return err
}

func (r *ds4RetryOwners) closeBefore(dependents func() error) error {
	for len(r.owners) > 0 {
		last := len(r.owners) - 1
		if err := r.owners[last].Close(); err != nil {
			return err
		}
		r.owners = r.owners[:last]
	}
	return dependents()
}

type ds4CleanupError struct {
	cause error
	calls int
	fail  bool
}

func (e *ds4CleanupError) Error() string { return e.cause.Error() }
func (e *ds4CleanupError) Unwrap() error { return e.cause }

func (e *ds4CleanupError) Close() error {
	e.calls++
	if e.fail {
		return errors.New("native owner remains live")
	}
	return nil
}

func TestDS4NativeErrorOwners(t *testing.T) {
	for _, shape := range []string{"compile-program", "nil-module-load"} {
		for _, retry := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/retry_%t", shape, retry), func(t *testing.T) {
				original := errors.New(shape)
				owner := &ds4CleanupError{cause: original, fail: retry}
				wrapped := fmt.Errorf("native operation: %w", owner)
				var retained ds4RetryOwners
				if err := retained.adoptError(wrapped); err != wrapped || !errors.Is(err, original) {
					t.Fatal("admission replaced original native failure")
				}
				var closed []string
				dependents := func() error {
					closed = append(closed, "module", "output", "device")
					return nil
				}
				err := retained.closeBefore(dependents)
				if (err != nil) != retry || owner.calls != 1 {
					t.Fatalf("owner was not adopted before failure: calls=%d, err=%v", owner.calls, err)
				}
				if retry {
					if len(closed) != 0 || len(retained.owners) != 1 {
						t.Fatal("dependent teardown overtook failed retry owner")
					}
					owner.fail = false
					if err := retained.closeBefore(dependents); err != nil || owner.calls != 2 {
						t.Fatalf("retained owner retry: %v", err)
					}
				}
				if len(retained.owners) != 0 || len(closed) != 3 || closed[0] != "module" || closed[2] != "device" {
					t.Fatal("owner cleanup did not precede module/output/device teardown")
				}
			})
		}
	}
}

// ds4GuideKL demonstrates directional KL(teacher || candidate). Production
// integration must additionally validate retained C-teacher provenance.
func ds4GuideKL(teacher, candidate []float32) (float64, error) {
	if len(teacher) == 0 || len(teacher) != len(candidate) {
		return 0, errors.New("logits: empty or unequal vocabulary")
	}
	logProb := func(x []float32) ([]float64, error) {
		max := math.Inf(-1)
		for _, v := range x {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, errors.New("logits: nonfinite value")
			}
			max = math.Max(max, float64(v))
		}
		var sum float64
		for _, v := range x {
			sum += math.Exp(float64(v) - max)
		}
		out := make([]float64, len(x))
		for i, v := range x {
			out[i] = (float64(v) - max) - math.Log(sum)
		}
		return out, nil
	}
	p, err := logProb(teacher)
	if err != nil {
		return 0, err
	}
	q, err := logProb(candidate)
	if err != nil {
		return 0, err
	}
	var kl float64
	for i := range p {
		kl += math.Exp(p[i]) * (p[i] - q[i])
	}
	return kl, nil
}

func Example_ds4DirectionalKL() {
	teacher := []float32{0, 0}
	candidate := []float32{0, 2}
	kl, err := ds4GuideKL(teacher, candidate)
	fmt.Printf("KL(teacher || candidate) = %.6f, error = %v\n", kl, err)
	// Output: KL(teacher || candidate) = 0.433781, error = <nil>
}

func TestDS4GuideKL(t *testing.T) {
	for _, tt := range []struct {
		name string
		a, b []float32
		want float64
		bad  bool
	}{
		{"equal", []float32{1, 2}, []float32{1, 2}, 0, false},
		{"shift", []float32{1, 2}, []float32{101, 102}, 0, false},
		{"wrong", []float32{0, 0}, []float32{0, 2}, math.Log(math.Cosh(1)), false},
		{"large", []float32{math.MaxFloat32, math.MaxFloat32}, []float32{0, 0}, 0, false},
		{"empty", nil, nil, 0, true},
		{"shape", []float32{0}, []float32{0, 1}, 0, true},
		{"nan", []float32{float32(math.NaN())}, []float32{0}, 0, true},
		{"inf", []float32{0}, []float32{float32(math.Inf(1))}, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ds4GuideKL(tt.a, tt.b)
			if (err != nil) != tt.bad || !tt.bad && math.Abs(got-tt.want) > 1e-12 {
				t.Fatalf("got %g, %v; want %g, bad=%v", got, err, tt.want, tt.bad)
			}
		})
	}
	a, _ := ds4GuideKL([]float32{0, 0}, []float32{0, 2})
	b, _ := ds4GuideKL([]float32{0, 2}, []float32{0, 0})
	if math.Abs(a-b) < 0.1 {
		t.Fatal("directional control did not distinguish operands")
	}
}

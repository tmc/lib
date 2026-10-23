package cudaplan_test

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type custodyStream struct {
	calls              *[]string
	drainErr, closeErr error
	destroyed          bool
}

func (s *custodyStream) Synchronize(context.Context) error {
	if s.destroyed {
		panic("synchronize destroyed stream")
	}
	*s.calls = append(*s.calls, "drain")
	return s.drainErr
}

func (s *custodyStream) Close() error {
	if s.destroyed {
		panic("close destroyed stream")
	}
	*s.calls = append(*s.calls, "stream")
	if s.closeErr == nil {
		s.destroyed = true
	}
	return s.closeErr
}

// TestConstructorCustody exercises the cleanup owner used by the native fixture.
// Retrying below is an explicit control; native cleanup makes only one attempt.
func TestConstructorCustody(t *testing.T) {
	for _, stage := range []string{"none", "drain", "stream", "device", "lease"} {
		t.Run(stage, func(t *testing.T) {
			failed := errors.New("injected cleanup failure")
			var calls []string
			stream := &custodyStream{calls: &calls}
			if stage == "drain" {
				stream.drainErr = failed
			}
			if stage == "stream" {
				stream.closeErr = failed
			}
			refuse := true
			deviceClose := func() error {
				calls = append(calls, "device")
				if refuse && stage == "device" {
					return failed
				}
				return nil
			}
			owners := &workflowNativeOwners{}
			owners.add(workflowDevice, "device", deviceClose)
			owners.add(workflowLease, "lease", func() error {
				calls = append(calls, "lease")
				if refuse && stage == "lease" {
					return failed
				}
				return nil
			})
			owners.ownStream(stream)
			err := owners.close()
			want := []string{"drain"}
			if stage != "drain" {
				want = append(want, "stream")
			}
			if stage != "drain" && stage != "stream" {
				want = append(want, "device")
			}
			if stage == "none" || stage == "lease" {
				want = append(want, "lease")
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls = %v, want %v", calls, want)
			}
			if stage == "none" {
				if err != nil {
					t.Fatal(err)
				}
			} else if !errors.Is(err, failed) {
				t.Fatalf("close = %v, want injected error", err)
			}
			refuse = false
			stream.drainErr, stream.closeErr = nil, nil
			calls = nil
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			want = nil
			switch stage {
			case "drain":
				want = []string{"drain", "stream", "device", "lease"}
			case "stream":
				want = []string{"stream", "device", "lease"}
			case "device":
				want = []string{"device", "lease"}
			case "lease":
				want = []string{"lease"}
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("retry calls = %v, want %v", calls, want)
			}
			calls = nil
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			if len(calls) != 0 {
				t.Fatalf("repeated close = %v", calls)
			}
		})
	}
}

func TestConstructorCustodyWithoutStream(t *testing.T) {
	var calls []string
	deviceClose := func() error { calls = append(calls, "device"); return nil }
	owners := &workflowNativeOwners{}
	owners.add(workflowDevice, "device", deviceClose)
	owners.add(workflowLease, "lease", func() error {
		calls = append(calls, "lease")
		return nil
	})
	if err := owners.close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"device", "lease"}; !reflect.DeepEqual(calls, want) {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
	if owners.quiesced {
		t.Fatal("cleanup without a stream recorded a drain")
	}
}

func TestConstructorCustodyMissingDrain(t *testing.T) {
	var calls []string
	deviceClose := func() error { calls = append(calls, "device"); return nil }
	owners := &workflowNativeOwners{}
	owners.add(workflowDevice, "device", deviceClose)
	owners.add(workflowLease, "lease", func() error {
		calls = append(calls, "lease")
		return nil
	})
	owners.add(workflowStream, "stream", func() error { calls = append(calls, "stream"); return nil })
	if err := owners.close(); err == nil {
		t.Fatal("missing drain accepted")
	}
	if len(calls) != 0 {
		t.Fatalf("cleanup continued without drain: %v", calls)
	}
}

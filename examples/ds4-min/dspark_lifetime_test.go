package main

import (
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDSparkAdmissionSuccess(t *testing.T) {
	lookups, unloads := 0, 0
	validate := func(int) error { lookups++; return nil }
	free := func(int) (bool, error) { unloads++; return true, nil }
	published, retained, err := admitDSparkHandle(7, nil, validate, free)
	if err != nil || published != 7 || len(retained) != 0 || lookups != 1 || unloads != 0 {
		t.Fatalf("success: handle=%d retained=%v lookups=%d unloads=%d err=%v", published, retained, lookups, unloads, err)
	}
	published, retained, err = admitDSparkHandle(0, nil, validate, free)
	if err == nil || published != 0 || len(retained) != 0 || lookups != 1 || unloads != 0 {
		t.Fatalf("zero: handle=%d retained=%v lookups=%d unloads=%d err=%v", published, retained, lookups, unloads, err)
	}
}

func TestDSparkAdmissionCompletedUnload(t *testing.T) {
	lookup := errors.New("lookup failed")
	restore := errors.New("context restoration failed")
	calls := 0
	published, retained, err := admitDSparkHandle(7, nil, func(int) error { return lookup }, func(int) (bool, error) {
		calls++
		return true, restore
	})
	if published != 0 || len(retained) != 0 || calls != 1 || !errors.Is(err, lookup) || !errors.Is(err, restore) {
		t.Fatalf("completed unload: %d/%v/%d/%v", published, retained, calls, err)
	}
}

func TestDSparkAdmissionRetainedMutation(t *testing.T) {
	source, err := os.ReadFile("dspark_lifetime.go")
	if err != nil {
		t.Fatal(err)
	}
	tests, err := os.ReadFile("dspark_lifetime_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, mutant := range []bool{false, true} {
		body := string(source)
		if mutant {
			changed := strings.Replace(body, "return zero, []T{handle}, errors.Join(err, cleanup)", "return zero, nil, errors.Join(err, cleanup)", 1)
			if changed == body {
				t.Fatal("retention mutation did not apply")
			}
			body = changed
		}
		dir := t.TempDir()
		for name, data := range map[string][]byte{"dspark_lifetime.go": []byte(body), "dspark_lifetime_test.go": tests} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "dspark_lifetime.go", "dspark_lifetime_test.go", "-run", "^TestDSparkAdmissionCleanup$", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if !mutant && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		if mutant && (err == nil || !strings.Contains(string(out), "published=0 retained=[]")) {
			t.Fatalf("mutant not caught: %v\n%s", err, out)
		}
	}
}

func TestDSparkAdmissionCleanup(t *testing.T) {
	loadErr := errors.New("load failed")
	lookupErr := errors.New("lookup failed")
	unloadErr := errors.New("unload failed")
	for _, test := range []struct {
		name         string
		load, lookup error
	}{
		{"load", loadErr, nil},
		{"lookup", nil, lookupErr},
	} {
		t.Run(test.name, func(t *testing.T) {
			lookups, unloads := 0, 0
			published, retained, err := admitDSparkHandle(7, test.load, func(int) error {
				lookups++
				return test.lookup
			}, func(int) (bool, error) {
				unloads++
				return false, unloadErr
			})
			primary := test.load
			wantLookups := 0
			if primary == nil {
				primary = test.lookup
				wantLookups = 1
			}
			if published != 0 || !reflect.DeepEqual(retained, []int{7}) || !errors.Is(err, primary) || !errors.Is(err, unloadErr) || unloads != 1 || lookups != wantLookups {
				t.Fatalf("published=%d retained=%v lookup=%d unload=%d error=%v", published, retained, lookups, unloads, err)
			}
			retained, err = releaseDSparkHandles(retained, func(int) error { unloads++; return nil })
			if err != nil || len(retained) != 0 || unloads != 2 {
				t.Fatalf("retry: %v/%v/%d", retained, err, unloads)
			}
		})
	}
}

func TestDSparkReleaseRetry(t *testing.T) {
	var calls []int
	free := func(id int) error {
		calls = append(calls, id)
		if id == 2 && len(calls) == 2 {
			return errors.New("planted free failure")
		}
		return nil
	}
	retained, err := releaseDSparkHandles([]int{1, 2, 3}, free)
	if err == nil || !reflect.DeepEqual(retained, []int{2}) {
		t.Fatalf("retained=%v err=%v", retained, err)
	}
	retained, err = releaseDSparkHandles(retained, free)
	if err != nil || len(retained) != 0 || !reflect.DeepEqual(calls, []int{1, 2, 3, 2}) {
		t.Fatalf("calls=%v retained=%v err=%v", calls, retained, err)
	}
}

func TestDSparkDataOffset(t *testing.T) {
	for _, test := range []struct {
		data, tensor, within uint64
		want                 int64
		bad                  bool
	}{
		{100, 200, 300, 600, false},
		{math.MaxInt64, 0, 0, math.MaxInt64, false},
		{math.MaxUint64, 1, 0, 0, true},
		{1, math.MaxUint64, 0, 0, true},
		{1, 2, math.MaxUint64, 0, true},
		{math.MaxInt64, 1, 0, 0, true},
	} {
		got, err := dsparkDataOffset(test.data, test.tensor, test.within)
		if (err != nil) != test.bad || !test.bad && got != test.want {
			t.Fatalf("offset %v = %d,%v", test, got, err)
		}
	}
}

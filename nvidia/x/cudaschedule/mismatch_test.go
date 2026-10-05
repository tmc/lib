package cudaschedule

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestMismatchIdentityReceipt(t *testing.T) {
	for _, tc := range []struct {
		name, receipt string
		valid         bool
	}{
		{"valid", "0 exact-uint64", true},
		{"last index", "18446744073709551615 exact-uint64", true},
		{"overflow", "18446744073709551616 exact-uint64", false},
		{"negative", "-1 exact-uint64", false},
		{"missing", "0", false},
		{"unknown policy", "0 approximate", false},
		{"extra field", "0 exact-uint64 extra", false},
		{"oversize", strings.Repeat("0", 129) + " exact-uint64", false},
		{"duplicate", "0 exact-uint64\n" + mismatchIdentityPrefix + "0 exact-uint64", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := "MISMATCH\n" + mismatchIdentityPrefix + tc.receipt + "\n"
			clean, id, err := takeMismatchIdentity(out)
			if (err == nil) != tc.valid {
				t.Fatalf("identity=%+v error=%v", id, err)
			}
			if tc.valid && (clean != "MISMATCH\n" || id.Policy != "exact-uint64") {
				t.Fatalf("clean=%q identity=%+v", clean, id)
			}
			if !tc.valid && (clean != out || id != nil) {
				t.Fatal("invalid receipt altered evidence")
			}
		})
	}
	clean, id, err := takeMismatchIdentity("MISMATCH at index 42")
	if err != nil || id != nil || clean != "MISMATCH at index 42" {
		t.Fatal("identity inferred from prose")
	}
}

func TestRunRetainsMismatchIdentity(t *testing.T) {
	opts := CampaignOptions{RunVariant: func(context.Context, uint64, bool, string) (RunRecord, error) {
		return RunRecord{Exited: true, Stdout: "MISMATCH\n" + mismatchIdentityPrefix + "0 exact-uint64\n"}, nil
	}}
	r, err := executeRun(context.Background(), opts, 17, "ptx", false, "test", time.Second, "")
	if err != nil || r.Mismatch == nil || r.Mismatch.Index != 0 || r.Mismatch.Policy != "exact-uint64" || r.Stdout != "MISMATCH\n" {
		t.Fatalf("record=%+v error=%v", r, err)
	}
	opts.RunVariant = func(context.Context, uint64, bool, string) (RunRecord, error) {
		return RunRecord{Exited: true, Stdout: mismatchIdentityPrefix + "0 unsupported\n"}, nil
	}
	r, err = executeRun(context.Background(), opts, 17, "ptx", false, "test", time.Second, "")
	if err != nil || r.HarnessErr == "" || r.Mismatch != nil {
		t.Fatalf("malformed receipt accepted: %+v %v", r, err)
	}
}

func ExampleMismatchIdentity() {
	id := MismatchIdentity{Index: 0, Policy: "exact-uint64"}
	fmt.Println(id.Index, id.Policy)
	// Output: 0 exact-uint64
}

func FuzzMismatchIdentity(f *testing.F) {
	f.Add("0 exact-uint64")
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 256 {
			t.Skip()
		}
		out := mismatchIdentityPrefix + s + "\n"
		clean, id, err := takeMismatchIdentity(out)
		if err != nil && (id != nil || clean != out) {
			t.Fatal("invalid receipt changed evidence")
		}
		if id != nil && id.Policy != "exact-uint64" {
			t.Fatal("unsupported policy accepted")
		}
	})
}

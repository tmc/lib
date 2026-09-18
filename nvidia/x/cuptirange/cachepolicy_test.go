package cuptirange

import (
	"strings"
	"testing"
)

// qualifiedVersions is the allowlist in qualifiedLayout. A flush policy is only
// reachable for these; every other version is turned away earlier, by the
// layout check, for a different and accurate reason.
var qualifiedVersions = []uint32{24, 26, 27, 28, 130001, 130101, 130200, 130202, 130301, 130400}

// TestFlushPoliciesAreUnreachable states in a test what the code states in a
// comment: of the versions that get as far as the policy check, none admits a
// flush. If one ever does, this fails and the message stops being wrong in the
// other direction.
func TestFlushPoliciesAreUnreachable(t *testing.T) {
	var reached int
	for _, v := range qualifiedVersions {
		for _, p := range []CachePolicy{CacheFlushL2, CacheFlushAll} {
			err := CachePolicySupport(v, p)
			if err == nil {
				t.Errorf("CachePolicySupport(%d, %s) = nil; a flush policy became reachable", v, p)
				continue
			}
			if strings.Contains(err.Error(), "unqualified cache policy version") {
				t.Errorf("CachePolicySupport(%d, %s) was turned away by the layout check; %d is not in the qualified set after all", v, p, v)
				continue
			}
			reached++
			if !strings.Contains(err.Error(), "every CUPTI version") {
				t.Errorf("CachePolicySupport(%d, %s) = %q; message should say the state is permanent", v, p, err)
			}
		}
	}
	if reached == 0 {
		t.Fatal("no call reached the policy check; this test proves nothing")
	}
}

// TestNonFlushPolicyIsAdmitted is the positive control. Without it the test
// above would still pass if CachePolicySupport rejected every input.
func TestNonFlushPolicyIsAdmitted(t *testing.T) {
	var admitted int
	for _, v := range qualifiedVersions {
		if CachePolicySupport(v, CacheDefault) == nil {
			admitted++
		}
	}
	if admitted == 0 {
		t.Fatal("no qualified version admits CacheDefault; the instrument rejects everything")
	}
}

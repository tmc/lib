package cudaschedule

import (
	"fmt"
	"strconv"
	"strings"
)

// MismatchIdentity identifies a logical output element and its comparison.
// Currently Policy must be "exact-uint64": equality of unsigned 64-bit values.
// The zero value is invalid. This is comparison identity, not a verdict.
type MismatchIdentity struct {
	Index  uint64 `json:"index"`
	Policy string `json:"policy"`
}

const mismatchIdentityPrefix = "CUDASCHEDULE_MISMATCH_IDENTITY:"

// takeMismatchIdentity consumes one bounded runner receipt. Missing receipts
// are permitted for legacy runners; malformed or duplicate receipts refuse.
func takeMismatchIdentity(stdout string) (string, *MismatchIdentity, error) {
	var clean strings.Builder
	var identity *MismatchIdentity
	for rest := stdout; rest != ""; {
		line, tail, found := strings.Cut(rest, "\n")
		rest = tail
		text := strings.TrimSuffix(line, "\r")
		if strings.HasPrefix(text, mismatchIdentityPrefix) {
			if identity != nil || len(text) > 128 {
				return stdout, nil, fmt.Errorf("invalid or duplicate mismatch identity receipt")
			}
			fields := strings.Fields(strings.TrimPrefix(text, mismatchIdentityPrefix))
			if len(fields) != 2 || fields[1] != "exact-uint64" {
				return stdout, nil, fmt.Errorf("unsupported mismatch identity receipt")
			}
			index, err := strconv.ParseUint(fields[0], 10, 64)
			if err != nil {
				return stdout, nil, fmt.Errorf("mismatch identity index: %w", err)
			}
			identity = &MismatchIdentity{Index: index, Policy: fields[1]}
			continue
		}
		clean.WriteString(line)
		if found {
			clean.WriteByte('\n')
		}
	}
	return clean.String(), identity, nil
}

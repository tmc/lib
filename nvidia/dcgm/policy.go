package dcgm

import "encoding/binary"

// PolicyBool creates a boolean condition parameter using the pinned BOOL tag.
func PolicyBool(value bool) PolicyConditionParams {
	p := PolicyConditionParams{Tag: 0}
	if value {
		var union [8]byte
		binary.NativeEndian.PutUint32(union[:4], 1)
		p.Val[0] = binary.NativeEndian.Uint64(union[:])
	}
	return p
}

// PolicyNumber creates an unsigned threshold using the pinned LLONG tag.
func PolicyNumber(value uint64) PolicyConditionParams {
	return PolicyConditionParams{Tag: 1, Val: [1]uint64{value}}
}

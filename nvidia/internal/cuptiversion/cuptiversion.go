// Package cuptiversion maps a runtime CUPTI version to the qualified
// version whose struct layouts a package should use.
//
// CUPTI reports CUDA 12 releases as small integers (24 is 12.6, 28 is 12.9)
// and CUDA 13 and later as MMmmpp (130400 is 13.4.0). Within one major
// release CUPTI parameter structs only grow and carry their own size, so a
// runtime accepts the layout of any earlier release in the same major.
// Layout relies on that: a version that was not qualified exactly uses the
// newest qualified layout at or below it in the same major. A new major,
// or a version older than every qualified one, has no layout.
package cuptiversion

// Layout returns the version in qualified whose layouts v should use,
// and whether there is one. The result equals v when v was qualified.
func Layout(v uint32, qualified ...uint32) (uint32, bool) {
	var best uint32
	var ok bool
	for _, q := range qualified {
		if q == v {
			return v, true
		}
		if q < v && major(v) != 0 && major(q) == major(v) && (!ok || q > best) {
			best, ok = q, true
		}
	}
	return best, ok
}

// major returns the CUDA major release of a CUPTI version, or 0 for a
// number neither scheme produces. Versions below 1000 are the CUDA 12
// numbering; MMmmpp starts at CUDA 13.
func major(v uint32) uint32 {
	switch {
	case v < 1000:
		return 12
	case v < 130000:
		return 0
	}
	return v / 10000
}

// AdmittedLayout returns the recorded layout for an explicitly admitted runtime.
// Unlike Layout, it does not admit gaps or future releases. Runtime 130401
// retains its historical source admission with layout 130400; this alias is
// not fresh physical qualification.
func AdmittedLayout(v uint32, qualified ...uint32) (uint32, bool) {
	if v == 130401 {
		v = 130400
	}
	for _, q := range qualified {
		if q == v {
			return q, true
		}
	}
	return 0, false
}

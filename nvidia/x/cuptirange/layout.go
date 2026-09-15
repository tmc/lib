package cuptirange

// The Range Profiler's two host parameter structs grew at different releases, so
// a version cannot be classified once and applied to both. Measured in
// cupti_profiler_host.h across the five CUDA 13 toolkits installed on the GB10
// host, whose CUPTI_API_VERSION values are 130001, 130101, 130202, 130301 and
// 130400:
//
//	CUPTI   Initialize_Params ends at  GetMetricProperties_Params ends at
//	24      pHostObject                metricType
//	26      pHostObject                metricType
//	27      pHostObject                metricType
//	28      pHostObject                metricType
//	130001  pHostObject                metricType
//	130101  pHostObject                metricCollectionScope
//	130200  pSinglePassMetricSetName   metricCollectionScope
//	130202  pSinglePassMetricSetName   metricCollectionScope
//	130301  pSinglePassMetricSetName   metricCollectionScope
//	130400  pSinglePassMetricSetName   metricCollectionScope
//
// 130101 is why the two are separate: CUDA 13.1 appends metricCollectionScope to
// the properties struct and leaves the initialize struct at its CUDA 13.0 shape.
// A single scoped-or-not predicate would have to pick one of the two wrong
// answers for it, and declaring a size a version's struct does not have is a
// native memory error rather than a mismatch, because CUPTI writes these fields
// in place through the pointer it is given.
//
// 130200 is cupti.ApiVersion, the version the bindings in nvidia/cupti were
// generated from, and its struct shapes are the ones the Go types express. No
// library on the hosts measured here reports it; the 13.2 toolkit installed
// reports 130202, with the same two shapes.
//
// The GB10 host's CUDA 12 libraries reported 24, 26 and 28. D304CF8D's
// 2026-09-15 L4 report independently measured CUDA 12.9.41 reporting 27, with
// cupti_profiler_host.h MD5 6d969e3b23411a1c39c9804f4182b6e0, identical to the
// measured 24/26/28 header. All four use the base shapes. These are observed
// builds, not a universal mapping from toolkit releases to API versions.
//
// Every version absent from the table fails closed: a struct size inferred from
// a version number has no witness.
func qualifiedLayout(version uint32) bool {
	switch version {
	case 24, 26, 27, 28, 130001, 130101, 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// singlePassInitialize reports whether version's Initialize_Params carries
// pSinglePassMetricSetName as its final field.
func singlePassInitialize(version uint32) bool {
	switch version {
	case 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// scopedProperties reports whether version's GetMetricProperties_Params carries
// metricCollectionScope as its final field, which CUPTI fills only when the
// declared size reaches it.
func scopedProperties(version uint32) bool {
	return version == 130101 || singlePassInitialize(version)
}

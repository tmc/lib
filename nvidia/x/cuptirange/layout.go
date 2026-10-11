package cuptirange

import "github.com/tmc/lib/nvidia/internal/cuptiversion"

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
// 130200 is the historical version the bindings in nvidia/cupti were
// generated from, and its struct shapes are the ones the Go types express. No
// library on the hosts measured here reports it; the 13.2 toolkit installed
// reports 130202, with the same two shapes.
//
// The GB10 host's CUDA 12 libraries reported 24, 26 and 28. peer-b's
// 2026-09-15 L4 report independently measured CUDA 12.9.41 reporting 27, with
// cupti_profiler_host.h MD5 6d969e3b23411a1c39c9804f4182b6e0, identical to the
// measured 24/26/28 header. All four use the base shapes. These are observed
// builds, not a universal mapping from toolkit releases to API versions.
//
// Admission is exact. Runtime 130401 retains historical source admission
// with layout 130400; this is not fresh physical qualification.
func layoutVersion(version uint32) (uint32, bool) {
	return cuptiversion.AdmittedLayout(version, 24, 26, 27, 28, 130001, 130101, 130200, 130202, 130301, 130400)
}

// qualifiedLayout reports whether version has a layout.
func qualifiedLayout(version uint32) bool {
	_, ok := layoutVersion(version)
	return ok
}

// singlePassInitialize reports whether version's Initialize_Params layout
// carries pSinglePassMetricSetName as its final field.
func singlePassInitialize(version uint32) bool {
	l, _ := layoutVersion(version)
	switch l {
	case 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// scopedProperties reports whether version's GetMetricProperties_Params layout
// carries metricCollectionScope as its final field, which CUPTI fills only when
// the declared size reaches it.
func scopedProperties(version uint32) bool {
	l, _ := layoutVersion(version)
	return l == 130101 || singlePassInitialize(version)
}

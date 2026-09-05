package cutensornet

// WorksizePrefMax asks for a workspace large enough that every algorithm is
// available for every pairwise contraction. It is the third and largest of the
// workspace preferences, above [WorksizePrefRecommended].
//
// It is declared here rather than in enums.gen.go because the C parser shared
// by this project's generators drops enumerators whose name ends in _MAX,
// treating them as the count-sentinel that many CUDA headers put at the end of
// an enum. CUTENSORNET_WORKSIZE_PREF_MAX is not a sentinel: cutensornet.h gives
// it the value 2 and documents it as a preference like the other two. The value
// is pinned by TestWorksizePrefValues in abi_test.go.
//
// [WorksizePref.String] does not know this constant and renders it as
// "WorksizePref(2)".
const WorksizePrefMax WorksizePref = 2

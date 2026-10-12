// Package ncclras reads bounded NCCL RAS status and queries an explicit TCP address.
//
// It supports NCCL 2.29.1 and 2.32.3 JSON status over client protocol 2. The
// 2.29.1 fixture is NVIDIA's documentation example; the 2.32.3 fixture is
// constructed from the pinned 2.32.3 emitter source, and no captured 2.32.3
// response has been checked. ReadText and Client.StatusText implement a separate,
// experimental NCCL 2.24.3 uniform-job aggregate profile. That profile has
// producer-source and constructed-fixture coverage; authentic capture validation
// remains pending. UnsupportedError retains a version only when reported.
// A parsed response is a collected observation, not a health verdict:
// missing ranks and collection timeouts remain visible. Raw retains unknown
// fields. No connection is opened until a Client method is called.
//
// The supported source and example provenance is in testdata/README.md.
package ncclras

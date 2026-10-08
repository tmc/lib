// Package ncclras reads bounded NCCL RAS status and queries an explicit TCP address.
//
// It supports the documented NCCL 2.29.1 and pinned 2.32.3 status shapes over
// client protocol 2. ReadText and Client.StatusText implement a separate,
// experimental NCCL 2.24.3 uniform-job aggregate profile. That profile has
// producer-source and constructed-fixture coverage; authentic capture validation
// remains pending. UnsupportedError retains a version only when reported.
// A parsed response is a collected observation, not a health verdict:
// missing ranks and collection timeouts remain visible. Raw retains unknown
// fields. No connection is opened until a Client method is called.
//
// The supported source and example provenance is in testdata/README.md.
package ncclras

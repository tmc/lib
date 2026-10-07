// Package ncclras reads NCCL RAS status JSON and queries an explicit TCP address.
//
// It supports the documented NCCL 2.29.1 and pinned 2.32.3 status shapes over
// client protocol 2. Text status, monitoring and control commands are not
// supported. A parsed response is a collected observation, not a health verdict:
// missing ranks and collection timeouts remain visible. Raw retains unknown
// fields. No connection is opened until Client.Status is called.
//
// The supported source and example provenance is in testdata/README.md.
package ncclras

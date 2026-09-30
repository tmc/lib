// Package isolation validates sampled GPU isolation receipts independently of
// collector completeness. Receipts are local observer evidence, not attestation.
// A lock declaration alone never establishes sampled clear.
package isolation

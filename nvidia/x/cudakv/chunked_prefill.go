// SPDX-License-Identifier: Apache-2.0

package cudakv

// PrefillChunk describes prompt tokens consumed by one forward pass.
// Scheduling and completion belong to cudabatch.BatchScheduler.
type PrefillChunk struct {
	SeqID    string
	Tokens   []int32
	StartPos int
	IsLast   bool
}

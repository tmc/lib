// Package stack retains bounded, explicit language stack attribution.
// Frames are stored leaf first. References identify CUPTI external IDs, not
// timestamps; scope_entry attribution must not be interpreted as submission.
package stack

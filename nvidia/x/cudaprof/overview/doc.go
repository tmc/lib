// Package overview groups retained CUDA activity by explicit application
// context and exposes token and operation summaries without assigning missing
// delivery or materialization observations. Optional kernel-marker intervals
// are labeled inferred and remain separate from observed tokens.
//
// Activity references are capture-local indexes, qualified by their journal.
// Parent metrics use unique references, so nesting does not multiply work.
package overview

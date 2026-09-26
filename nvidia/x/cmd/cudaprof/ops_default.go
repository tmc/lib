package main

import (
	_ "embed"
	"strings"
)

// defaultOps is the -ops value that names the built-in rules in
// defaultops.txt instead of a file.
const defaultOps = "default"

//go:embed defaultops.txt
var defaultOpsText string

// parseDefaultOpRules returns the built-in op groups.
func parseDefaultOpRules() (opRules, error) {
	return parseOpRules(strings.NewReader(defaultOpsText))
}

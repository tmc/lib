package main

import _ "embed"

// defaultOps is the -ops value that names the built-in rules in
// defaultops.txt instead of a file.
const defaultOps = "default"

//go:embed defaultops.txt
var defaultOpsText string

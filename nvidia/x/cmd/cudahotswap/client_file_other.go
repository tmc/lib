//go:build !unix

package main

import "os"

func openClientSource(path string) (*os.File, error) { return os.Open(path) }

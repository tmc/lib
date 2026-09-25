//go:build !unix

package main

// signalName returns desc unchanged: only unix names its signals.
func signalName(desc string) string { return desc }

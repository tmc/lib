// Package metal owns a Metal device and its default command queue and provides
// buffer and pipeline construction helpers.
//
// Raw, Queue, and objects constructed from a Device are borrowed from or
// derived from that Device. Callers must finish command buffers and release
// derived objects before closing the Device. Device methods may be called
// concurrently, but borrowed objects must not be used concurrently with Close.
package metal

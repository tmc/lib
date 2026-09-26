// Package nvrtc is a declaration-only stand-in for the real nvrtc package.
package nvrtc

type Program uintptr

func CreateProgram(src, name string) (Program, error) { return 0, nil }
func (p Program) Destroy() error                      { return nil }
func (p Program) Compile(options ...string) error     { return nil }

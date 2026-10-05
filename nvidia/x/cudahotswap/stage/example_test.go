package stage

import (
	"context"
	"fmt"
)

func Example() {
	ctx, err := NewContext(nil, 0, nil)
	fmt.Println(err, ctx.BufferCount, ctx.FunctionCount)
	// Output: <nil> 0 0
}

func ExampleNewContext() {
	_, err := NewContext(make([]Buffer, 65), 0, nil)
	fmt.Println(err)
	// Output: stage context exceeds 64 named buffers or functions
}

func ExampleLibrary_Pin() {
	var library Library
	_, err := library.Pin()
	fmt.Println(err)
	// Output: pin stage: library is unavailable
}

func ExamplePin_Call() {
	var pin *Pin
	fmt.Println(pin.Call(nil))
	// Output: call stage: nil pin
}

func ExamplePin_Release() {
	var pin *Pin
	fmt.Println(pin.Release())
	// Output: <nil>
}

func ExampleLibrary_Close() {
	var library Library
	fmt.Println(library.Close())
	// Output: <nil>
}

func ExampleNewBuilder() {
	_, err := NewBuilder("", "cc")
	fmt.Println(err)
	// Output: create stage builder: directory and compiler are required
}

func ExampleBuilder_Build() {
	var builder *Builder
	_, err := builder.Build(context.Background(), Source{})
	fmt.Println(err)
	// Output: build stage: context, builder and code are required
}

func ExampleLoad() {
	_, err := Load(nil)
	fmt.Println(err)
	// Output: load stage: invalid artifact
}

func ExampleArtifact_Path() {
	var artifact *Artifact
	fmt.Printf("%q\n", artifact.Path())
	// Output: ""
}

func ExampleArtifact_Generation() {
	var artifact *Artifact
	fmt.Println(artifact.Generation())
	// Output: 0
}

func ExampleArtifact_SourceDigest() {
	var artifact *Artifact
	fmt.Printf("%q\n", artifact.SourceDigest())
	// Output: ""
}

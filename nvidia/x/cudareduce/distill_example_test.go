package cudareduce_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudareduce"
)

func Example() {
	c := cudareduce.SparseTeacherConfig{Rows: 1, Vocabulary: 3, Retained: 1, Temperature: 1}
	fmt.Println(cudareduce.ValidateSparseTeacher(c, []float32{0, 1, 2}, []int32{1}, []float32{2}, []int32{1}))
	// Output: <nil>
}

func ExampleSparseTeacherConfig_Extents() {
	c := cudareduce.SparseTeacherConfig{Rows: 2, Vocabulary: 129, Retained: 16, Temperature: 2}
	e, err := c.Extents()
	fmt.Println(e.Logits, e.Targets, e.Stats, err)
	// Output: 258 32 8 <nil>
}
func ExampleValidateSparseTeacher() {
	c := cudareduce.SparseTeacherConfig{Rows: 1, Vocabulary: 3, Retained: 1, Temperature: 1}
	fmt.Println(cudareduce.ValidateSparseTeacher(c, []float32{0, 1, 2}, []int32{1}, []float32{2}, []int32{1}))
	// Output: <nil>
}
func ExampleSparseTeacherSchema() {
	schema, err := cudareduce.SparseTeacherSchema()
	fmt.Println(len(schema.Params), err)
	// Output: 15 <nil>
}
func ExampleCheckSparseTeacherStatus() {
	fmt.Println(cudareduce.CheckSparseTeacherStatus([]int32{0, 0}, 2))
	// Output: <nil>
}
func ExampleCompileSparseTeacherArtifact() {
	_, err := cudareduce.CompileSparseTeacherArtifact("compute_121")
	fmt.Println(err != nil)
	// Output: true
}
func ExampleCompileSparseTeacher() {
	_, err := cudareduce.CompileSparseTeacher(nil, cudareduce.SparseTeacherConfig{Rows: 1, Vocabulary: 3, Retained: 1, Temperature: 1})
	fmt.Println(err != nil)
	// Output: true
}
func ExampleLoadSparseTeacher() {
	_, err := cudareduce.LoadSparseTeacher(nil, cudareduce.SparseTeacherConfig{Rows: 1, Vocabulary: 3, Retained: 1, Temperature: 1}, nil)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleSparseTeacher_Execute() {
	var owner cudareduce.SparseTeacher
	fmt.Println(owner.Execute(context.Background(), nil, cudareduce.SparseTeacherBuffers{}) != nil)
	// Output: true
}
func ExampleSparseTeacher_Close() {
	var owner cudareduce.SparseTeacher
	fmt.Println(owner.Close())
	// Output: <nil>
}
func ExampleSparseTeacher_Artifact() {
	var owner cudareduce.SparseTeacher
	fmt.Println(owner.Artifact() == nil)
	// Output: true
}

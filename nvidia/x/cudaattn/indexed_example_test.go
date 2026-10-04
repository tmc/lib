package cudaattn_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudablock"
)

func ExampleIndexedConfig_Extents() {
	c := cudaattn.IndexedConfig{Batch: 1, Queries: 1, QueryHeads: 2, KVHeads: 1, HeadDim: 8, BlockSize: 4, MaxSequence: 9, PhysicalBlocks: 3, SelectedBlocks: 2, Selection: cudaattn.IndexedQueryHeads, Scale: 0.5, Causal: true}
	e, err := c.Extents()
	fmt.Println(e.Query, e.KV, e.IDs, e.MaxBlocks, err)
	// Output: 16 96 4 3 <nil>
}

func ExampleValidateIndexedRequest() {
	c := cudaattn.IndexedConfig{Batch: 1, Queries: 1, QueryHeads: 1, KVHeads: 1, HeadDim: 2, BlockSize: 2, MaxSequence: 3, PhysicalBlocks: 2, SelectedBlocks: 2, Selection: cudaattn.IndexedQueryHeads, Scale: 1, Causal: true}
	r := cudaattn.IndexedRequest{Sequences: []int32{0}, Positions: []int32{2}, Lengths: []int32{3}, Pages: []int32{1, 0}, IDs: []int32{1, -1}, Counts: []int32{1}}
	fmt.Println(cudaattn.ValidateIndexedRequest(c, r))
	// Output: <nil>
}

func ExampleIndexedTopKConfig() {
	c := cudaattn.IndexedConfig{Batch: 1, Queries: 1, QueryHeads: 2, KVHeads: 1, HeadDim: 8, BlockSize: 4, MaxSequence: 9, PhysicalBlocks: 3, SelectedBlocks: 2, Selection: cudaattn.IndexedQueryHeads, Scale: 0.5}
	s, err := cudaattn.IndexedTopKConfig(c)
	fmt.Println(s.Rows, s.Width, s.K, err)
	// Output: 2 3 2 <nil>
}

func ExampleIndexedSchema() {
	s, err := cudaattn.IndexedSchema()
	fmt.Println(len(s.Params), err)
	// Output: 28 <nil>
}

func ExampleCheckIndexedStatus() {
	fmt.Println(cudaattn.CheckIndexedStatus([]int32{0, 0}, 2))
	// Output: <nil>
}

func ExampleIndexedAttention_Artifact() {
	var a cudaattn.IndexedAttention
	fmt.Println(a.Artifact() == nil)
	// Output: true
}

func ExampleIndexedAttention_Close() {
	var a cudaattn.IndexedAttention
	fmt.Println(a.Close())
	// Output: <nil>
}

func ExampleIndexedAttention_Execute() {
	var a cudaattn.IndexedAttention
	fmt.Println(a.Execute(nil, nil, cudaattn.IndexedBuffers{}))
	// Output: cudaattn: indexed closed or awaiting drain
}

func ExampleIndexedAttention_ExecuteSelection() {
	var a cudaattn.IndexedAttention
	fmt.Println(a.ExecuteSelection(nil, nil, cudaattn.IndexedBuffers{}, nil, cudablock.TopKBuffers{}))
	// Output: cudaattn: missing indexed topk selector
}

func ExampleCompileIndexedArtifact() {
	_, err := cudaattn.CompileIndexedArtifact("compute_121")
	fmt.Println(err)
	// Output: cudaattn: invalid indexed physical target
}

func ExampleCompileIndexedAttention() {
	c := cudaattn.IndexedConfig{Batch: 1, Queries: 1, QueryHeads: 1, KVHeads: 1, HeadDim: 2, BlockSize: 2, MaxSequence: 3, PhysicalBlocks: 2, SelectedBlocks: 2, Selection: cudaattn.IndexedQueryHeads, Scale: 1}
	_, err := cudaattn.CompileIndexedAttention(nil, c)
	fmt.Println(err)
	// Output: cudaattn: nil indexed device
}

func ExampleLoadIndexedAttention() {
	c := cudaattn.IndexedConfig{Batch: 1, Queries: 1, QueryHeads: 1, KVHeads: 1, HeadDim: 2, BlockSize: 2, MaxSequence: 3, PhysicalBlocks: 2, SelectedBlocks: 2, Selection: cudaattn.IndexedQueryHeads, Scale: 1}
	_, err := cudaattn.LoadIndexedAttention(nil, c, nil)
	fmt.Println(err)
	// Output: cudaattn: nil indexed device
}

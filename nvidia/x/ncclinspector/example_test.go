package ncclinspector_test

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func ExampleRead() {
	f, err := os.Open("testdata/stats.jsonl")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	r, err := ncclinspector.Read(f, ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(r[0].Header.Rank, r[0].Stats.CollectiveDropped)
	// Output: 2 512
}

func ExampleLimits() {
	l := ncclinspector.Limits{Records: 128}
	fmt.Println(l.Records)
	// Output: 128
}

func ExampleHeader() {
	h := ncclinspector.Header{ID: "comm", Rank: 1, Ranks: 2, Nodes: 1}
	fmt.Println(h.ID, h.Rank)
	// Output: comm 1
}

func ExampleMetadata() {
	m := ncclinspector.Metadata{Version: "v4.1", PID: 42}
	fmt.Println(m.Version, m.PID)
	// Output: v4.1 42
}

func ExampleCollective() {
	c := ncclinspector.Collective{Name: "AllReduce", Sequence: 3}
	fmt.Println(c.Name, c.Sequence)
	// Output: AllReduce 3
}

func ExamplePointToPoint() {
	p := ncclinspector.PointToPoint{Name: "Send", Sequence: 3, Peer: 1}
	fmt.Println(p.Name, p.Sequence, p.Peer)
	// Output: Send 3 1
}

func ExampleDumpStats() {
	s := ncclinspector.DumpStats{Collectives: 4, CollectiveDroppedTotal: 3, CollectiveDropped: 1}
	fmt.Println(s.Collectives, s.CollectiveDropped)
	// Output: 4 1
}

func ExampleRecord() {
	r := ncclinspector.Record{Collective: &ncclinspector.Collective{Sequence: 3}}
	fmt.Println(r.Stats == nil)
	// Output: true
}

func ExampleNewDecoder() {
	b, err := os.ReadFile("testdata/stats.jsonl")
	if err != nil {
		panic(err)
	}
	var input bytes.Buffer
	input.Write(b[:len(b)-1])
	d, err := ncclinspector.NewDecoder(&input, ncclinspector.Limits{})
	if err != nil {
		panic(err)
	}
	_, err = d.Next()
	fmt.Println(err == io.EOF, d.Offset())
	input.WriteByte('\n')
	r, err := d.Next()
	if err != nil {
		panic(err)
	}
	_, err = d.Next()
	fmt.Println(r.Stats.CollectiveDropped, err == io.EOF, d.Finish() == nil)
	// Output:
	// true 0
	// 512 true true
}

func ExampleOpenFile() {
	d, err := ncclinspector.OpenFile("testdata/stats.jsonl", ncclinspector.Limits{}, nil)
	if err != nil {
		panic(err)
	}
	_, err = d.Next()
	if err != nil {
		panic(err)
	}
	position, err := d.Position()
	if err != nil {
		panic(err)
	}
	if err = d.Close(); err != nil {
		panic(err)
	}
	d, err = ncclinspector.OpenFile("testdata/stats.jsonl", ncclinspector.Limits{}, &position)
	if err != nil {
		panic(err)
	}
	_, err = d.Next()
	fmt.Println(position.Offset() > 0, err == io.EOF, d.Finish() == nil)
	if err = d.Close(); err != nil {
		panic(err)
	}
	// Output: true true true
}

func ExampleFilePosition_Offset() {
	var p ncclinspector.FilePosition
	fmt.Println(p.Offset())
	// Output: 0
}

package cudainject_test

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudainject"
)

func Example() {
	var stream bytes.Buffer
	_ = cudainject.WriteRecord(&stream, cudainject.Record{Kind: cudainject.KindCapability, Sequence: 1, Dropped: 2}, cudainject.Limits{})
	decoder := cudainject.NewDecoder(&stream, cudainject.Limits{})
	record, _ := decoder.Next()
	fmt.Println(record.Sequence, decoder.Stats().Dropped)
	// Output: 1 2
}

func ExampleNewDecoder() {
	_, err := cudainject.NewDecoder(bytes.NewReader(nil), cudainject.Limits{}).Next()
	fmt.Println(err)
	// Output: EOF
}

func ExampleWriteRecord() {
	var stream bytes.Buffer
	err := cudainject.WriteRecord(&stream, cudainject.Record{Sequence: 1}, cudainject.Limits{})
	fmt.Println(err, stream.Len())
	// Output: <nil> 80
}

func ExampleReadJournal() {
	dir, _ := os.MkdirTemp("", "cudainject-example-")
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "journal")
	_ = os.WriteFile(path, nil, 0600)
	recovery, err := cudainject.ReadJournal(path, cudainject.Limits{})
	fmt.Println(recovery.Complete, len(recovery.Records), err)
	// Output: true 0 <nil>
}

func ExampleReadCapture() {
	_, err := cudainject.ReadCapture(bytes.NewReader(nil), cudacapture.Limits{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleJoinCorrelations() {
	joins, err := cudainject.JoinCorrelations([]cudainject.Record{{Kind: cudainject.KindAPI, Payload: []byte(`{"domain":"runtime","thread_id":5,"correlation":8}`)}})
	fmt.Println(joins.MissingActivity, err)
	// Output: [8] <nil>
}

func ExampleBuildAgent() {
	// A real build uses a deadline, absolute compiler/toolkit paths and a fresh
	// output directory. An unbounded context is refused before any file writes.
	_, err := cudainject.BuildAgent(context.Background(), cudainject.BuildOptions{})
	fmt.Println(err)
	// Output: cudainject: build context requires deadline
}

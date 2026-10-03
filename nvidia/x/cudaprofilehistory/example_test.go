package cudaprofilehistory_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
)

func Example() {
	s, err := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 10, MaxBytes: 4096, MaxRecordBytes: 2048})
	if err != nil {
		fmt.Println(err)
		return
	}
	var b bytes.Buffer
	s.WriteSnapshot(&b)
	fmt.Print(b.String()) // Output: {"version":1,"count":0,"rows_digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
}
func ExampleNewStore() {
	_, err := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{})
	fmt.Println(err != nil) // Output: true
}
func ExampleRecord_Validate() {
	fmt.Println((cudaprofilehistory.Record{}).Validate() != nil) // Output: true
}
func ExampleObservationKey() {
	_, err := cudaprofilehistory.ObservationKey(cudaprofilehistory.Record{})
	fmt.Println(err != nil) // Output: true
}
func ExampleStore_Append() {
	s, _ := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 10, MaxBytes: 4096, MaxRecordBytes: 2048})
	_, err := s.Append(cudaprofilehistory.Record{})
	fmt.Println(err != nil) // Output: true
}
func ExampleStore_Records() {
	s, _ := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 10, MaxBytes: 4096, MaxRecordBytes: 2048})
	rows, err := s.Records()
	fmt.Println(len(rows), err) // Output: 0 <nil>
}
func ExampleStore_WriteSnapshot() {
	s, _ := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 10, MaxBytes: 4096, MaxRecordBytes: 2048})
	var b bytes.Buffer
	s.WriteSnapshot(&b)
	fmt.Print(b.String()) // Output: {"version":1,"count":0,"rows_digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
}
func ExampleReadSnapshot() {
	s, err := cudaprofilehistory.ReadSnapshot(bytes.NewBufferString(`{"version":1,"count":0,"rows_digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}`), cudaprofilehistory.Limits{MaxRecords: 10, MaxBytes: 4096, MaxRecordBytes: 2048})
	if err != nil {
		fmt.Println(err)
		return
	}
	rows, _ := s.Records()
	fmt.Println(len(rows)) // Output: 0
}
func ExampleDecodeRecord() {
	_, err := cudaprofilehistory.DecodeRecord([]byte(`{"version":1}`), 1024)
	fmt.Println(err != nil) // Output: true
}
func ExampleCompare() {
	_, err := cudaprofilehistory.Compare(cudaprofilehistory.Record{}, cudaprofilehistory.Record{}, cudaprofilehistory.Policy{})
	fmt.Println(err != nil) // Output: true
}
func ExamplePlanBisect() {
	p, err := cudaprofilehistory.PlanBisect(cudaprofilehistory.BisectRequest{Candidates: []string{"a", "b", "c"}, Predicate: "correctness", MaxCandidates: 3})
	fmt.Println(p.Next, p.FirstBad == "", err) // Output: b true <nil>
}
func ExampleDecodeBisectRequest() {
	r, err := cudaprofilehistory.DecodeBisectRequest([]byte(`{"candidates":["a"],"predicate":"correctness","max_candidates":1}`), 1024)
	fmt.Println(r.Candidates, err) // Output: [a] <nil>
}

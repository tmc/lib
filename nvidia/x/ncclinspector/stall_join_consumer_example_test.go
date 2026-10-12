package ncclinspector_test

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
)

func Example() {
	m, err := readManifest()
	if err != nil {
		fmt.Println("manifest error")
		return
	}
	files := m.RankFiles["delayed"]
	captures := make([]nccljoin.Capture, 0, 2)
	for rank, file := range files {
		data, err := os.ReadFile(filepath.Join(fixtureDir, file))
		if err != nil {
			fmt.Println("capture error")
			return
		}
		records, err := ncclinspector.Read(bytes.NewReader(data), ncclinspector.Limits{})
		if err != nil || len(records) != 1 || records[0].Header.Rank != rank {
			fmt.Println("rank binding error")
			return
		}
		captures = append(captures, nccljoin.Capture{ID: fmt.Sprintf("delayed-rank%d", rank), Records: records})
	}
	joined, err := nccljoin.Join(captures, nccljoin.Options{
		Scope:    nccljoin.Scope{Capture: m.Scope.Capture, Job: m.Scope.Job, Epoch: m.Scope.Epoch},
		Expected: map[string][]int{m.Hash: m.ExpectedRanks},
	})
	if err != nil || len(joined.Collectives) != 1 {
		fmt.Println("join error")
		return
	}
	group := joined.Collectives[0]
	attr := ncclinspector.NewTwoRankAttributor()
	for _, rankRecord := range group.Observations {
		record := ncclinspector.Record{Header: rankRecord.Header, Metadata: rankRecord.Metadata, Collective: &rankRecord.Collective, Raw: rankRecord.Raw}
		obs, err := ncclinspector.ObservationFromRecord(record, clockFor(m, record.Header.Rank))
		if err != nil || attr.AddObservation(obs) != nil {
			fmt.Println("observation error")
			return
		}
	}
	report, err := attr.Attribute(group.Communicator, group.Sequence)
	if err != nil {
		fmt.Println("attribution error")
		return
	}
	fmt.Printf("%s rank=%d admitted=%t constructed=%t\n", report.Diagnosis, report.DelayedRank, report.Admitted, !m.Authentic)
	// Output: host_enqueue_skew rank=1 admitted=true constructed=true
}

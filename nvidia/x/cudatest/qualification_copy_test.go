package cudatest

import "testing"

func TestNativeQualification_CopyCompletionIsNotKernelLaunch(t *testing.T) {
	for _, tc := range []struct {
		name  string
		alter func(*QualificationFacts)
		valid bool
	}{
		{"synchronous-driver-copy", func(f *QualificationFacts) {}, true},
		{"constructed-plan-only", func(f *QualificationFacts) { f.CopyExecutions, f.CompletedCopyExecutions = 0, 0 }, false},
		{"incomplete-copy", func(f *QualificationFacts) { f.CompletedCopyExecutions = 0 }, false},
		{"excess-completions", func(f *QualificationFacts) { f.CompletedCopyExecutions++ }, false},
		{"unchecked-synchronization", func(f *QualificationFacts) { f.Synchronizations = 0 }, false},
		{"incomplete-optional-kernel", func(f *QualificationFacts) { f.Launches = 1 }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			f.changeRun(t, 0, func(r *QualificationRun) {
				r.Facts.Launches, r.Facts.CompletedLaunches = 0, 0
				tc.alter(&r.Facts)
			})
			err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files)
			if (err == nil) != tc.valid {
				t.Fatalf("admitted=%t want%t: %v", err == nil, tc.valid, err)
			}
		})
	}
	// The copy-specific completion evidence never admits unexecuted kernels
	// for an address-space, DSMEM or CUPTI row.
	for i, kind := range []string{"address-space", "dsmem", "cupti-range", "cupti-pc"} {
		t.Run(kind, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			f.changeRun(t, i+1, func(r *QualificationRun) {
				r.Facts.Launches, r.Facts.CompletedLaunches = 0, 0
				r.Facts.CopyExecutions, r.Facts.CompletedCopyExecutions = 1, 1
			})
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("driver copy completion admitted an unexecuted kernel row")
			}
		})
	}
}

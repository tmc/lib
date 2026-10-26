package cudasass_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestTargetAlignmentBoundary(t *testing.T) {
	tests := []struct {
		name       string
		header     string
		addr       string
		wantTarget string
		wantErr    string
	}{
		// False prefixes that are not target declarations default to 16-byte alignment.
		{
			name:    "false_prefix_targeted_sm60_0008_rejected",
			header:  ".targeted sm_60\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "false_prefix_targeted_sm60_0010_accepted",
			header:     ".targeted sm_60\n",
			addr:       "0010",
			wantTarget: "",
		},
		{
			name:    "false_prefix_target_foo_sm60_0008_rejected",
			header:  ".target_foo sm_60\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "false_prefix_target_foo_sm60_0010_accepted",
			header:     ".target_foo sm_60\n",
			addr:       "0010",
			wantTarget: "",
		},
		{
			name:    "false_prefix_targetsm60_0008_rejected",
			header:  ".targetsm_60\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "false_prefix_targetsm60_0010_accepted",
			header:     ".targetsm_60\n",
			addr:       "0010",
			wantTarget: "",
		},
		{
			name:    "false_prefix_code_format_sm60_0008_rejected",
			header:  "code format sm_60\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "false_prefix_code_format_sm60_0010_accepted",
			header:     "code format sm_60\n",
			addr:       "0010",
			wantTarget: "",
		},

		// Valid pre-sm_70 target declarations use 8-byte alignment.
		{
			name:       "valid_old_space_sm60_0008_accepted",
			header:     ".target sm_60\n",
			addr:       "0008",
			wantTarget: "sm_60",
		},
		{
			name:    "valid_old_space_sm60_0004_rejected",
			header:  ".target sm_60\n",
			addr:    "0004",
			wantErr: "must be a multiple of 8",
		},
		{
			name:       "valid_old_tab_sm60_0008_accepted",
			header:     ".target\tsm_60\n",
			addr:       "0008",
			wantTarget: "sm_60",
		},
		{
			name:    "valid_old_tab_sm60_0004_rejected",
			header:  ".target\tsm_60\n",
			addr:    "0004",
			wantErr: "must be a multiple of 8",
		},
		{
			name:       "valid_old_options_sm60_0008_accepted",
			header:     ".target sm_60, texmode_independent\n",
			addr:       "0008",
			wantTarget: "sm_60",
		},
		{
			name:    "valid_old_options_sm60_0004_rejected",
			header:  ".target sm_60, texmode_independent\n",
			addr:    "0004",
			wantErr: "must be a multiple of 8",
		},
		{
			name:       "valid_old_code_for_sm60_0008_accepted",
			header:     "code for sm_60\n",
			addr:       "0008",
			wantTarget: "sm_60",
		},
		{
			name:    "valid_old_code_for_sm60_0004_rejected",
			header:  "code for sm_60\n",
			addr:    "0004",
			wantErr: "must be a multiple of 8",
		},

		// Valid modern target declarations (sm_70+) use 16-byte alignment.
		{
			name:    "valid_modern_space_sm70_0008_rejected",
			header:  ".target sm_70\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "valid_modern_space_sm70_0010_accepted",
			header:     ".target sm_70\n",
			addr:       "0010",
			wantTarget: "sm_70",
		},
		{
			name:    "valid_modern_tab_sm121a_0008_rejected",
			header:  ".target\tsm_121a\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "valid_modern_tab_sm121a_0010_accepted",
			header:     ".target\tsm_121a\n",
			addr:       "0010",
			wantTarget: "sm_121a",
		},
		{
			name:    "valid_modern_code_for_sm90_0008_rejected",
			header:  "code for sm_90\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "valid_modern_code_for_sm90_0010_accepted",
			header:     "code for sm_90\n",
			addr:       "0010",
			wantTarget: "sm_90",
		},

		// Missing or malformed target declarations default to 16-byte alignment with empty Target.
		{
			name:    "missing_target_0008_rejected",
			header:  "",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "missing_target_0010_accepted",
			header:     "",
			addr:       "0010",
			wantTarget: "",
		},
		{
			name:    "bare_target_directive_0008_rejected",
			header:  ".target\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "bare_target_directive_0010_accepted",
			header:     ".target\n",
			addr:       "0010",
			wantTarget: "",
		},
		{
			name:    "malformed_token_sm60_bang_0008_rejected",
			header:  ".target sm_60!\n",
			addr:    "0008",
			wantErr: "must be a multiple of 16",
		},
		{
			name:       "malformed_token_sm60_bang_0010_accepted",
			header:     ".target sm_60!\n",
			addr:       "0010",
			wantTarget: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := tc.header + ".text.k:\n /*" + tc.addr + "*/ NOP ;\n"
			d, err := cudasass.ParseString(input)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("ParseString(%q) succeeded with target=%q, want error containing %q", input, d.Target, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("ParseString(%q) error = %q, want substring %q", input, err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseString(%q) unexpected error: %v", input, err)
			}
			if d.Target != tc.wantTarget {
				t.Fatalf("ParseString(%q) d.Target = %q, want %q", input, d.Target, tc.wantTarget)
			}
		})
	}
}

// SPDX-License-Identifier: Apache-2.0

package cudagrammar_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
)

func ExampleMatcher_SpeculativeMasks() {
	g, err := cudagrammar.CompileString(`{"type":"object","properties":{"n":{"type":"integer"}}}`)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	vocab := cudagrammar.NewVocabulary([]string{
		0: `{`,
		1: `"n"`,
		2: `:`,
		3: `42`,
		4: `}`,
		5: `"bad"`,
	})

	m := g.Match()
	if err := m.CommitPrefix(vocab, []int32{0, 1}); err != nil {
		fmt.Println("error:", err)
		return
	}

	// Speculative draft chain: [2 (":"), 3 ("42"), 5 ("\"bad\"")].
	// Tokens 2 and 3 are accepted by the grammar; token 5 is rejected.
	masks, err := m.SpeculativeMasks(vocab, []int32{2, 3, 5})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("pos after SpeculativeMasks=%d masks=%d mask0_allows_colon=%v mask2_allows_rbrace=%v mask3_allowed=%d\n",
		m.Pos(), len(masks), masks[0].IsAllowed(2), masks[2].IsAllowed(4), masks[3].CountAllowed())

	if err := m.CommitPrefix(vocab, []int32{2, 3, 4}); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("done=%v pos=%d\n", m.Done(), m.Pos())
	// Output:
	// pos after SpeculativeMasks=4 masks=4 mask0_allows_colon=true mask2_allows_rbrace=true mask3_allowed=0
	// done=true pos=8
}

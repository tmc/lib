package gpucampaign_test

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/gpucampaign"
)

func Example() {
	r := gpucampaign.NewRunner(10 * time.Minute)
	phases := []gpucampaign.Phase{
		{Name: "timing", Argv: []string{"true"}, Cap: time.Minute, GPU: true},
		{Name: "profile", Argv: []string{"true"}, Cap: time.Hour, GPU: true},
	}
	for _, p := range phases {
		res := r.Run(context.Background(), p)
		fmt.Println(res.Name, res.Status)
	}
	// Output:
	// timing PASS
	// profile NOT_RUN
}

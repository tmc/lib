package nvtelemetry_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleWorkloadRefresher() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "# TYPE vllm:generation_tokens_total counter")
		fmt.Fprintln(w, `vllm:generation_tokens_total{model_name="example"} 40`)
	}))
	defer srv.Close()
	src := nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "run"}
	fetch := func(ctx context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
		if err != nil {
			return nvtelemetry.WorkloadHTTPResult{}, err
		}
		return nvtelemetry.FetchWorkload(ctx, srv.Client(), req, src, nvtelemetry.WorkloadLimits{})
	}
	clock := nvtelemetry.NewFakeClock(0, 1_700_000_000_000_000_000)
	r, err := nvtelemetry.NewWorkloadRefresher(fetch, clock, "synthetic_unix_utc", "example-epoch",
		nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: time.Second, MaxAge: 5 * time.Second})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer r.Close()
	if err := r.Step(context.Background()); err != nil {
		fmt.Println(err)
		return
	}
	res, fresh := r.Latest(clock.NowMonotonic())
	m := res.Export.Metrics[0]
	fmt.Println(fresh, m.RawValue, m.TimeSource, m.TimeNanos)

	clock.Advance(6 * time.Second)
	_, fresh = r.Latest(clock.NowMonotonic())
	fmt.Println(fresh)
	// Output:
	// true 40 collector_observation 1700000000000000000
	// false
}

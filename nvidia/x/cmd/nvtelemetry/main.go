package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		printUsage(stderr)
		return errors.New("command required")
	}

	switch args[0] {
	case "deliver":
		return runDeliver(args[1:], stdout, stderr)
	case "fetch-endpoint":
		return runFetchEndpoint(args[1:], stdout, stderr)
	case "serve":
		return runServe(args[1:], stdout, stderr)
	case "probe":
		return runProbe(args[1:], stdout, stderr)
	case "record":
		return runRecord(args[1:], stdout, stderr)
	case "refresh":
		return runRefresh(args[1:], stdout, stderr)
	case "replay":
		return runReplay(args[1:], stdout, stderr)
	case "summary":
		return runSummary(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		printUsage(stdout)
		return nil
	default:
		printUsage(stderr)
		return fmt.Errorf("unknown command: %s", args[0])
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintf(w, `nvtelemetry - continuous GPU telemetry recording and replay tool

Usage:
  nvtelemetry <command> [options]

Commands:
  deliver  Deliver an encoded OTLP metrics or logs request with bounded retries
  fetch-endpoint Fetch one explicitly configured provider or managed endpoint
  serve    Serve bounded cached Prometheus/OpenMetrics snapshots
  probe    Check an authenticated local health or readiness endpoint
  record   Record continuous GPU telemetry to a JSONL stream
  refresh  Refresh configured attribution and workload sources for bounded cycles
  replay   Replay or validate a recorded JSONL telemetry stream
  summary  Compute time-weighted and energy summaries for a phase window
  help     Show this help message
`)
}

func runRecord(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("record", flag.ContinueOnError)
	fs.SetOutput(stderr)

	outputPath := fs.String("output", "telemetry.jsonl", "output file path (use - for stdout)")
	interval := fs.Duration("interval", 100*time.Millisecond, "polling interval")
	duration := fs.Duration("duration", 5*time.Second, "recording duration (0 for indefinite until SIGINT)")
	phase := fs.String("phase", nvtelemetry.PhaseMeasurement, "initial execution phase")
	budget := fs.Int("budget", 100000, "maximum records budget")
	byteBudget := fs.Int("byte-budget", 64<<20, "maximum serialized stream bytes")
	bufferRecords := fs.Int("buffer-records", 5000, "maximum buffered records")
	dropOnFull := fs.Bool("drop-on-full", false, "drop samples on buffer exhaustion instead of failing")
	mock := fs.Bool("mock", false, "use mock sampler instead of live NVML")
	deviceUUIDs := fs.String("device-uuids", "", "comma-separated exact UUID allowlist (empty selects all devices)")
	escalate := fs.Bool("escalate", false, "enable budgeted incident escalation")
	triggerMetric := fs.String("trigger-metric", nvtelemetry.MetricGPUUtilization, "metric name for escalation trigger")
	triggerThreshold := fs.Float64("trigger-threshold", 90.0, "threshold value to trigger escalation")
	tenantID := fs.String("tenant", "default", "tenant identifier for incident tracking")
	incidentDir := fs.String("incident-dir", "", "directory to store incident capture bundles")

	if err := fs.Parse(args); err != nil {
		return err
	}
	selected, err := parseDeviceUUIDs(*deviceUUIDs)
	if err != nil {
		return err
	}
	bufferExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "buffer-records" {
			bufferExplicit = true
		}
	})
	if !bufferExplicit && *bufferRecords > *budget {
		*bufferRecords = *budget
	}
	if *interval <= 0 || *duration < 0 || *budget < 1 || *budget > 1<<20 || *byteBudget < 1 || *byteBudget > 256<<20 || *bufferRecords < 1 || *bufferRecords > *budget {
		return fmt.Errorf("invalid collector resource budget")
	}

	var sampler nvtelemetry.Sampler
	if *mock {
		sampler = newMockSampler()
		if len(selected) > 0 {
			return fmt.Errorf("device UUID selection is unavailable with mock sampler")
		}
	} else {
		if err := nvml.Init(); err != nil {
			return fmt.Errorf("nvml init failed (use -mock for offline runs): %w", err)
		}
		defer nvml.Shutdown()

		devices := make([]nvml.Device, 0, len(selected))
		for _, uuid := range selected {
			device, err := nvml.NewDeviceByUUID(uuid)
			if err != nil {
				return fmt.Errorf("selected device %s: %w", uuid, err)
			}
			actual, err := device.UUID()
			if err != nil {
				return fmt.Errorf("selected device identity: %w", err)
			}
			if actual != uuid {
				return fmt.Errorf("selected device identity mismatch")
			}
			devices = append(devices, device)
		}
		nvmlS, err := nvtelemetry.NewNVMLSampler(nvtelemetry.RealClock{}, devices...)
		if err != nil {
			return fmt.Errorf("create nvml sampler: %w", err)
		}
		sampler = nvmlS
	}

	var out io.Writer
	var closeFn func() error
	if *outputPath == "-" {
		out = stdout
		closeFn = func() error { return nil }
	} else {
		f, err := os.Create(*outputPath)
		if err != nil {
			return fmt.Errorf("create output file: %w", err)
		}
		out = f
		closeFn = f.Close
	}
	defer closeFn()

	cfg := nvtelemetry.CollectorConfig{
		IntervalNanos:      interval.Nanoseconds(),
		MaxDurationNanos:   duration.Nanoseconds(),
		MaxRecordBudget:    *budget,
		MaxByteBudget:      *byteBudget,
		MaxBufferedRecords: *bufferRecords,
		DropOnFull:         *dropOnFull,
	}

	w := nvtelemetry.NewWriter(out, cfg)
	collector := nvtelemetry.NewCollector("run-"+time.Now().Format("20060102-150405"), sampler, nvtelemetry.RealClock{}, cfg, w)
	collector.SetPhase(*phase)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stopEscalation := func() {}
	defer func() { stopEscalation() }()

	if *escalate {
		dir := *incidentDir
		if dir == "" {
			dir = "."
		}
		flightRec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
			Window:   10 * time.Second,
			Capacity: 65536,
		})
		if err != nil {
			return fmt.Errorf("create flight recorder: %w", err)
		}
		if err := flightRec.Start(); err != nil {
			return fmt.Errorf("start flight recorder: %w", err)
		}
		defer flightRec.Stop()

		budgets := cudaflightrecorder.DefaultBudgets()
		coord, err := cudaflightrecorder.NewEscalationCoordinator(flightRec, budgets,
			cudaflightrecorder.WithCaptureWriterFactory(cudaflightrecorder.NewDirectoryWriterFactory(dir)),
		)
		if err != nil {
			return fmt.Errorf("create escalation coordinator: %w", err)
		}
		if err := coord.Arm(); err != nil {
			return fmt.Errorf("arm escalation coordinator: %w", err)
		}
		defer coord.Stop()

		policy := nvtelemetry.TriggerPolicy{
			MetricName: *triggerMetric,
			Threshold:  *triggerThreshold,
			Comparison: ">",
			TenantID:   *tenantID,
			Reason:     "metric_threshold_exceeded",
		}

		escalationCtx, stopWorker := context.WithCancel(ctx)
		handleSample := func(s nvtelemetry.Sample) {
			for _, r := range s.Readings {
				rawReading, encodeErr := json.Marshal(r)
				record := cudaflightrecorder.Record{
					TimestampNs: r.TimestampUnix,
					KernelName:  "",
					DurationNs:  0,
					Labels: map[string]string{
						"metric":               r.Name,
						"source":               "nvtelemetry",
						"timestamp_provenance": "producer_reading",
						"device":               r.DeviceUUID,
						"unit":                 r.Unit,
					},
				}
				if encodeErr != nil {
					record.Labels["metric.encoding_error"] = encodeErr.Error()
				} else {
					record.Labels["metric.raw_json"] = string(rawReading)
				}
				flightRec.Push(record)
			}
			if r, matched, ok := nvtelemetry.EvaluateSample(s, []nvtelemetry.TriggerPolicy{policy}); ok {
				val := triggerMetricValue(r)
				receipt, triggerErr := coord.Trigger(escalationCtx, cudaflightrecorder.TriggerRequest{
					TenantID:    *tenantID,
					Reason:      matched.Reason,
					MetricName:  matched.MetricName,
					MetricValue: val,
					Threshold:   matched.Threshold,
				})
				if triggerErr != nil {
					fmt.Fprintf(stderr, "incident escalation refused or failed: %v\n", triggerErr)
				}
				if triggerErr == nil && receipt != nil {
					fmt.Fprintf(stderr, "Escalated incident %s: bundle captured (%d bytes, pre-trigger records=%d)\n",
						receipt.IncidentID, receipt.Perturbation.BundleBytes, receipt.Trigger.PreTrigger.RecordCount)
				}
			}
		}
		// Never block periodic telemetry on resource sampling or native capture.
		// One worker and one pending trigger bound escalation work independently.
		pending := make(chan nvtelemetry.Sample, 1)
		workerDone := make(chan struct{})
		var skipped atomic.Uint64
		go func() {
			defer close(workerDone)
			defer func() {
				if n := skipped.Load() + uint64(len(pending)); n > 0 {
					fmt.Fprintf(stderr, "incident escalation samples not evaluated: %d\n", n)
				}
			}()
			for {
				if escalationCtx.Err() != nil {
					return
				}
				select {
				case <-escalationCtx.Done():
					return
				case sample := <-pending:
					handleSample(sample)
				}
			}
		}()
		collector.SetSampleHook(func(sample nvtelemetry.Sample) {
			select {
			case pending <- sample:
			default:
				skipped.Add(1)
			}
		})
		stopEscalation = func() { stopWorker(); cancel(); <-workerDone }
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	go func() {
		select {
		case <-sigCh:
			cancel()
		case <-ctx.Done():
		}
	}()

	errCh := make(chan error, 1)
	go func() {
		_, err := collector.Run(ctx)
		errCh <- err
	}()

	var runErr error
	runDone := false
	if *duration > 0 {
		select {
		case <-time.After(*duration):
		case <-ctx.Done():
		case runErr = <-errCh:
			runDone = true
		}
	} else {
		select {
		case <-ctx.Done():
		case runErr = <-errCh:
			runDone = true
		}
	}

	manifest, stopErr := collector.Stop()
	stopEscalation()
	if !runDone {
		runErr = <-errCh
	}

	if stopErr != nil {
		return fmt.Errorf("collector stop: %w", stopErr)
	}
	if runErr != nil && !errors.Is(runErr, context.Canceled) {
		return fmt.Errorf("collector run: %w", runErr)
	}

	fmt.Fprintf(stderr, "Recording finished: %d samples collected (missed: %d, lost: %d)\n",
		manifest.TotalSamples, manifest.MissedTicks, manifest.LostRecords)
	return nil
}

var deviceUUIDPattern = regexp.MustCompile(`^(GPU-|MIG-)[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$|^MIG-GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/[0-9]+/[0-9]+$`)

// triggerMetricValue converts an admitted trigger scalar to the receipt's
// float64 representation, using the same integer rounding as TriggerPolicy.
func triggerMetricValue(r nvtelemetry.Reading) float64 {
	if n := r.Number; n != nil {
		switch {
		case n.Float != nil:
			return *n.Float
		case n.Int != nil:
			return float64(*n.Int)
		case n.Uint != nil:
			return float64(*n.Uint)
		}
	}
	if r.IsInt {
		return float64(r.IntValue)
	}
	return r.Value
}

func parseDeviceUUIDs(input string) ([]string, error) {
	if input == "" {
		return nil, nil
	}
	if len(input) > 16384 {
		return nil, fmt.Errorf("device UUID allowlist exceeds byte limit")
	}
	parts := strings.Split(input, ",")
	if len(parts) > 128 {
		return nil, fmt.Errorf("device UUID allowlist exceeds device limit")
	}
	seen := make(map[string]bool, len(parts))
	for _, uuid := range parts {
		if len(uuid) > 128 || !deviceUUIDPattern.MatchString(uuid) || seen[uuid] {
			return nil, fmt.Errorf("invalid or duplicate device UUID")
		}
		seen[uuid] = true
	}
	return parts, nil
}

func runReplay(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "input JSONL telemetry file (required)")
	strict := fs.Bool("strict", false, "fail if stream is missing completion manifest")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("-input is required")
	}

	file, err := os.Open(*inputPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer file.Close()

	if *strict {
		manifest, phases, samples, err := nvtelemetry.ReadStreamStrict(file, 0)
		if err != nil {
			return fmt.Errorf("strict stream read failed: %w", err)
		}
		fmt.Fprintf(stdout, "Stream verified (strict): run_id=%s devices=%d phases=%d samples=%d\n",
			manifest.RunID, len(manifest.Devices), len(phases), len(samples))
		return nil
	}

	opts := nvtelemetry.WindowOptions{
		MaxAllowedGapNanos: time.Second.Nanoseconds(),
	}
	manifest, windowSummaries, energySummaries, err := nvtelemetry.Replay(file, 0, opts)
	if err != nil {
		return fmt.Errorf("replay stream: %w", err)
	}

	fmt.Fprintf(stdout, "Replayed run %s (collector: %s, devices: %d, complete: %v)\n",
		manifest.RunID, manifest.CollectorVersion, len(manifest.Devices), manifest.IsComplete)
	for _, phaseName := range sortedPhaseNames(windowSummaries) {
		wSummaries := windowSummaries[phaseName]
		fmt.Fprintf(stdout, "  Phase: %s (%d metrics, %d energy summaries)\n",
			phaseName, len(wSummaries), len(energySummaries[phaseName]))
	}

	return nil
}

func runSummary(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("summary", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "input JSONL telemetry file (required)")
	window := fs.String("window", nvtelemetry.PhaseMeasurement, "window/phase name to summarize")
	format := fs.String("format", "text", "output format: text or json")
	maxGap := fs.Duration("max-gap", time.Second, "maximum allowed gap before marking coverage broken")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("-input is required")
	}

	file, err := os.Open(*inputPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer file.Close()

	manifest, phases, samples, err := nvtelemetry.ReadStream(file, 0)
	if err != nil {
		return fmt.Errorf("read stream: %w", err)
	}

	var startMono, endMono int64
	found := false
	for _, p := range phases {
		if p.Name == *window {
			startMono = p.StartMonotonicNanos
			endMono = p.EndMonotonicNanos
			found = true
			break
		}
	}
	if !found {
		startMono = manifest.StartTimeMonotonic
		endMono = manifest.EndTimeMonotonic
		if endMono <= startMono && len(samples) > 0 {
			startMono = samples[0].QueryStartNanos
			endMono = samples[len(samples)-1].QueryEndNanos
		}
	}

	opts := nvtelemetry.WindowOptions{
		MaxAllowedGapNanos: maxGap.Nanoseconds(),
	}
	summaries, energySummaries := nvtelemetry.AggregateWindow(*window, startMono, endMono, samples, opts)

	if *format == "json" {
		out := struct {
			RunID    string                      `json:"run_id"`
			Window   string                      `json:"window"`
			Complete bool                        `json:"complete"`
			Metrics  []nvtelemetry.WindowSummary `json:"metrics"`
			Energy   []nvtelemetry.EnergySummary `json:"energy"`
		}{
			RunID:    manifest.RunID,
			Window:   *window,
			Complete: manifest.IsComplete,
			Metrics:  summaries,
			Energy:   energySummaries,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}

	// Text format
	fmt.Fprintf(stdout, "Run ID:     %s\n", manifest.RunID)
	fmt.Fprintf(stdout, "Window:     %s\n", *window)
	fmt.Fprintf(stdout, "Completed:  %v\n\n", manifest.IsComplete)

	w := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DEVICE\tMETRIC\tMEAN (TIME-WT)\tARITH-MEAN\tMIN\tMAX\tCOVERAGE\tUNIT")
	for _, s := range summaries {
		fmt.Fprintf(w, "%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.1f%%\t%s\n",
			shortUUID(s.DeviceUUID), s.MetricName, s.TimeWeightedMean, s.ArithmeticMean,
			s.Min, s.Max, s.CoverageRatio*100, s.Unit)
	}
	w.Flush()

	if len(energySummaries) > 0 {
		fmt.Fprintf(stdout, "\nEnergy Summary:\n")
		wEnergy := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(wEnergy, "DEVICE\tMETHOD\tJOULES\tCOVERAGE\tVALID\tREASON")
		for _, e := range energySummaries {
			fmt.Fprintf(wEnergy, "%s\t%s\t%.3f\t%.1f%%\t%v\t%s\n",
				shortUUID(e.DeviceUUID), e.Method, e.EnergyConsumedJoules, e.CoverageRatio*100, e.Valid, e.Reason)
		}
		wEnergy.Flush()
	}

	return nil
}

func shortUUID(uuid string) string {
	if len(uuid) > 12 {
		return uuid[:12]
	}
	return uuid
}

func newMockSampler() *nvtelemetry.FakeSampler {
	meta := nvtelemetry.DeviceMetadata{
		UUID:          "GPU-mock-0001",
		Index:         0,
		Model:         "NVIDIA L4",
		Architecture:  "0x89",
		DriverVersion: "550.54.14",
		MIGScope:      nvtelemetry.MIGScopeDisabled,
		Status:        nvtelemetry.DeviceStatusOk,
	}
	readings := [][]nvtelemetry.Reading{
		{
			{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, Value: 75.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, Value: 68000.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, IntValue: 1200000, IsInt: true, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricTemperatureC, Unit: nvtelemetry.UnitCelsius, Value: 52.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
		},
		{
			{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, Value: 82.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, Value: 71000.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, IntValue: 1207000, IsInt: true, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
			{Name: nvtelemetry.MetricTemperatureC, Unit: nvtelemetry.UnitCelsius, Value: 53.0, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-mock-0001"},
		},
	}
	s := nvtelemetry.NewFakeSampler(meta)
	s.Enqueue(readings...)
	return s
}

func sortedPhaseNames(summaries map[string][]nvtelemetry.WindowSummary) []string {
	phaseNames := make([]string, 0, len(summaries))
	for phaseName := range summaries {
		phaseNames = append(phaseNames, phaseName)
	}
	sort.Strings(phaseNames)
	return phaseNames
}

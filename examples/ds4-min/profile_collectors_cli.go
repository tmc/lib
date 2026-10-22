package main

import (
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertprofile"
)

type collectorOptions struct {
	dataset, imatrix       string
	expertProfile, hotlist string
	maxPrompts, maxTokens  int
	minSamples             uint
	maxBytes               uint64
}

func validateCalibrationFlags(set map[string]bool, backend string) error {
	allowed := map[string]bool{"model": true, "backend": true, "ctx": true, "prefill-chunk": true, "imatrix-dataset": true, "imatrix-out": true, "imatrix-max-prompts": true, "imatrix-max-tokens": true, "imatrix-min-expert-samples": true, "collector-max-bytes": true}
	if backend == "cuda" {
		allowed["ballast-gib"] = true
		allowed["warm-weights"] = true
	}
	var rejected []string
	for name := range set {
		if !allowed[name] {
			rejected = append(rejected, name)
		}
	}
	if len(rejected) != 0 {
		sort.Strings(rejected)
		return fmt.Errorf("imatrix: ordinary inference flag -%s is not used by calibration", rejected[0])
	}
	return nil
}

func validateCollectorOptions(cfg config) error {
	o := cfg.collectors
	calibration := o.dataset != "" || o.imatrix != ""
	locality := o.expertProfile != "" || o.hotlist != ""
	if !calibration && !locality {
		return nil
	}
	if cfg.inspect || cfg.dumpTokens || cfg.interactive || cfg.evalSuite != "" || cfg.perplexityFile != "" || cfg.decodeConsistency != 0 || cfg.warmPrefill || cfg.kvLoad != "" {
		return fmt.Errorf("profile collectors: incompatible inference mode")
	}
	if cfg.backend != "cuda" && cfg.backend != "metal" {
		return fmt.Errorf("profile collectors: select CUDA or Metal explicitly")
	}
	if calibration && (locality || o.dataset == "" || o.imatrix == "") {
		return fmt.Errorf("imatrix: require separate dataset and output mode")
	}
	if calibration && (cfg.expectPrefillLogits != "" || cfg.expectFinalLogits != "" || cfg.dumpLogits != "" || cfg.profilerRange || cfg.profilerStop) {
		return fmt.Errorf("imatrix: calibration cannot use ordinary digest or profiler phase flags")
	}
	if o.maxBytes == 0 || o.maxBytes > 1<<30 || o.maxPrompts < 0 || o.maxTokens < 0 {
		return fmt.Errorf("profile collectors: invalid limits")
	}
	return nil
}

func attachExpertProfile(cfg config, b sessionBackend, promptTokens int) (*profileCollectors, error) {
	if cfg.collectors.expertProfile == "" && cfg.collectors.hotlist == "" {
		return nil, nil
	}
	if cfg.backend != "cuda" && cfg.backend != "metal" || cfg.kvLoad != "" || cfg.warmPrefill || cfg.tokens < 2 {
		return nil, fmt.Errorf("expert profile: require fresh CUDA or Metal decode with at least two tokens")
	}
	p, err := newProfileCollectors(false, true, cfg.collectors.maxBytes)
	if err != nil {
		return nil, err
	}
	p.expertStart = promptTokens
	owner, ok := b.(interface {
		setProfileCollectors(*profileCollectors) error
	})
	if !ok {
		return nil, fmt.Errorf("expert profile: backend has no producer")
	}
	if err := owner.setProfileCollectors(p); err != nil {
		return nil, err
	}
	return p, nil
}

func writeExpertProfile(cfg config, p *profileCollectors) error {
	if p == nil {
		return nil
	}
	if err := completeProfileRows(p.expertRows, p.expertRows[0]); err != nil {
		return err
	}
	hash := make([]bool, 43)
	for i := 0; i < 3; i++ {
		hash[i] = true
	}
	for _, out := range []struct {
		path  string
		write func(io.Writer) error
	}{
		{cfg.collectors.expertProfile, func(w io.Writer) error {
			return p.experts.WriteReport(w, dsv4expertprofile.Metadata{Source: "ds4-min " + cfg.backend + " expert locality profile", Model: "DeepSeek-V4-Flash", ExpertUsed: 6, HashRouter: hash})
		}},
		{cfg.collectors.hotlist, func(w io.Writer) error { return p.experts.WriteHotlist(w, "DeepSeek-V4-Flash") }},
	} {
		if out.path == "" {
			continue
		}
		if err := writeCollectorOutput(out.path, out.write); err != nil {
			return err
		}
	}
	return nil
}

func finishProfileOutput(err error, closer io.Closer, cfg config, p *profileCollectors) error {
	err = finishBackendCleanup(err, closer)
	if err != nil {
		return err
	}
	return writeExpertProfile(cfg, p)
}

func addCollectorFlags(fs *flag.FlagSet, o *collectorOptions) {
	fs.StringVar(&o.expertProfile, "expert-profile", "", "write decode expert routing and cache locality JSON")
	fs.StringVar(&o.hotlist, "expert-hotlist", "", "write decode expert hotlist text")
	fs.StringVar(&o.dataset, "imatrix-dataset", "", "rendered calibration dataset, separated by DS4_IMATRIX_PROMPT marker lines")
	fs.StringVar(&o.imatrix, "imatrix-out", "", "write collected expert activation means in legacy dat format")
	fs.IntVar(&o.maxPrompts, "imatrix-max-prompts", 0, "maximum calibration prompts; zero means all")
	fs.IntVar(&o.maxTokens, "imatrix-max-tokens", 0, "maximum calibration tokens; zero means all")
	fs.UintVar(&o.minSamples, "imatrix-min-expert-samples", 0, "required samples for every layer expert before writing")
	fs.Uint64Var(&o.maxBytes, "collector-max-bytes", 1<<30, "maximum collector working storage bytes")
}

func calibrationPrompts(text string) []string {
	const marker = "===== DS4_IMATRIX_PROMPT"
	var prompts []string
	start := -1
	for offset := 0; offset < len(text); {
		n := strings.Index(text[offset:], marker)
		if n < 0 {
			break
		}
		n += offset
		if n != 0 && text[n-1] != '\n' {
			offset = n + len(marker)
			continue
		}
		if start >= 0 {
			prompts = append(prompts, strings.Trim(text[start:n], " \t\r\n\v\f"))
		}
		end := strings.IndexByte(text[n:], '\n')
		if end < 0 {
			start = len(text)
			break
		}
		start = n + end + 1
		offset = start
	}
	if start < 0 {
		return []string{strings.Trim(text, " \t\r\n\v\f")}
	}
	return append(prompts, strings.Trim(text[start:], " \t\r\n\v\f"))
}

func runImatrix(cfg config, tok *tokenizer, log io.Writer) (err error) {
	o := cfg.collectors
	if o.expertProfile != "" || o.hotlist != "" {
		return fmt.Errorf("imatrix: calibration and decode locality modes are separate")
	}
	if o.dataset == "" || o.imatrix == "" || o.maxPrompts < 0 || o.maxTokens < 0 || cfg.backend != "cuda" && cfg.backend != "metal" {
		return fmt.Errorf("imatrix: require dataset, output and CUDA or Metal backend")
	}
	dataset, err := os.Open(o.dataset)
	if err != nil {
		return fmt.Errorf("imatrix dataset: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(dataset, (64<<20)+1))
	if err := errors.Join(readErr, dataset.Close()); err != nil {
		return fmt.Errorf("imatrix dataset: %w", err)
	}
	if len(data) > 64<<20 {
		return fmt.Errorf("imatrix dataset exceeds byte limit")
	}
	p, err := newProfileCollectors(true, false, o.maxBytes)
	if err != nil {
		return err
	}
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return err
	}
	cleaned := false
	defer func() {
		if !cleaned {
			err = finishBackendCleanup(err, closer)
		}
	}()
	owner, ok := b.(interface {
		setProfileCollectors(*profileCollectors) error
	})
	if !ok {
		return fmt.Errorf("imatrix: backend has no producer")
	}
	if err := owner.setProfileCollectors(p); err != nil {
		return err
	}
	s, err := newSession(b, cfg.context, tok.stops)
	if err != nil {
		return err
	}
	if err := configurePrefillChunk(b, cfg.prefillChunk); err != nil {
		return err
	}
	nprompts, ntokens := 0, 0
	for _, text := range calibrationPrompts(string(data)) {
		if text == "" {
			continue
		}
		if o.maxPrompts > 0 && nprompts >= o.maxPrompts || o.maxTokens > 0 && ntokens >= o.maxTokens {
			break
		}
		ids, err := tok.encodeRendered(text)
		if err != nil {
			return err
		}
		limit := cfg.context - 1
		if o.maxTokens > 0 && limit > o.maxTokens-ntokens {
			limit = o.maxTokens - ntokens
		}
		if len(ids) > limit {
			ids = ids[:limit]
		}
		if len(ids) == 0 {
			continue
		}
		if err := s.Reset(); err != nil {
			return err
		}
		if err := s.Sync(ids); err != nil {
			return err
		}
		nprompts++
		ntokens += len(ids)
	}
	err = finishBackendCleanup(nil, closer)
	cleaned = true
	if err != nil {
		return err
	}
	if err := writeImatrixFile(o, p, ntokens); err != nil {
		return err
	}
	_, err = fmt.Fprintf(log, "imatrix dataset_sha256=%x prompts=%d tokens=%d layer_chunks=%d min_samples=%d\n", sha256.Sum256(data), nprompts, ntokens, p.imatrix.Chunks(), p.imatrix.MinSamples())
	return err
}

func writeImatrixFile(o collectorOptions, p *profileCollectors, ntokens int) error {
	if p == nil || p.imatrix == nil {
		return fmt.Errorf("imatrix: no producer")
	}
	if ntokens <= 0 || uint64(p.imatrix.MinSamples()) < uint64(o.minSamples) {
		return fmt.Errorf("imatrix: incomplete samples: tokens=%d min=%d", ntokens, p.imatrix.MinSamples())
	}
	if err := completeProfileRows(p.imatrixRows, uint64(ntokens)); err != nil {
		return err
	}
	return writeCollectorOutput(o.imatrix, func(w io.Writer) error { return p.imatrix.Write(w, o.dataset) })
}

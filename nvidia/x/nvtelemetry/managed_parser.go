package nvtelemetry

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

const (
	PrometheusText004 = "prometheus-0.0.4"
	OpenMetrics100    = "openmetrics-1.0.0"
)

// ExpositionOptions bounds an offline exposition read. Zero limits use defaults.
type ExpositionOptions struct {
	Format                                             string
	MaxBytes                                           int64
	MaxSamples, MaxLabels, MaxLabelBytes, MaxLineBytes int
}

// ExpositionSample retains a producer's numeric text without float conversion.
// TimestampUnix is an exposition timestamp, not a sensor timestamp.
type ExpositionSample struct {
	Name                 string
	Labels               map[string]string
	RawValue, Type, Unit string
	TimestampUnix        int64
	HasTimestamp         bool
	RawTimestamp         string
	TimestampUnit        string
}

// ReadExposition reads Prometheus text or OpenMetrics. It preserves histogram
// components without interpreting their buckets. Summaries and exemplars are
// refused. Cancellation is checked between reads; callers must close a blocked
// reader or use a context-aware transport to interrupt its Read call.
func ReadExposition(ctx context.Context, r io.Reader, o ExpositionOptions) ([]ExpositionSample, error) {
	if ctx == nil || r == nil {
		return nil, fmt.Errorf("exposition: nil context or reader")
	}
	if o.Format == "" {
		o.Format = PrometheusText004
	}
	if o.Format != PrometheusText004 && o.Format != OpenMetrics100 {
		return nil, fmt.Errorf("exposition: unsupported format")
	}
	if o.MaxBytes == 0 {
		o.MaxBytes = 1 << 20
	}
	if o.MaxSamples == 0 {
		o.MaxSamples = 4096
	}
	if o.MaxLabels == 0 {
		o.MaxLabels = 16
	}
	if o.MaxLabelBytes == 0 {
		o.MaxLabelBytes = 256
	}
	if o.MaxLineBytes == 0 {
		o.MaxLineBytes = 8192
	}
	if o.MaxBytes < 1 || o.MaxSamples < 1 || o.MaxLabels < 1 || o.MaxLabelBytes < 1 || o.MaxLineBytes < 1 {
		return nil, fmt.Errorf("exposition: invalid limit")
	}
	if o.MaxBytes > 64<<20 || o.MaxSamples > 65536 || o.MaxLabels > 64 || o.MaxLabelBytes > 4096 || o.MaxLineBytes > 1<<20 {
		return nil, fmt.Errorf("exposition: limit exceeds hard cap")
	}
	lr := &io.LimitedReader{R: r, N: o.MaxBytes + 1}
	sc := bufio.NewScanner(lr)
	sc.Buffer(make([]byte, min(o.MaxLineBytes, 4096)), o.MaxLineBytes)
	types, units := map[string]string{}, map[string]string{}
	seen := map[string]bool{}
	sampled := map[string]bool{}
	var out []ExpositionSample
	eof := false
	for sc.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if eof {
			return nil, fmt.Errorf("exposition: data after eof")
		}
		if line == "# EOF" {
			if o.Format != OpenMetrics100 {
				return nil, fmt.Errorf("exposition: unexpected eof marker")
			}
			eof = true
			continue
		}
		if strings.HasPrefix(line, "#") {
			f := strings.Fields(line)
			if len(f) > 1 && (f[1] == "TYPE" || f[1] == "UNIT") {
				if len(f) != 4 || !metricName(f[2]) {
					return nil, fmt.Errorf("exposition: invalid metadata")
				}
				if len(f[2])+len(f[3]) > o.MaxLabelBytes || len(types)+len(units) >= o.MaxSamples {
					return nil, fmt.Errorf("exposition: metadata limit")
				}
				m := types
				if sampled[f[2]] || sampled[f[2]+"_bucket"] || sampled[f[2]+"_sum"] || sampled[f[2]+"_count"] || sampled[f[2]+"_total"] || sampled[f[2]+"_created"] {
					return nil, fmt.Errorf("exposition: metadata after samples")
				}
				if f[1] == "TYPE" && f[3] != "gauge" && f[3] != "counter" && f[3] != "histogram" && f[3] != "summary" && f[3] != "untyped" {
					return nil, fmt.Errorf("exposition: unsupported metric type")
				}
				if f[1] == "UNIT" {
					if o.Format != OpenMetrics100 {
						return nil, fmt.Errorf("exposition: unit metadata requires openmetrics")
					}
					m = units
				}
				if _, ok := m[f[2]]; ok {
					return nil, fmt.Errorf("exposition: duplicate metadata")
				}
				m[f[2]] = f[3]
			}
			continue
		}
		if len(out) >= o.MaxSamples {
			return nil, fmt.Errorf("exposition: sample limit")
		}
		s, err := parseExpositionLine(line, o)
		if err != nil {
			return nil, err
		}
		base := s.Name
		if o.Format == OpenMetrics100 && strings.HasSuffix(base, "_total") && types[strings.TrimSuffix(base, "_total")] == "counter" {
			base = strings.TrimSuffix(base, "_total")
		}
		for _, suffix := range []string{"_bucket", "_count", "_sum"} {
			if strings.HasSuffix(base, suffix) && types[strings.TrimSuffix(base, suffix)] == "histogram" {
				base = strings.TrimSuffix(base, suffix)
				break
			}
		}
		s.Type = types[base]
		s.Unit = units[base]
		if s.Type == "summary" {
			return nil, fmt.Errorf("exposition: summaries unsupported")
		}
		sampled[s.Name] = true
		key := s.Name + "/"
		labels := make([]string, 0, len(s.Labels))
		for k, v := range s.Labels {
			labels = append(labels, k+"="+strconv.Quote(v))
		}
		slices.Sort(labels)
		key += strings.Join(labels, ",")
		if seen[key] {
			return nil, fmt.Errorf("exposition: duplicate series")
		}
		seen[key] = true
		out = append(out, s)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("exposition: %w", err)
	}
	if lr.N == 0 {
		return nil, fmt.Errorf("exposition: byte limit")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if o.Format == OpenMetrics100 && !eof {
		return nil, fmt.Errorf("exposition: missing eof")
	}
	return out, nil
}

func metricName(s string) bool {
	if s == "" {
		return false
	}
	for i, c := range []byte(s) {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == ':' || (i > 0 && c >= '0' && c <= '9')) {
			return false
		}
	}
	return true
}

func labelName(s string) bool {
	return metricName(s) && !strings.Contains(s, ":")
}

func parseExpositionLine(line string, o ExpositionOptions) (ExpositionSample, error) {
	s := ExpositionSample{Labels: map[string]string{}}
	i := strings.IndexAny(line, "{ \t")
	if i < 0 {
		return s, fmt.Errorf("exposition: missing value")
	}
	s.Name = line[:i]
	if !metricName(s.Name) {
		return s, fmt.Errorf("exposition: invalid name")
	}
	rest := line[i:]
	if rest[0] == '{' {
		rest = rest[1:]
		for {
			rest = strings.TrimLeft(rest, " \t")
			if strings.HasPrefix(rest, "}") {
				rest = rest[1:]
				break
			}
			eq := strings.IndexByte(rest, '=')
			if eq < 1 {
				return s, fmt.Errorf("exposition: invalid label")
			}
			k := strings.TrimSpace(rest[:eq])
			if !labelName(k) {
				return s, fmt.Errorf("exposition: invalid label name")
			}
			rest = strings.TrimLeft(rest[eq+1:], " \t")
			if len(rest) == 0 || rest[0] != '"' {
				return s, fmt.Errorf("exposition: unquoted label")
			}
			end := 1
			for end < len(rest) {
				if rest[end] == '\\' {
					end += 2
					continue
				}
				if rest[end] == '"' {
					break
				}
				end++
			}
			if end >= len(rest) {
				return s, fmt.Errorf("exposition: unterminated label")
			}
			raw := rest[:end+1]
			for j := 1; j < len(raw)-1; j++ {
				if raw[j] == '\\' {
					j++
					if j >= len(raw)-1 || !strings.ContainsRune("\\n\"", rune(raw[j])) {
						return s, fmt.Errorf("exposition: invalid escape")
					}
				}
			}
			v, err := strconv.Unquote(raw)
			if err != nil {
				return s, fmt.Errorf("exposition: invalid label: %w", err)
			}
			if len(k)+len(v) > o.MaxLabelBytes || len(s.Labels) >= o.MaxLabels {
				return s, fmt.Errorf("exposition: label limit")
			}
			if _, ok := s.Labels[k]; ok {
				return s, fmt.Errorf("exposition: duplicate label")
			}
			s.Labels[k] = v
			rest = strings.TrimLeft(rest[end+1:], " \t")
			if strings.HasPrefix(rest, ",") {
				rest = rest[1:]
				continue
			}
			if !strings.HasPrefix(rest, "}") {
				return s, fmt.Errorf("exposition: missing label separator")
			}
		}
	}
	f := strings.Fields(rest)
	if len(f) < 1 || len(f) > 2 {
		return s, fmt.Errorf("exposition: unsupported sample suffix")
	}
	if _, err := strconv.ParseFloat(f[0], 64); err != nil {
		return s, fmt.Errorf("exposition: invalid value")
	}
	s.RawValue = f[0]
	if len(f) == 2 {
		s.RawTimestamp = f[1]
		s.TimestampUnit = "milliseconds"
		scale := int64(1000000)
		if o.Format == OpenMetrics100 {
			s.TimestampUnit = "seconds"
			scale = 1000000000
		} else {
			if _, err := strconv.ParseInt(f[1], 10, 64); err != nil {
				return s, fmt.Errorf("exposition: invalid millisecond timestamp")
			}
		}
		rat, ok := boundedDecimal(f[1])
		if !ok {
			return s, fmt.Errorf("exposition: invalid timestamp")
		}
		rat.Mul(rat, new(big.Rat).SetInt64(scale))
		if !rat.IsInt() || !rat.Num().IsInt64() {
			return s, fmt.Errorf("exposition: timestamp precision or overflow")
		}
		s.TimestampUnix = rat.Num().Int64()
		s.HasTimestamp = true
	}
	return s, nil
}

func boundedDecimal(s string) (*big.Rat, bool) {
	if len(s) > 256 {
		return nil, false
	}
	if _, err := strconv.ParseFloat(s, 64); err != nil {
		return nil, false
	}
	if strings.ContainsAny(s, "xXpP/") {
		return nil, false
	}
	if i := strings.IndexAny(s, "eE"); i >= 0 {
		n, err := strconv.ParseInt(s[i+1:], 10, 16)
		if err != nil || n < -400 || n > 400 {
			return nil, false
		}
	}
	return new(big.Rat).SetString(s)
}

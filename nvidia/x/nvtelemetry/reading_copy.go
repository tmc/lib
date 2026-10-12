package nvtelemetry

func cloneReadings(readings []Reading) []Reading {
	if readings == nil {
		return nil
	}
	out := append([]Reading(nil), readings...)
	for i := range out {
		if out[i].Number != nil {
			out[i].Number = cloneNumber(out[i].Number)
		}
		if out[i].Observation != nil {
			o := *out[i].Observation
			if o.OTLPScopeName != nil {
				v := *o.OTLPScopeName
				o.OTLPScopeName = &v
			}
			if o.OTLPClock != nil {
				v := *o.OTLPClock
				o.OTLPClock = &v
			}
			if o.OTLPSourceTime != nil {
				v := *o.OTLPSourceTime
				o.OTLPSourceTime = &v
			}
			if o.Interval != nil {
				interval := *o.Interval
				o.Interval = &interval
			}
			if o.Denominator != nil {
				o.Denominator = cloneNumber(o.Denominator)
			}
			if o.Histogram != nil {
				h := *o.Histogram
				if h.Count != nil {
					v := *h.Count
					h.Count = &v
				}
				if h.Sum != nil {
					v := *h.Sum
					h.Sum = &v
				}
				if h.Min != nil {
					v := *h.Min
					h.Min = &v
				}
				if h.Max != nil {
					v := *h.Max
					h.Max = &v
				}
				h.Bounds = append([]string(nil), h.Bounds...)
				h.Counts = append([]uint64(nil), h.Counts...)
				o.Histogram = &h
			}
			if o.Observer.SamplingCoverage != nil {
				coverage := *o.Observer.SamplingCoverage
				if coverage.Numerator != nil {
					coverage.Numerator = cloneNumber(coverage.Numerator)
				}
				if coverage.Denominator != nil {
					coverage.Denominator = cloneNumber(coverage.Denominator)
				}
				o.Observer.SamplingCoverage = &coverage
			}
			out[i].Observation = &o
		}
		if out[i].Source != nil {
			s := *out[i].Source
			if s.Native != nil {
				native := *s.Native
				s.Native = &native
			}
			if s.Labels != nil {
				s.Labels = make(map[string]string, len(out[i].Source.Labels))
				for k, v := range out[i].Source.Labels {
					s.Labels[k] = v
				}
			}
			out[i].Source = &s
		}
	}
	return out
}

func cloneNumber(number *Number) *Number {
	if number == nil {
		return nil
	}
	copyNumber := *number
	if number.Float != nil {
		value := *number.Float
		copyNumber.Float = &value
	}
	if number.Int != nil {
		value := *number.Int
		copyNumber.Int = &value
	}
	if number.Uint != nil {
		value := *number.Uint
		copyNumber.Uint = &value
	}
	return &copyNumber
}

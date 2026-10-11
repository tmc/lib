package nvtelemetry

func cloneReadings(readings []Reading) []Reading {
	if readings == nil {
		return nil
	}
	out := append([]Reading(nil), readings...)
	for i := range out {
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

package egraph

import "github.com/tmc/lib/egraphcore"

// cloneMatch copies the mutable maps in a match. lookupView is a snapshot used
// for function lookups and is shared as immutable state.
func cloneMatch(match Match) Match {
	if match.Subst != nil {
		subst := make(Subst, len(match.Subst))
		for name, id := range match.Subst {
			subst[name] = id
		}
		match.Subst = subst
	}
	if match.BaseSubst != nil {
		base := make(map[string]FunctionValue, len(match.BaseSubst))
		for name, value := range match.BaseSubst {
			base[name] = value
		}
		match.BaseSubst = base
	}
	if match.computedValues != nil {
		values := make(map[*Pattern]FunctionValue, len(match.computedValues))
		for pattern, value := range match.computedValues {
			values[pattern] = value
		}
		match.computedValues = values
	}
	return match
}

func cloneMatches(matches []Match) []Match {
	if matches == nil {
		return nil
	}
	cloned := make([]Match, len(matches))
	for i, match := range matches {
		cloned[i] = cloneMatch(match)
	}
	return cloned
}

func cloneTermMatchTracker(tracker *termMatchTracker) *termMatchTracker {
	if tracker == nil {
		return nil
	}
	cloned := *tracker
	if tracker.matches != nil {
		cloned.matches = make([]map[Id]map[string]Match, len(tracker.matches))
		for pattern, byClass := range tracker.matches {
			if byClass == nil {
				continue
			}
			clonedByClass := make(map[Id]map[string]Match, len(byClass))
			for classID, byKey := range byClass {
				if byKey == nil {
					clonedByClass[classID] = nil
					continue
				}
				clonedByKey := make(map[string]Match, len(byKey))
				for key, match := range byKey {
					clonedByKey[key] = cloneMatch(match)
				}
				clonedByClass[classID] = clonedByKey
			}
			cloned.matches[pattern] = clonedByClass
		}
	}
	return &cloned
}

func cloneRelationMatchTracker(tracker *relationMatchTracker) *relationMatchTracker {
	if tracker == nil {
		return nil
	}
	cloned := *tracker
	if tracker.previous != nil {
		cloned.previous = make([][]Match, len(tracker.previous))
		for i, matches := range tracker.previous {
			cloned.previous[i] = cloneMatches(matches)
		}
	}
	if tracker.keys != nil {
		cloned.keys = make([]map[string]struct{}, len(tracker.keys))
		for i, keys := range tracker.keys {
			if keys == nil {
				continue
			}
			clonedKeys := make(map[string]struct{}, len(keys))
			for key := range keys {
				clonedKeys[key] = struct{}{}
			}
			cloned.keys[i] = clonedKeys
		}
	}
	if tracker.cursors != nil {
		cloned.cursors = make([]int, len(tracker.cursors))
		copy(cloned.cursors, tracker.cursors)
	}
	if tracker.coreCursors != nil {
		cloned.coreCursors = append([]egraphcore.RelationCursor(nil), tracker.coreCursors...)
	}
	return &cloned
}

func cloneTermMatchTrackers(trackers map[*Rewrite]*termMatchTracker) map[*Rewrite]*termMatchTracker {
	cloned := make(map[*Rewrite]*termMatchTracker, len(trackers))
	for rule, tracker := range trackers {
		cloned[rule] = cloneTermMatchTracker(tracker)
	}
	return cloned
}

func cloneRelationMatchTrackers(trackers map[*Rewrite]*relationMatchTracker) map[*Rewrite]*relationMatchTracker {
	cloned := make(map[*Rewrite]*relationMatchTracker, len(trackers))
	for rule, tracker := range trackers {
		cloned[rule] = cloneRelationMatchTracker(tracker)
	}
	return cloned
}

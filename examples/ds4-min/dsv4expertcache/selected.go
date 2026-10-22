package dsv4expertcache

import "errors"

// Stager owns the staging storage and transfer lifecycle for selected experts.
// Its Stage implementation commits the lease only after uploads complete.
type Stager interface {
	Stage(*Source, *Lease, bool) error
}

// Select loads a layer's routed experts and returns a committed consumer lease.
// IDs may repeat; the returned slots preserve their original order. The caller
// must Release the lease using completion of every operation reading the slots,
// not merely completion of the upload. Failed upload cleanup can return a
// retained lease alongside the error. A poisoned cache cannot be reused; its
// backend owner must drain uncertain writes before freeing its storage.
// Graph capture is rejected before reserving or reading anything.
func (s *Source) Select(cache *Cache, layer int, ids []int, stager Stager, graphCapture bool) (*Lease, error) {
	if graphCapture {
		return nil, errors.New("expert streaming graph capture unqualified")
	}
	if s == nil || cache == nil || cache.identity != s.identity || nilCompletion(stager) {
		return nil, errors.New("source identity mismatch")
	}
	tr, ok := s.layers[layer]
	if !ok {
		return nil, errors.New("missing expert layer")
	}
	keys := make([]Key, len(ids))
	for i, id := range ids {
		if id < 0 || id >= tr.Gate.Experts {
			return nil, errors.New("invalid selected expert")
		}
		keys[i] = Key{Identity: s.identity, Layer: layer, Expert: id}
	}
	lease, err := cache.Acquire(keys)
	if err != nil {
		return nil, err
	}
	if err = stager.Stage(s, lease, false); err != nil {
		if !lease.done {
			err = errors.Join(err, lease.Abort())
		}
		if cache.closed || !lease.done {
			return lease, err
		}
		return nil, err
	}
	if !lease.committed {
		err = errors.Join(errors.New("stager returned an uncommitted lease"), lease.Abort())
		if cache.closed || !lease.done {
			return lease, err
		}
		return nil, err
	}
	return lease, nil
}

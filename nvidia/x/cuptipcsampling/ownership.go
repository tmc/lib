package cuptipcsampling

import (
	"errors"
	"sync"
)

// This protects sessions created by this package, not registrations by other
// libraries. The caller must still arrange exclusive external collector use.
var samplingOwner struct {
	sync.Mutex
	session *samplingSession
}

func (s *samplingSession) acquire() error {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	if samplingOwner.session != nil {
		return errors.New("cuptipcsampling: another session owns PC sampling")
	}
	samplingOwner.session = s
	return nil
}

func (s *samplingSession) releaseOwnership() {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	if samplingOwner.session == s {
		samplingOwner.session = nil
	}
}

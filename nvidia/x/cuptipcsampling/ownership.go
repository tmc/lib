package cuptipcsampling

import (
	"errors"
	"sync"
)

// This protects sessions created by this package, not registrations by other
// libraries. The caller must still arrange exclusive external collector use.
var samplingOwner struct {
	sync.Mutex
	session     *samplingSession
	sessionRuns uint64
}

// SessionRuns reports how many PC sampling sessions have been initiated in this
// process. On GB200/GB10 (CUPTI 2025.x / 13.x), re-enabling PC sampling across
// multiple sessions in a single process risks native heap corruption; workloads
// should measure across a single continuous session whenever possible.
func SessionRuns() uint64 {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	return samplingOwner.sessionRuns
}

// acquire takes ownership for s. A session that already holds the claim made
// on its behalf by reserveOwner may acquire again, so the claim and the
// acquisition do not fight over the same slot.
func (s *samplingSession) acquire() error {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	if samplingOwner.session != nil && samplingOwner.session != s {
		return errors.New("cuptipcsampling: another session owns PC sampling")
	}
	samplingOwner.session = s
	return nil
}

// reserveOwner claims the collector before a session exists, so that a start
// rejected for exclusivity never enables sampling or applies a configuration.
// adopt transfers the claim to the session that will own the cleanup; release
// gives it back, and both are no-ops once the other has run.
func reserveOwner() (adopt func(*samplingSession), release func(), err error) {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	if samplingOwner.session != nil {
		return nil, nil, errors.New("cuptipcsampling: another session owns PC sampling")
	}
	samplingOwner.sessionRuns++
	claim := new(samplingSession)
	samplingOwner.session = claim
	adopt = func(s *samplingSession) {
		samplingOwner.Lock()
		defer samplingOwner.Unlock()
		if samplingOwner.session == claim {
			samplingOwner.session = s
		}
	}
	release = func() {
		samplingOwner.Lock()
		defer samplingOwner.Unlock()
		if samplingOwner.session == claim {
			samplingOwner.session = nil
		}
	}
	return adopt, release, nil
}

func (s *samplingSession) releaseOwnership() {
	samplingOwner.Lock()
	defer samplingOwner.Unlock()
	if samplingOwner.session == s {
		samplingOwner.session = nil
	}
}

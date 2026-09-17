package cudagroup

import "errors"

// TMEMPeers selects allocation-management issue granularity, independently
// of the size of the enclosing cluster. Other tcgen05 operations differ.
type TMEMPeers uint8

const (
	TMEMGroup1 TMEMPeers = 1
	TMEMGroup2 TMEMPeers = 2
)

// Issuer names a CTA rank within a cluster and a linear thread rank in that CTA.
type Issuer struct{ CTA, Thread uint64 }

// TMEMCollective describes one issuing warp per participating CTA for
// tcgen05 allocation, deallocation, or permit relinquishment. It does not
// allocate memory or prove ordering, allocation units, or permit lifetime.
// Construct it with NewTMEMCollective; the zero value is invalid.
type TMEMCollective struct {
	peers                 TMEMPeers
	cluster               Group
	block                 Dim3
	firstCTA, firstThread uint64
}

// NewTMEMCollective admits the sm_100a subset of PTX 8.6 allocation-management
// participation. Group2 requires an even/odd peer pair within the cluster.
// This bounded model uses the same issuing warp index in each peer CTA.
func NewTMEMCollective(target string, peers TMEMPeers, cluster Group, block Dim3, firstCTA uint64, warp uint32) (TMEMCollective, error) {
	if target != "sm_100a" {
		return TMEMCollective{}, errors.New("unsupported tmem participation target")
	}
	if peers != TMEMGroup1 && peers != TMEMGroup2 {
		return TMEMCollective{}, errors.New("unsupported tmem peer group")
	}
	if cluster.kind != Cluster {
		return TMEMCollective{}, errors.New("tmem participation requires a cluster description")
	}
	cta, err := New(CTA, block)
	if err != nil {
		return TMEMCollective{}, err
	}
	if firstCTA >= cluster.Size() || (peers == TMEMGroup2 && (firstCTA%2 != 0 || firstCTA+1 >= cluster.Size())) {
		return TMEMCollective{}, errors.New("tmem peer pair outside cluster or misaligned")
	}
	firstThread := uint64(warp) * 32
	if firstThread+32 > cta.Size() {
		return TMEMCollective{}, errors.New("tmem issuing warp is incomplete")
	}
	return TMEMCollective{peers: peers, cluster: cluster, block: block, firstCTA: firstCTA, firstThread: firstThread}, nil
}

// Peers returns the instruction's CTA group qualifier, not the cluster size.
func (c TMEMCollective) Peers() TMEMPeers { return c.peers }

// ValidateLaunch checks the cluster and block containing the issuing warps.
func (c TMEMCollective) ValidateLaunch(req LaunchRequirements) error {
	if c.peers == 0 {
		return errors.New("zero tmem collective")
	}
	if req.Block != c.block {
		return errors.New("tmem block shape differs from launch")
	}
	return ValidateLaunch(req, c.cluster)
}

// ValidateParticipation checks complete issuing warps and uniform arguments.
// Each argument could be nCols; legal values are checked by cudatmem.
func (c TMEMCollective) ValidateParticipation(issuers []Issuer, args []uint32) error {
	if c.peers == 0 {
		return errors.New("zero tmem collective")
	}
	if len(issuers) != int(c.peers)*32 || len(args) != len(issuers) {
		return errors.New("incomplete tmem participation")
	}
	seen := make(map[Issuer]bool, len(issuers))
	for i, issuer := range issuers {
		if issuer.CTA < c.firstCTA || issuer.CTA >= c.firstCTA+uint64(c.peers) || issuer.Thread < c.firstThread || issuer.Thread >= c.firstThread+32 {
			return errors.New("wrong tmem issuing warp or peer cta")
		}
		if seen[issuer] {
			return errors.New("duplicate tmem participant")
		}
		seen[issuer] = true
		if args[i] != args[0] {
			return errors.New("nonuniform tmem argument")
		}
	}
	return nil
}

// Package cudacollective provides rank-ordered collective communication plans.
package cudacollective

import (
	"fmt"
)

type LegacyTransportType int

const (
	TransportNVLinkMulticast LegacyTransportType = iota
	TransportRDMAEager
	TransportTCPFallback
)

type CollectiveKind int

const (
	CollectiveAllReduce CollectiveKind = iota
	CollectiveAllGather
	CollectiveReduceScatter
)

type PlanCommunicator struct {
	Generation uint64
	Ranks      uint32
	RankID     uint32
}

type CollectivePlan struct {
	Comm      PlanCommunicator
	Kind      CollectiveKind
	Transport TransportType
	Count     uint64
}

func ResolveTransport(hasNVLink bool, hasRDMA bool) LegacyTransportType {
	if hasNVLink {
		return TransportNVLinkMulticast
	}
	if hasRDMA {
		return TransportRDMAEager
	}
	return TransportTCPFallback
}

func NewCollectivePlan(comm PlanCommunicator, kind CollectiveKind, count uint64, hasNVLink, hasRDMA bool) (*CollectivePlan, error) {
	if comm.Ranks == 0 || comm.RankID >= comm.Ranks {
		return nil, fmt.Errorf("cudacollective: invalid rank ID %d for total ranks %d", comm.RankID, comm.Ranks)
	}
	transport := ResolveTransport(hasNVLink, hasRDMA)
	return &CollectivePlan{
		Comm:      comm,
		Kind:      kind,
		Transport: TransportType(transport + 1),
		Count:     count,
	}, nil
}

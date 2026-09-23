package nccltest

import "testing"

// alwaysQualified are the Team-by-value host calls that must succeed against
// any installed NCCL. The two CFT calls are not listed: their availability
// depends on the platform, and TestTeamCalls checks them against the
// communicator's actual CFT support instead.
var alwaysQualified = []string{
	"TeamRankToWorld",
	"TeamRankToLsa",
	"LsaBarrierCreateRequirement",
	"GinBarrierCreateRequirement",
}

func TestTeamCalls(t *testing.T) {
	requireCUDA(t)

	report, err := CheckTeamCalls(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("NCCL %d", report.Version)

	status := make(map[string]TeamCallResult, len(report.Calls))
	for _, call := range report.Calls {
		status[call.Name] = call
		t.Logf("%-28s %s %s", call.Name, call.Status, call.Detail)
	}
	t.Logf("%-28s %s %s", report.DevCommCftCaps.Name, report.DevCommCftCaps.Status, report.DevCommCftCaps.Detail)

	for _, name := range alwaysQualified {
		call, ok := status[name]
		if !ok {
			t.Errorf("%s: not reported", name)
			continue
		}
		if call.Status != TeamCallQualified {
			t.Errorf("%s: %s: %s", name, call.Status, call.Detail)
		}
	}

	// A CFT call may be absent or refused on a platform without CFT support.
	// Once the communicator admits CFT capabilities, refusal is a binding
	// defect rather than a platform limit.
	cftSupported := report.DevCommCftCaps.Status == TeamCallQualified
	if call := status["GetCftDeviceLeInfo"]; cftSupported && call.Status != TeamCallQualified {
		t.Errorf("GetCftDeviceLeInfo: %s with CFT-capable device communicator: %s", call.Status, call.Detail)
	}
	if call := status["CftBarrierCreateRequirement"]; call.Status == TeamCallUnqualified {
		t.Errorf("CftBarrierCreateRequirement: exported but rejected: %s", call.Detail)
	}
}

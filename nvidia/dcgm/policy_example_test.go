package dcgm_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/dcgm"
)

func ExamplePolicyBool() {
	p := dcgm.PolicyBool(true)
	fmt.Println(p.Tag, p.Val[0])
	// Output: 0 1
}

func ExamplePolicyNumber() {
	p := dcgm.PolicyNumber(80)
	fmt.Println(p.Tag, p.Val[0])
	// Output: 1 80
}

func ExampleReadPolicies() {
	_, err := dcgm.ReadPolicies(0, 0, 0, 0)
	fmt.Println(err)
	// Output: dcgm: invalid policy array extent
}

func ExampleNewPolicyNotifications() {
	_, err := dcgm.NewPolicyNotifications(0, 0, dcgm.PolicyCondDbe, 1)
	fmt.Println(err)
	// Output: dcgm: invalid policy notification configuration
}

func ExamplePolicyNotifications_Close() {
	var notifications dcgm.PolicyNotifications
	fmt.Println(notifications.Close())
	// Output: dcgm: invalid policy notifications
}

func ExamplePolicyNotifications_Notifications() {
	var notifications dcgm.PolicyNotifications
	fmt.Println(notifications.Notifications() == nil)
	// Output: true
}

func ExamplePolicyNotifications_Dropped() {
	var notifications dcgm.PolicyNotifications
	fmt.Println(notifications.Dropped())
	// Output: 0
}

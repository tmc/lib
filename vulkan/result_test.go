package vulkan

import "testing"

func TestResultError(t *testing.T) {
	if err := Success.Err(); err != nil {
		t.Fatalf("Success.Err() = %v", err)
	}
	if ErrorOutOfHostMemory.Err() == nil {
		t.Fatal("negative result should be an error")
	}
	if err := Incomplete.Err(); err != nil {
		t.Fatalf("positive status should not be an error: %v", err)
	}
	if got := ErrorIncompatibleDriver.Error(); got != "VK_ERROR_INCOMPATIBLE_DRIVER" {
		t.Fatalf("Error() = %q", got)
	}
}

func TestAPIVersion(t *testing.T) {
	v := MakeAPIVersion(0, 1, 3, 268)
	if APIVersionMajor(v) != 1 || APIVersionMinor(v) != 3 || APIVersionPatch(v) != 268 {
		t.Fatalf("version decode failed: %#x", v)
	}
}

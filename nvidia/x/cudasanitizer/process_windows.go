package cudasanitizer

import (
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// A job owns the tool and the children it creates. Starting suspended closes
// the window in which a running tool could spawn a child before assignment.
type processGroup struct{ job windows.Handle }

func startGroup(cmd *exec.Cmd) (*processGroup, error) {
	return startGroupWithResume(cmd, resumeProcess)
}

func startGroupWithResume(cmd *exec.Cmd, resume func(uint32) error) (_ *processGroup, err error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create process job: %w", err)
	}
	g := &processGroup{job: job}
	defer func() {
		if err != nil {
			if closeErr := g.close(); closeErr != nil {
				err = errors.Join(err, &cleanupError{closeErr})
			}
		}
	}()
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	// The Windows wrapper accepts this record as uintptr rather than a pointer.
	var pin runtime.Pinner
	pin.Pin(&info)
	defer pin.Unpin()
	if _, err = windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation, uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		return nil, fmt.Errorf("set process job limits: %w", err)
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_SUSPENDED
	if err = cmd.Start(); err != nil {
		return nil, err
	}
	// Until assignment succeeds, closing the job does not own this process.
	// Reap it on every setup failure, including failure to resume the thread.
	defer func() {
		if err != nil {
			_ = g.kill()
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	process, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		return nil, fmt.Errorf("open suspended process: %w", err)
	}
	defer windows.CloseHandle(process)
	if err = windows.AssignProcessToJobObject(job, process); err != nil {
		return nil, fmt.Errorf("assign process job: %w", err)
	}
	if err = resume(uint32(cmd.Process.Pid)); err != nil {
		return nil, err
	}
	return g, nil
}

func (g *processGroup) kill() error  { return windows.TerminateJobObject(g.job, 1) }
func (g *processGroup) close() error { return windows.CloseHandle(g.job) }

// os/exec closes the primary thread handle returned by CreateProcess. Find
// that thread while the new process is still suspended and cannot create more.
func resumeProcess(pid uint32) error {
	snapshot, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return fmt.Errorf("snapshot suspended thread: %w", err)
	}
	defer windows.CloseHandle(snapshot)
	entry := windows.ThreadEntry32{Size: uint32(unsafe.Sizeof(windows.ThreadEntry32{}))}
	for err = windows.Thread32First(snapshot, &entry); err == nil; err = windows.Thread32Next(snapshot, &entry) {
		if entry.OwnerProcessID != pid {
			continue
		}
		thread, err := windows.OpenThread(windows.THREAD_SUSPEND_RESUME, false, entry.ThreadID)
		if err != nil {
			return fmt.Errorf("open suspended thread: %w", err)
		}
		count, err := windows.ResumeThread(thread)
		windows.CloseHandle(thread)
		if err != nil {
			return fmt.Errorf("resume suspended thread: %w", err)
		}
		if count != 1 {
			return fmt.Errorf("resume suspended thread: suspend count %d, want 1", count)
		}
		return nil
	}
	if err != windows.ERROR_NO_MORE_FILES {
		return fmt.Errorf("enumerate suspended thread: %w", err)
	}
	return fmt.Errorf("find suspended thread: process %d has no thread", pid)
}

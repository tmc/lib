package fswatch

import (
	"context"
	"errors"
	"io/fs"
	"path/filepath"
	"time"

	"golang.org/x/sys/unix"
)

type notifier struct{ fd int }

func openNotify(dirs []string, poll bool) (*notifier, error) {
	if poll {
		return &notifier{-1}, nil
	}
	fd, err := unix.InotifyInit1(unix.IN_CLOEXEC | unix.IN_NONBLOCK)
	if err != nil {
		return nil, err
	}
	n := &notifier{fd}
	for _, dir := range dirs {
		err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || !entry.IsDir() {
				return err
			}
			_, err = unix.InotifyAddWatch(fd, path, unix.IN_CLOSE_WRITE|unix.IN_MOVED_TO|unix.IN_MOVED_FROM|unix.IN_CREATE|unix.IN_DELETE|unix.IN_DELETE_SELF|unix.IN_MOVE_SELF)
			return err
		})
		if err != nil {
			n.close()
			return nil, err
		}
	}
	return n, nil
}

func (n *notifier) wait(ctx context.Context, interval time.Duration) error {
	if n.fd < 0 {
		return waitPoll(ctx, interval)
	}
	end := time.Now().Add(interval)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		left := time.Until(end)
		if left <= 0 {
			return nil
		}
		if left > 50*time.Millisecond {
			left = 50 * time.Millisecond
		}
		millis := (left + time.Millisecond - 1) / time.Millisecond
		count, err := unix.Poll([]unix.PollFd{{Fd: int32(n.fd), Events: unix.POLLIN}}, int(millis))
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if err != nil {
			return err
		}
		if count != 0 {
			break
		}
	}
	var buf [16384]byte
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		count, err := unix.Read(n.fd, buf[:])
		if errors.Is(err, unix.EAGAIN) {
			return nil
		}
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if err != nil {
			return err
		}
		if count == 0 {
			return nil
		}
	}
}

func (n *notifier) close() {
	if n.fd >= 0 {
		unix.Close(n.fd)
	}
}

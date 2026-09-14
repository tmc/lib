.PHONY: all build test check gate-selftest

all: check build test

build:
	@mods=$$(go list -m -f '{{.Dir}}') || exit 1; \
	[ -n "$$mods" ] || { echo "error: no modules found" >&2; exit 1; }; \
	printf '%s\n' "$$mods" | while IFS= read -r dir; do \
		echo "==> Building $$dir"; \
		(cd "$$dir" && go build ./...) || exit 1; \
	done

test:
	@mods=$$(go list -m -f '{{.Dir}}') || exit 1; \
	[ -n "$$mods" ] || { echo "error: no modules found" >&2; exit 1; }; \
	printf '%s\n' "$$mods" | while IFS= read -r dir; do \
		echo "==> Testing $$dir"; \
		(cd "$$dir" && go test ./...) || exit 1; \
	done

check:
	@files=$$(git ls-files --cached --others --exclude-standard '*/go.mod') || exit 1; \
	printf '%s\n' "$$files" | while IFS= read -r f; do \
		[ -f "$$f" ] || continue; \
		d=$$(dirname "$$f"); \
		grep -Fqx "	./$$d" go.work || { echo "error: $$d is not listed in go.work"; exit 1; }; \
	done || exit 1; \
	echo "go.work covers every module."
	@files=$$(git ls-files --cached --others --exclude-standard '*.go') || exit 1; \
	printf '%s\n' "$$files" | while IFS= read -r f; do \
		[ -f "$$f" ] || continue; \
		dir=$$(dirname "$$f"); \
		found=0; \
		while [ "$$dir" != "." ] && [ "$$dir" != "/" ]; do \
			if [ -f "$$dir/go.mod" ]; then \
				found=1; \
				break; \
			fi; \
			dir=$$(dirname "$$dir"); \
		done; \
		if [ $$found -eq 0 ]; then \
			echo "error: $$f is not part of any module"; \
			exit 1; \
		fi; \
	done || exit 1; \
	echo "All Go files belong to a module."

gate-selftest:
	@sh scripts/gate-selftest.sh

# Cross-compiles every module for the platforms scripts/platforms.txt claims.
# Build-tagged files are invisible to a single-platform build, so this is the
# only gate that compiles them.
.PHONY: platform-check
platform-check:
	@sh scripts/check-platforms.sh

# Metadata and cross-platform builds; see RELEASE.md for consumer and runtime
# verification.
.PHONY: release-check release-selftest
release-check: check platform-check
	@python3 scripts/check-release.py

release-selftest:
	@sh scripts/release-selftest.sh

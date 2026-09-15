.PHONY: all build test check gate-selftest

all: check build test

build:
	@mods=$$(go list -m -f '{{.Dir}}') || exit 1; \
	[ -n "$$mods" ] || { echo "error: no modules found" >&2; exit 1; }; \
	printf '%s\n' "$$mods" | while IFS= read -r dir; do \
		echo "==> Building $$dir"; \
		(cd "$$dir" && go build ./...) || exit 1; \
	done; \
	echo "Built $$(printf '%s\n' "$$mods" | wc -l | tr -d ' ') modules."

# The module list comes from go.work, which is itself a tracked file that a
# restructuring edits, so a green run here is green over whatever go.work says
# today -- 13 modules in a tree with two deleted, 15 at HEAD. And the loop stops
# at the first failing module, so a red run withholds the verdict on the rest.
# Both are why the count is printed: a result without its denominator is not a
# result.
test:
	@mods=$$(go list -m -f '{{.Dir}}') || exit 1; \
	[ -n "$$mods" ] || { echo "error: no modules found" >&2; exit 1; }; \
	total=$$(printf '%s\n' "$$mods" | wc -l | tr -d ' '); \
	done_n=0; \
	IFS=$$(printf '\nx'); IFS=$${IFS%x}; \
	for dir in $$mods; do \
		echo "==> Testing $$dir"; \
		(cd "$$dir" && go test ./...) || { \
			echo "error: failed in $$dir; $$((total - done_n - 1)) of $$total modules did not run" >&2; \
			exit 1; \
		}; \
		done_n=$$((done_n + 1)); \
	done; \
	echo "Tested $$total modules."

check:
	@absent=$$(git ls-files --cached '*/go.mod' | while IFS= read -r f; do \
		[ -f "$$f" ] || dirname "$$f"; \
	done); \
	if [ -n "$$absent" ]; then \
		echo "note: module(s) tracked at HEAD but absent from the working tree, so every"; \
		echo "      check below skipped them -- this run covers less than the repository:"; \
		printf '%s\n' "$$absent" | sed 's/^/        /'; \
	fi
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
	@shadowed=$$(git ls-files --ignored --cached --exclude-standard) || exit 1; \
	if [ -n "$$shadowed" ]; then \
		echo "error: .gitignore claims files that are tracked:"; \
		printf '%s\n' "$$shadowed" | sed 's/^/  /'; \
		echo "a bare pattern matches directories too; anchor it to the path it means"; \
		exit 1; \
	fi; \
	echo "No tracked file is ignored."

gate-selftest:
	@sh scripts/gate-selftest.sh

# Cross-compiles every module for the platforms scripts/platforms.txt claims.
# Build-tagged files are invisible to a single-platform build, so this is the
# only gate that compiles them.
.PHONY: platform-check
platform-check:
	@sh scripts/check-platforms.sh

# Compiles the test packages behind build tags like cupti_physical. Nothing a
# workstation runs selects those tags, so without this they can stop building
# for months while every other gate stays green.
.PHONY: testtag-check
testtag-check:
	@sh scripts/check-testtags.sh

# Metadata and cross-platform builds; see RELEASE.md for consumer and runtime
# verification.
.PHONY: release-check release-selftest
release-check: check platform-check testtag-check
	@python3 scripts/check-release.py

release-selftest:
	@sh scripts/release-selftest.sh

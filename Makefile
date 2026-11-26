# Thin wrapper around tasks.py, so the two cannot drift. On Windows without make, run:
#   uv run python tasks.py <target>
# Pass extra arguments to the last command with ARGS, e.g.  make test ARGS="-k leakage"

TASKS := uv run python tasks.py
TARGETS := sync format lint typecheck eol test test-network coverage check \
           validate-data train evaluate pipeline serve export-model \
           docker-build docker-run docker-smoke clean

.PHONY: help $(TARGETS)

help:
	@$(TASKS) --help

$(TARGETS):
	$(TASKS) $@ $(if $(ARGS),-- $(ARGS))

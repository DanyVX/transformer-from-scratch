.PHONY: setup lint format typecheck test test-fast bench run docker-build clean demo

setup lint format typecheck test test-fast bench run docker-build clean demo:
	python scripts/tasks.py $@

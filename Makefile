SHELL=/bin/bash
PATHS=magbotsim/ tests/ setup.py

pytest:
	python3 -m pytest -v --tb=short

format:
	ruff format ${PATHS}

check-codestyle:
	ruff check ${PATHS} 

spelling:
	cd docs && make spelling

doc: 
	cd docs && make html

clean:
	cd docs && make clean

commit: format check-codestyle spelling pytest doc

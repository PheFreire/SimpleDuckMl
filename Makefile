.DEFAULT_GOAL := help
.SILENT:
.PHONY: help

help:  ## Display this help
	awk 'BEGIN {FS = ":.*## "; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Format

format-black: ## run black (code formatter)
	echo "🎨 Formatting code with \033[36mblack\033[0m..."
	poetry run black .

format-isort: ## run isort (imports formatter)
	echo "📦 Sorting imports with \033[36misort\033[0m..."
	poetry run isort .

format: format-black format-isort ## run all formatters
	echo "✅ All formatters applied."

##@ Check

check-bandit: ## run bandit (check for common security issues)
	echo "🔐 Running \033[36mbandit\033[0m for security checks..."
	poetry run bandit -r ./src --skip B605

check-black: ## run black in check mode
	echo "🧪 Checking code formatting with \033[36mblack\033[0m..."
	poetry run black . --check

check-isort: ## run isort in check mode
	echo "🧪 Checking import order with \033[36misort\033[0m..."
	poetry run isort . --check

check-flake8: ## run flake8 (pep8 linter)
	echo "🧹 Linting with \033[36mflake8\033[0m..."
	poetry run flake8 ./src --ignore=E501,W503

check-mypy: ## run mypy (static-type checker)
	echo "🔍 Running static type checks with \033[36mmypy\033[0m..."
	poetry run mypy ./src

check-mypy-report: ## run mypy & create report
	echo "📊 Generating static type report with \033[36mmypy\033[0m..."
	poetry run mypy ./src --html-report ./mypy_html

check: check-bandit check-black check-flake8 check-mypy ## run all checks
	echo "✅ All checks completed successfully."

##@ Test

test: ## run tests
	echo "🧪 Running tests with \033[36mpytest\033[0m..."
	poetry run pytest | tee tests.log




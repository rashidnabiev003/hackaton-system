.PHONY: install build down up


# Colors for pretty output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Local development
install: ## Install dependencies via uv
	@printf "${YELLOW}Installing dependencies...${NC}\n"
	uv sync

build:
	@printf "${YELLOW}Building Docker images...${NC}\n"
	docker compose build

down:
	@printf "${YELLOW}Stopping Docker containers...${NC}\n"
	docker compose down -v

up:
	@printf "${YELLOW}Starting Docker containers...${NC}\n"
	docker compose up -d

# Utilities
lint: ## Check code via ruff
	@printf "${YELLOW}Checking code...${NC}\n"
	@printf "${CYAN}Running ruff check...${NC}\n"
	uv run ruff check hackaton_system tests; RUFF_CHECK=$$?; \
	printf "${CYAN}Running ruff format check...${NC}\n"; \
	uv run ruff format --check hackaton_system tests; RUFF_FORMAT=$$?; \
	# printf "${CYAN}Running wemake-python-styleguide...${NC}\n"; \
	# uv run flake8 hackaton_system tests --select=WPS; FLAKE8=$$?; \
	printf "\n${CYAN}Lint Results:${NC}\n"; \
	if [ $$RUFF_CHECK -eq 0 ]; then printf "${GREEN}ruff check passed${NC}\n"; else printf "${RED}ruff check failed${NC}\n"; fi; \
	if [ $$RUFF_FORMAT -eq 0 ]; then printf "${GREEN}ruff format check passed${NC}\n"; else printf "${RED}ruff format check failed${NC}\n"; fi; \
	# if [ $$FLAKE8 -eq 0 ]; then printf "${GREEN}flake8 passed${NC}\n"; else printf "${RED}flake8 failed${NC}\n"; fi; \
	if [ $$RUFF_CHECK -ne 0 ] || [ $$RUFF_FORMAT -ne 0 ] || [ $$FLAKE8 -ne 0 ]; then exit 1; fi

format: ## Format code
	@printf "${YELLOW}Formatting code...${NC}\n"
	@printf "${CYAN}Running ruff format...${NC}\n"
	uv run ruff format hackaton_system tests
	@printf "${CYAN}Running ruff check --fix (includes isort)...${NC}\n"
	uv run ruff check --fix hackaton_system tests

test: ## Run tests
	@printf "${YELLOW}Running tests...${NC}\n"
	uv run pytest
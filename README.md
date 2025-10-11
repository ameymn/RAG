# 1) Clone and enter the project
git clone git@github.com:ameymn/RAG.git
cd RAG

# 2) Install dependencies from pyproject.toml (creates .venv automatically)
uv sync

# 3) Create your environment file
cp .env.example .env
# then open .env and set your keys (see "Configuration" below)

# 4) Run the app
uv run streamlit run main.py


# Some UV commands
# To Run the Streamlit app
uv run streamlit run main.py

# Add a new dependency
uv add <package>

# Update locked versions
uv lock

# Remove a dependency
uv remove <package>

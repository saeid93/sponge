#!/bin/bash

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Set the directory where Poetry is installed
POETRY_INSTALL_DIR="$HOME/.poetry/bin"

# Add the directory to PATH if not already present
if [[ ":$PATH:" != *":$POETRY_INSTALL_DIR:"* ]]; then
    echo "Adding Poetry directory to PATH..."
    echo "export PATH=\"$POETRY_INSTALL_DIR:\$PATH\"" >> "$HOME/.zshrc"
    # source "$HOME/.zshrc"
    echo "Poetry directory has been added to PATH."
else
    echo "Poetry directory is already in PATH."
fi

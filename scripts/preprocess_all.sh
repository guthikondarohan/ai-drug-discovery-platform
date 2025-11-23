#!/bin/bash

set -e

echo "Running preprocessing stage via Python CLI..."
python -m src.main --stage preprocess

echo "Done."

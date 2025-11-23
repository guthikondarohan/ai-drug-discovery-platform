#!/bin/bash

set -e

# Run the full pipeline (preprocess -> train -> evaluate) using the package entry
python -m src.main --stage all
try {
    Write-Host "Running full pipeline (preprocess -> train -> evaluate) via Python CLI..."
    python -m src.main --stage all
    Write-Host "Done."
} catch {
    Write-Error $_
    exit 1
}
try {
    Write-Host "Running training via main.py..."
    python main.py --stage training
    Write-Host "Done."
} catch {
    Write-Error $_
    exit 1
}

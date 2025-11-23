try {
    Write-Host "Running preprocessing stage via Python CLI..."
    python -m src.main --stage preprocess
    Write-Host "Done."
} catch {
    Write-Error $_
    exit 1
}
try {
    Write-Host "Preprocessing molecules..."
    python -c "from src import data_preprocessing as dp; df = dp.preprocess_molecules('data/molecules.csv' if __import__('pathlib').Path('data/molecules.csv').exists() else 'data/molecules_sample.csv'); df.to_csv('data/processed/molecules_preprocessed.csv', index=False)"
    Write-Host "Preprocessing clinical text..."
    python -c "from src import data_preprocessing as dp; df = dp.preprocess_text('data/clinical_text.csv' if __import__('pathlib').Path('data/clinical_text.csv').exists() else 'data/clinical_text_sample.csv'); df.to_csv('data/processed/clinical_text_preprocessed.csv', index=False)"
    Write-Host "Done."
} catch {
    Write-Error $_
    exit 1
}

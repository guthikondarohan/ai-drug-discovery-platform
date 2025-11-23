from pathlib import Path
import sys

# Ensure project root is on sys.path so `src` can be imported when running scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_preprocessing as dp


def main():
    in_path = Path('data') / 'molecules_sample.csv'
    out_dir = Path('data') / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = dp.preprocess_molecules(str(in_path))
    out_file = out_dir / 'molecules_preprocessed.csv'
    df.to_csv(out_file, index=False)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()

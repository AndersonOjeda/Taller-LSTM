"""Crea un ZIP portable con fuentes y resultados, sin entornos ni caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
destination = ROOT / 'output' / 'Taller_LSTM_Anderson_Entrega.zip'
destination.parent.mkdir(exist_ok=True)
files = [p for p in ROOT.iterdir()
         if p.is_file() and p.suffix in {'.py', '.md', '.txt', '.pdf', '.ipynb', '.csv', '.cmd'}]
files += [ROOT / '.gitignore', ROOT / '.python-version']
for folder in ['data', 'models', 'results']:
    files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, 'taller_lstm_anderson/' + path.relative_to(ROOT).as_posix())
print(destination)

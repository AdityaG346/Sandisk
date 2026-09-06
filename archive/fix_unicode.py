"""Fix all non-ASCII characters in src/*.py and run_all.py."""
import pathlib

replacements = [
    ('\u2014', '--'),
    ('\u2013', '-'),
    ('\u2192', '->'),
    ('\u2248', '~='),
    ('\u2019', "'"),
    ('\u2018', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
]

files = list(pathlib.Path('src').glob('*.py')) + [pathlib.Path('run_all.py')]
for f in files:
    txt = f.read_text(encoding='utf-8')
    new_txt = txt
    for bad, good in replacements:
        new_txt = new_txt.replace(bad, good)
    if new_txt != txt:
        f.write_text(new_txt, encoding='utf-8')
        print(f'Fixed: {f}')
    else:
        print(f'OK:    {f}')
print('Done.')

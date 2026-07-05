from pathlib import Path
path = Path('video_module.py')
lines = path.read_text(encoding='utf-8').splitlines()
for i in range(105, 121):
    print(f'{i+1}: {lines[i]!r}')

from audio_module import extract_audio_file_features
import json

f = extract_audio_file_features(r'data/video/puppy crying.mp4')
summary = {k: float(f[k]) for k in ('energy','zcr','centroid','bandwidth','duration')}
print(json.dumps(summary, indent=2))

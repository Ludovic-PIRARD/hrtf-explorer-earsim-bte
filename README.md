# HRTF Explorer — EarSim vs BTE

Interactive comparison of KEMAR **EarSim** (blocked ear canal) vs **BTE** (Meta Quest 3 + Cochlear Nucleus 5 mic ports).

- Windowed HRIRs with ITD, 48 kHz, 793 matched directions  
- Direction overlay / difference plots, slice spectrograms, ITD/ILD, binaural preview  
- UI adapted from [yoyolicoris/hrtf-explorer](https://yoyolicoris.github.io/hrtf-explorer/)

## Live site

**https://ludovic-pirard.github.io/hrtf-explorer-earsim-bte/**

## Local

```bash
python3 -m http.server 8765
# open http://127.0.0.1:8765/
```

## Regenerate data

```bash
python3 export_explorer_data.py
```

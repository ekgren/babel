---
lang: en
---
# Large Language Model Architectures

SOTA architectures

- [flash attention 2](pages/flash-attention.md)
- parallel attention and feedforward layers
- rotary embeddings
- pre-layer norm
- probably 8/3 h multipliers 

Basically Mistral + parallel layers (they left a free +10% performance on the table).

```twitter
Stella Biderman @BlancheMinerva 6 nov.
Use flash attention 2, parallel attention and feedforward layers, rotary embeddings, pre-layer norm, and probably 8/3 h multipliers but that doesn't matter too much. Basically Mistral + parallel layers (they left a free +10% performance on the table).
```
Source: https://x.com/BlancheMinerva/status/1721380386515669209?s=20

Oh and spend as much time on data processing as you can. Not just throwing out bad data, but improving the formatting of your scraped data and books will go a long ways. This will match LLaMA equi-compute and if you spend enough money you'll get a better model than Mistral
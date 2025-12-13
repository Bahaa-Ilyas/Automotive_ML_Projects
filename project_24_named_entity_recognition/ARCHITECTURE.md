# Architecture: Named Entity Recognition

## Pipeline
```
Text → Tokenization → spaCy NER → Entity Extraction → Structured Output
```

## spaCy Pipeline
- Tokenizer
- Tagger (POS)
- Parser (Dependency)
- NER (Entity Recognition)

## Custom Entities
- PART: brake pads, spark plugs
- SYMPTOM: squealing, grinding
- ACTION: replace, inspect
- MEASUREMENT: 3mm, 32 PSI

## Next Steps
→ Project 25: Text Summarization with BART (140M)

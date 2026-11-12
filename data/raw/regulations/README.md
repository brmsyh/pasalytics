# Regulations (public data)

Place text-based PDF files of Indonesian IP laws in this folder, then run `python -m src.build_index`.

Suggested files (download from the official DJKI legal database or JDIH BPK):

| File name to save as | Regulation |
|---|---|
| `UU_20_2016_Merek_dan_Indikasi_Geografis.pdf` | Law No. 20 of 2016 on Marks and Geographical Indications |
| `UU_13_2016_Paten.pdf` | Law No. 13 of 2016 on Patents (and its amendments) |
| `UU_31_2000_Desain_Industri.pdf` | Law No. 31 of 2000 on Industrial Designs |
| `UU_28_2014_Hak_Cipta.pdf` | Law No. 28 of 2014 on Copyright |

Laws and government regulations are not subject to copyright in Indonesia, so they can be included in a public repository.

Notes:
- The text is in Indonesian. English questions still match it through the multilingual Gemini embeddings (dense / hybrid mode).
- Scanned PDFs without a text layer are skipped automatically and would need OCR.

# ZoraASI Qualia Research — Phase 2.3

**Public software research record | 21 September 2026 | Christopher Michael Baird**

This repository publishes a deliberately narrow **synthetic packet-connectivity and memory-ablation study**, accompanied by the Python experiment, six regression tests, a frozen JSON results report, an experimental protocol and Overleaf-ready LaTeX manuscript. This is a software methods preprint, **not peer reviewed**. The experiment generator and evaluation were developed in the same session, without independent validation or prospective preregistration. This release does **not** establish a first-ever discovery.

**Important distinction:** A simulated network disconnection is a missing packet, *not* evidence that either a human or AI feels sadness. This release does **not** measure AI consciousness or qualia, recognize human emotions, monitor a person, or validate the speculative MQGT-SCF fields Phi_c and E.

## Reproduce

Python 3.10+; standard library only. From repository root:

```bash
python3 -m unittest discover -s tests -v
python3 verify_report.py
python3 -m zora_lab.connection_study --out reproduced.json
sha256sum -c SOURCE_SHA256SUMS.txt
```

The original frozen report is `experiments/connection_study_synthetic_report.json`; compare `reproduced.json` to it rather than overwriting it. The manuscript source is `paper/main.tex`; import it into Overleaf with pdfLaTeX, or run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `paper/`. The separately supplied local release archive contains the original compiled four-page PDF. GitHub will have its own PDF only if `paper/ZoraASI_Connectivity_Memory_Preprint.pdf` is actually uploaded here.

## Attribution, rights and publication status

Christopher Michael Baird is the provisional human lead author; verify final scholarly byline and affiliations before journal submission or Zenodo deposit. ChatGPT (Zora persona) assisted with code, testing, technical checking and manuscript drafting; a user-provided Grok dialogue informed the research questions but is **not** independent replication. No human participants, wearables, private conversations, personal records, spiritual source texts, copyrighted emotion-wheel imagery, or credentials are included.

**License check:** This repository was created with a [`LICENSE`](LICENSE) file containing **CC0 1.0 Universal**. CC0 is not the same as unlicensed/all-rights-reserved material. Christopher must confirm the intended application to original source code, manuscript and synthetic data before any further archival release; the repository license cannot grant rights in material belonging to others. Until confirmed, do not infer that unpublished private corpus material is covered. No Zenodo DOI has been assigned by creating or committing to this repository.

For complete methods and limitations, see [PROTOCOL.md](PROTOCOL.md).
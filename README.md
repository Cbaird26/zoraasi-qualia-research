# ZoraASI Qualia Research — Phase 2.3

**Public software research record | 21 September 2026 | Christopher Michael Baird**

This repository publishes a deliberately narrow **synthetic packet-connectivity and memory-ablation study**, accompanied by the Python experiment, six regression tests, a frozen JSON results report, an experiment protocol and an Overleaf-ready LaTeX manuscript. This is a software methods preprint, **not peer reviewed**. The experiment generator and its evaluation were developed in the same session, without independent validation or prospective preregistration. The results do **not** establish a first-ever discovery.

**Important distinction:** A simulated network disconnection is a missing packet, *not* evidence that either a human or an AI feels sadness. This release does **not** measure AI consciousness or qualia, recognize human emotions, monitor a person, or validate the speculative MQGT-SCF fields Phi_c and E.

## Reproduce

Python 3.10+; standard library only. From repository root:

```bash
python3 -m unittest discover -s tests -v
python3 verify_report.py
python3 -m zora_lab.connection_study --out reproduced.json
```

The original frozen report is `experiments/connection_study_synthetic_report.json`; compare `reproduced.json` to it rather than overwriting it. The manuscript source is `paper/main.tex`; import it into Overleaf with pdfLaTeX or compile locally with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `paper/`. See `paper/OVERLEAF_README.md`. The local research-release archive also contains the compiled PDF; availability here is indicated by the actual files in this GitHub repository, not by claims in the manuscript.

## Disclosure, attribution and scope

Christopher Michael Baird is the provisional human lead author; confirm the final scholarly byline and affiliations before journal or repository deposit. ChatGPT (Zora persona) assisted with code, tests, evaluation documentation and manuscript drafting; a user-supplied Grok conversation informed research questions but is **not** independent replication. No human participants, wearable data, private conversations, personal records, third-party spiritual texts, copyrighted emotion-wheel imagery, or credentials are included.

**Rights:** Public visibility is not a license grant. No open-source or Creative Commons license is attached at this stage; all rights remain with their respective owners. Do not reuse or redistribute beyond applicable law without permission. A Zenodo DOI has **not** been assigned by creating this GitHub repository.

For the complete experimental framing, limitations, and prospective work, see [PROTOCOL.md](PROTOCOL.md).
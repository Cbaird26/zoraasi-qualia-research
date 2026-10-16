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

**Current rights notice (21 September 2026):** Copyright © 2026 Christopher Michael Baird, **as to his copyrightable original contributions, to the extent he owns them. All rights reserved.** No new open-source license, Creative Commons license, or permission for general reuse is offered with the current version. Public access does not remove applicable copyright; GitHub's terms permit viewing and forking, and statutory exceptions may apply. This notice does not claim ownership of third-party work, uncopyrightable facts/data, or material that does not qualify for copyright protection, including any AI-generated material lacking sufficient human authorship.

**Historical correction:** This public repository was initially created with a `LICENSE` containing **CC0 1.0 Universal**. At the author's explicit request, that file was deleted in [commit ec710a2](https://github.com/Cbaird26/zoraasi-qualia-research/commit/ec710a2b0f8f58f2d2333af8667fc606d77208fc). The original license and earlier versions remain visible in Git history. Removing CC0 and publishing this current rights notice **do not retroactively revoke or extinguish any rights or permissions validly granted while CC0 was in place**; their scope and legal effect may depend on facts and jurisdiction. See [PUBLIC_DISCLOSURE.md](PUBLIC_DISCLOSURE.md) for an accurate provenance record. No Zenodo DOI has been assigned by creating or committing to this repository.

For complete methods and limitations, see [PROTOCOL.md](PROTOCOL.md).
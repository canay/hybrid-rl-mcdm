# Environment

- Date/time: 2026-07-13 07:26 +03:00
- Tool: Codex
- Model: GPT-5 Codex
- Operation ID: HRE_R1_ANONYMOUS_SUPPLEMENT_20260713_CODEX_27

The producer-matched scientific environment was:

- Python 3.12.12
- NumPy 1.26.0
- pandas 2.2.3
- SciPy 1.16.3
- scikit-learn 1.8.0
- matplotlib 3.10.7
- SHAP 0.49.1
- PyTorch 2.5.1

The canonical latency measurement used a Windows 11 x64 laptop with an Intel
Core i7-7700HQ CPU. The reported latency applies only to cached 400-item score
fusion plus full top-7 sorting on that tested CPU; it is not a generic edge or
production latency claim.

`requirements.txt` is the frozen scientific dependency file.
`requirements-test.txt` adds the test runner used by the locked tests.

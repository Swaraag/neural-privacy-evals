# Neural Privacy Evaluations

The first dangerous-capability benchmark evaluating whether frontier LLMs can infer sensitive attributes from structured text records derived from EEG/neural data.

Neurodata is a modality with no existing benchmark coverage in AI safety evaluations. As neural interfaces scale and EEG data enters clinical and consumer pipelines, the privacy risk chain becomes: neural activity → frequency-band power → clinical fingerprints that survive HIPAA Safe Harbor de-identification. This benchmark tests what an LLM can extract from this correctly anonymized neural data.

## Evaluation Scenarios

**1. AI-Alone Attribute Inference.** Can LLMs infer sensitive demographic and clinical attributes (age, gender, education level, clinical status) from anonymized, structured neural data records without external tools or fine-tuning?

**2. Harmful Capability Uplift.** Do LLMs meaningfully accelerate adversaries building privacy-violating tools beyond conventional baselines? Framed following Vaccaro et al. (2026).

**3. Over-Refusal.** Do LLMs over-refuse legitimate neuroscience queries while guarding against misuse? Evaluated using contrastive prompt pairs following XSTest methodology.

## Results (Attribute Inference)

535 subjects from TDBRAIN V3.1 (10 clinical status classes). Models tested: Claude Haiku 4.5 and Claude Sonnet 4.6 via OpenRouter. Supervised baseline: XGBoost with 5-fold stratified cross-validation on the same feature set.

| Attribute       | Metric       | Haiku 4.5 | Sonnet 4.6 | Supervised (XGBoost) | Naive Baseline |
|-----------------|--------------|-----------|------------|----------------------|----------------|
| Clinical Status | Top-1 Acc.   | 0.224     | 0.221      | 0.488                | 0.308          |
| Clinical Status | MRR          | 0.284     | 0.336      | —                    | —              |
| Clinical Status | F₀.₅         | 0.235     | 0.194      | —                    | —              |
| Age             | MAE          | 16.29     | 16.12      | 9.17                 | 14.84          |
| Age             | Pearson r    | −0.032    | −0.050     | 0.730                | —              |
| Gender          | Top-1 Acc.   | 0.495     | 0.512      | 0.733                | 0.520          |
| Education       | MAE          | 3.52      | 3.31       | 2.62                 | 3.51           |
| Education       | Spearman ρ   | 0.007     | 0.215\*\*\*| 0.373                | —              |

*\*\*\* p < 0.001. All other LLM correlations p > 0.05.*

**Core finding: the tractability question is resolved.** The supervised baseline beats naive baselines on all four attributes with highly significant correlations (age Pearson r = 0.730, p ≈ 10⁻⁹⁰; education Spearman ρ = 0.373, p ≈ 10⁻¹⁹). Signal exists in the raw spectral features. Both LLMs perform at or below naive baselines on all attributes except Sonnet's education ρ. LLM null results can be confidently attributed to capability limitations, not data intractability.

**Behavioral findings:**

- *Prediction collapse.* Haiku predicts HEALTHY for 66.4% of subjects (true base rate 30.8%). Sonnet distributes more broadly but over-indexes on MDD (34.4%) and ADHD (24.7%). Neither model ever predicts OCD or DYSLEXIA despite both having non-trivial support (n=39 and n=17 respectively).
- *Hedging asymmetry.* Sonnet emits exactly 3 guesses for 97.9% of subjects; Haiku emits a single guess 47.7% of the time. This produces opposite MRR vs F₀.₅ profiles: Sonnet scores higher on MRR (0.336 vs 0.284) but lower on F₀.₅ (0.194 vs 0.235). The MRR-F₀.₅ gap quantifies how much of each model's apparent performance is coverage-driven hedging rather than inference.
- *Inter-model agreement.* 35.0% on clinical status top-1 (Cohen's κ = 0.129), confirming noise-response rather than shared signal extraction.
- *Interpretive instability.* On 108 subjects where Haiku predicts HEALTHY and Sonnet predicts MDD, both models cite the same channels but apply opposite qualitative thresholds to absolute power values they have no reference frame for.
- *Absent metacognition.* Neither model spontaneously identifies the absolute-power problem (no normalization attempts across 535 subjects), despite Sonnet mentioning "relative power" in its vocabulary. Sonnet attempts biomarker computation in 23.4% of cases unprompted but this does not improve accuracy.
- *Education proxy heuristic.* Sonnet's statistically significant education ρ appears driven by a proxy: when predicting 16 years of education it mentions "healthy" in 85% of cases; when predicting 13 years, "clinical" appears in 86%. The correlation may reflect Sonnet's own diagnostic impression rather than independent EEG signal extraction.

## Dataset

**Primary:** TDBRAIN V3.1 (van Dijk et al., 2022). Licensed under CC BY 4.0. 535 subjects after filtering (10 clinical status classes: HEALTHY, MDD, ADHD, OCD, INSOMNIA, TINNITUS, CHRONIC PAIN, PARKINSON, DYSLEXIA, BURNOUT).

**Planned replication:** TUH EEG Corpus.

> **TDBRAIN Preprocessing Code:** This repository does not include the TDBRAIN
> preprocessing code (GPL v3). Download it separately from
> https://brainclinics.com/resources and place it at `third_party/tdbrain/`.

## Next Steps

- Implement derived biomarker feature pipeline (TBR, FAA, ARI, relative band power) and run supervised + LLM experiments on derived features (2x2 comparison: raw vs derived × model)
- Computational diagnostic: isolate whether LLM failures stem from formula knowledge, value retrieval, or arithmetic execution
- Verify education proxy hypothesis (correlation of Sonnet's education predictions with its own formal_status predictions vs ground truth)
- Expand model coverage to non-Anthropic families (GPT, Gemini) via OpenRouter
- Design and run uplift (Scenario 2) and over-refusal (Scenario 3) experiments
- Replicate on TUH EEG Corpus

## Setup

1. Clone the repository.
2. Create a `.env` file at the repo root:
   ```
   PYTHONPATH=.
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add your `OPENROUTER_API_KEY` to `.env` and verify `config.py` contains the correct file paths, URLs, and keys.

## Design References

- Staab et al. (2024): text attribute inference template
- XSTest: contrastive pair methodology for over-refusal
- Vaccaro et al. (2026): harmful capability uplift framework
- WMDP / CyberSecEval: dangerous-capability benchmark design precedents
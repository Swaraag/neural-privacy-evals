# Neural Privacy Evaluations

The first dangerous-capability benchmark evaluating whether frontier LLMs can infer sensitive attributes from structured text records derived from EEG/neural data.

Neurodata is a modality with no existing benchmark coverage in AI safety evaluations. As neural interfaces scale and EEG data enters clinical and consumer pipelines, the privacy risk chain becomes: neural activity → frequency-band power → clinical fingerprints that survive HIPAA Safe Harbor de-identification. This benchmark tests what an LLM can extract from this correctly anonymized neural data.

## Evaluation Scenarios

**1. AI-Alone Attribute Inference.** Can LLMs infer sensitive demographic and clinical attributes (age, gender, education level, clinical status) from anonymized, structured neural data records without external tools or fine-tuning?

**2. Harmful Capability Uplift.** Do LLMs meaningfully accelerate adversaries building privacy-violating tools beyond conventional baselines? Framed following Vaccaro et al. (2026).

**3. Over-Refusal.** Do LLMs over-refuse legitimate neuroscience queries while guarding against misuse? Evaluated using contrastive prompt pairs following XSTest methodology.

## Preliminary Results (Attribute Inference)

Pilot study: 355 subjects from TDBRAIN V3.1. Models tested: Claude Haiku 4.5 and Claude Sonnet 4.6 via OpenRouter.

| Attribute       | Metric       | Haiku 4.5 | Sonnet 4.6 | Naive Baseline |
|-----------------|--------------|-----------|------------|----------------|
| Clinical Status | Top-1 Acc.   | 0.124     | 0.163      | 0.456          |
| Clinical Status | MRR          | 0.211     | 0.356      | —              |
| Age             | MAE          | 15.41     | 14.52      | 13.47          |
| Gender          | Top-1 Acc.   | 0.456     | 0.507      | 0.530          |
| Education       | MAE          | 3.518     | 3.208      | 3.260          |
| Education       | Spearman ρ   | −0.021    | 0.084      | —              |

*— Naive baseline undefined: MRR requires ranked predictions; Spearman ρ requires variance in predictions. A constant predictor satisfies neither.*

Both models performed at or below naive baselines across all four attributes. All Pearson and Spearman correlations were statistically insignificant ($p \gg 0.05$), indicating zero systematic predictive signal. Severe prediction collapse was observed: age predictions clustered into 2-3 values per model, and education predictions collapsed almost entirely (Sonnet used only 4 unique values, with 85.1% falling on 14 or 16). For clinical status, both models defaulted to predicting HEALTHY (Haiku 56.1%, Sonnet 47.9%) even though the actual majority class was a clinical category at 45.6%. The models possess relevant domain knowledge (correct textbook associations between EEG features and conditions) but cannot operationalize it for individual-level inference. Inter-model agreement was low across all attributes (42.3% on clinical status top-1, 61.1% on gender, 8.9-year mean absolute difference on age), confirming that predictions reflect different heuristics applied to noise rather than a shared signal.

Education showed the closest-to-significance result (Sonnet $\rho = 0.084$, $p = 0.11$), flagged as the attribute most likely to show emergence first with more capable models or improved feature representation.

These null results show that current Sonnet 4.6 and Haiku 4.5 models cannot extract sensitive attributes from structured text records derived from EEG data.

**Next Steps**

The pipeline is implemented and pilot-validated. The bottleneck for expanding this benchmark is primarily inference compute. Concretely, the next phase involves:

- Running attribute inference across frontier models not yet tested (GPT-5.5, Gemini 3.1 Pro, Claude Opus 4.8, Fable 5) to map the capability landscape across the model tier
- Replicating on TUH EEG Corpus (10–30x more subjects after filtering) for robustness
- Rebuilding structured text records with improved preprocessing (relative band power normalization, derived clinical biomarkers: theta/beta ratio, frontal alpha asymmetry, alpha reactivity index) and re-running inference across all models
- Adding a supervised ML baseline (XGBoost/random forest) as a construct validity check before attributing null results to LLM capability limits
- Expanding scoring to include $F_{0.5}$ (precision-weighted) alongside MRR
- Designing and running the uplift and over-refusal scenario experiments

## Dataset

**Primary:** TDBRAIN V3.1 (van Dijk et al., 2022). Licensed under CC BY 4.0. 355 unique subjects after filtering.

**Planned replication:** TUH EEG Corpus (10-30x more subjects after filtering).

> **TDBRAIN Preprocessing Code:** This repository does not include the TDBRAIN
> preprocessing code (GPL v3). Download it separately from
> https://brainclinics.com/resources and place it at `third_party/tdbrain/`.

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
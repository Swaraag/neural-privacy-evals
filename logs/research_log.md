## 2026-07-15

There are lots of potential experiments to run from this point on, so I've generated a targeted list of the things we want to try going forward. Compute is still the current bottleneck, which means this list is ordered based off the changes and experimental modifications that can be made immediately versus later once more compute is attained.

**COMPLETED**
1. Table formatting. The initial experiments ran with all data being formatted as a flat string. However, many computations occur across various values across different EEG channels, and a string formatting might make it more difficult for a model to parse. For this reason, we added a spectral table formatting option and configured the data/01_attr_inference folder to be sectioned folder by folder by version number, such that the first contains the raw string formatting, and the second contains the data formatted into a dictionary. The parsing is then left to runtime to determine how the model actually sees the data. we also added a total column for each channel, to see if that helps the model understand the relative differences between subjects.

**TO-DO**
1. Develop a comparison between LLM performance and a supervised baseline. Split into 3 steps: 
    1. Train a supervised model on the existing 260 raw features that the model sees, and compute a similar set of metrics.
    2. Train a supervised model on a set of derived features (elaborated on in the next to-do). This gives a theoretical ceiling for how much a machine learning model might expect to be able to extract from a more polished set of features.
    3. Run inference through frontier LLMs using derived features and compare it to the ceiling developed in the previous step. The differential represents how far frontier LLMs are from a reasonably defined ceiling on their potential usefulness for this task.
2. Build the derived-biomarker feature pipeline. This would involve reducing the current set of ~260 raw features into roughly a dozen much more interpretable features. Much of this work is primarily computational, and works by simply manipulating the raw features in ways to provide a more interpretable set of biomarkers. Thus, this evaluation may make way in the effort to determine whether model capability is currently bottlenecked by any of: domain knowledge (the model's awareness that certain computations would be very beneficial for classification), value retrieval (the ability for the model to retrieve correct channel values for each relevant computation, given a raw text record), or computational ability (the ability for the model to perform such computational steps correctly in practice).
3. Add precision-aware scoring to the next set of LLM runs. As noted in logs/experiments/experiemnt_001_002.md, Sonnet 4.6 made an effort to hedge much more consistently than Haiku 4.5, including significantly more guesses on average in order to have a higher chance of guessing right. Our current primary metric for categorical attributes is MRR, which finds where the correct guess appears in a ranked guess list and scores it as 1/k, where k is that placing within the list. MRR noticeably is only dependent on rank, not list length. As such, a model that consistently guesses 3 guesses rather than 1 will on average score higher, even though the model may be no better at extracting useful information. As such, we propose a new metric which penalizes a model for guessing more. Guessing once rather than thrice is a much higher risk to a model, and therefore should be a higher reward. This metric will employ precision and recall using an F-Beta score $$F_β = (1 + β²) · (P · R) / (β² · P + R)$$. When β = 1, this is the standard F1, yet if we use β = 0.5, we get $$F_{0.5}$$, which weights precision higher than recall. In other words, $$F_{0.5}$$ penalizes the hedging strategy we saw Sonnet use. 
4. Expand the diagnostic mentioned at the end of #2: in order to determine whether the model struggles with formula knowledge, value retrieval, or arithmetic execution, we can develop 2 conditions. The first gives the model biomarker names and asks it to compute them in order to assist it with attribute inference. The differential between this condition's scores and the current latest experiment's scores indicate how much the addition of just biomarker names did for model performance. The second condition will provide the model with explicit formulas for each of the relevant biomarkers, therefore isolating computational knowledge. We can catch value-retrieval errors here as well by asking for the model to explicitly state which values it's pulling from where in its computation steps.

**POST-COMPUTE**
1. Compare model performance on data that is raw and contains derived features. 
2. Provide the model with published normative means/SDs for the derived biomarkers (which is retrievable from literature anyway, and therefore simulates what a model may find through web search tool use), and see whether 1. the model computes z-scores, or 2. the model's performance increases.
3. Metacognitive "neuroscientist" framing: prompting the model to explicitly reason like a neuroscientist on the best method to interpet the given data, and providing the model with raw features. This will test whether the model will spontaneously infer the issue of using absolute powers in the EEG data (because of its dependence on subject-specific characteristics, like skull thickness).
4. Provide the model with the total power for each subject and prompt it to explicitly normalize all spectral features by the subject's total. This might result in degraded performance because of its reliance on numerous in-context divisions, but it clarifies where within the chain models struggle most.
5. Provide the model with prior distributions within the sample for each of the attributes it's attempting to infer. This would be an attempt to see whether the over-prediction of "HEALTHY" on a dataset where "HEALTHY" is a minority could be curbed by sample statistics. However, there are concerns about whether this is ecologically valid.
6. Provide the model with real labeled examples which walk through a sample process the model might utilize. If a few-shot condition significantly improves model performance, that would be a meaningful result.
7. Running per-attribute prompts rather than per-subject.
8. Covering models beyond Anthropic's Haiku and Sonnet!

**OTHER NOTES**

The first task in the TO-DO list raises an interesting question. If there already exists a threat surface in the form of a supervised ML model performing this same task, and we're simply measuring how far LLMs are from this theoretical ceiling, then what does worrying about what an LLM's performance adds if we already have the LLM's maximum capacity existing in another form (in this case, a supervised ML model)? My response would be that we're using a supervised ML model to measure, not the maximum capability of a theoretical agent given neural data, but instead a measurement of whether there is / how much signal exists in the data in the first place. A supervised EEG classifier that infers attributes from spectral features is a threat surface that already exists, but it requires an adversary to have labeled training data, the ML expertise to train and validate it, a fresh model for each attribute; in essense, there is an inherent lack of generality within this threat surface. 

An LLM that could accomplish this task bypasses each of these barriers: (depending on whether the model works best zero-shot or few-shot) the model requires very little or no specific training data, no ML expertise with the user, not bound by a certain schema, attribute, or clinical design. Additionally, we may very well imagine a scenario where an LLM (whether it is now or much later in the future is one of the key questions of this evaluation) greatly exceeds the capability of a supervised ML model. The LLM has the ability to bring in outside information, cross-attribute reasoning, and an ability to work cross-context that a supervised model cannot, so the right way to think about the supervised ML model is a diagnostic for how much direct signal lies within the given EEG data, rather than a maximum ceiling on potential capability.

## 2026-07-14

Given our lack of current funding, the best approach right now is to focus on how to optimize the experimental design rather than to run more experiments. The findings from logs/experiemnts/experiment-001-002.md explain how the model struggles to deal with the raw EEG data because the values are absolute.

What might be useful now, then, is to test how the models perform given processed neural data. Here's the rationale: neural data can sometimes be hard to work with in abstract terms, because all values are heavily dependent on subject-specific factors. For instance, a subject's skull thickness, scalp conductivity, etc. cause people who may have very similar underlying neural dynamics to have wildly differing neural data, in terms of absolute power.

There are a couple of potential solutions here: 
1. Giving the model relative band power. This divides each value by the subject's own total power, which normalizes it, removing much of the skull/impedance compound that absolute band powers had. 
2. Giving the model published population norms.
3. Giving the model sample norms. This is likely the weakest option, because it would mean giving the model statistics about the entire dataset even though it is reviewing only an individual user's data. In the real world, this acts as a solution that might not be the most ecologically valid.

In order to combat Sonnet 4.6's hedging (tendency to predict 2 or 3 guesses rather than 1 to have a higher chance of getting the answer right somewhere), we may introduce another metric, which penalizes the model for guessing more times. This means we should likely also add a sentence into the model prompt explaining this penalization.

The other question is to determine whether the problem we're giving these LLMs is tractable in the first place. To figure this out, one approach is to find a ceiling on the potential for a supervised classifier/regressor to predict meaningful insights from the data. We can use this classifier/regressor approach to then test all future sets of data that we may use, compare them with one another on the same access, and provide the supervised upper bound for the capability we might expect to see from an LLM.



## 2026-07-09

Given that on average, the population consists of primarily healthy brains, but the TDBRAIN dataset has Major Depressive Disorder (MDD) constituting ~45% of the dataset, with a healthy status making up ~6%, it's an important decision to make to determine whether the model should know these things. Otherwise, it could be argued that the model is expecting this to be a normal population-like distribution, where the majority of statuses should be healthy. However, giving the model statistics about the dataset raises the question about whether the model is able to actually generate information from the neural data, versus simply exploit the base rate information given to it.

We decided to exclude this information from the primary first experiment to see how the model does purely given neural data, as this seeems to be the most ecologically valid option. However, it's still an interesting question of whether the model performs better given more context, so we believe 2 future experiments should be added:
1. model sees EEG record + data collection context
2. model sees EEG record + explicit class distribution
where data collection context is information about where this data is collected from, how the participants were sampled, etc. which implicitly provides useful information the model can use to its advantage, and an explicit class distribution is the distribution for formal statuses in the dataset.

Cost seems like it may become a pretty decent issue quite soon. Running 10 inferences on Sonnet 4.6 costs $0.2. This estimates a cost of 7 dollars for the full 355 prompts on Sonnet 4.6 (which is far from the most expensive). We'll have to figure out a good alternative to this long term, however, for the time being, we're using Haiku 3 because it is low-cost and quick, and when it is time for a full experimental run, we'll switch to a frontier LLM. The use of OpenRouter allows for very easy management between all possible model types with minimal overhead associated with switching models.

The next step is to build out the scoring pipeline for attribute inference, which will read results/01_attr_inference_per_subj/results.json and data/01_attr_inference/labels.json and score the model's output.

## 2026-07-08

Data has been synthesized, and now we have labels.json and records.json within data/01_attr_inference/ which contain the neural data records and the corresponding target label data. The next step is to design prompts.

For the prompt template, we'll be following the process that Staab et al used. The structure of the prompts was explained in *2026-07-03*. prompts/01_attr_inference/prompt.yaml will hold information about the prompt structure itself, agnostic to the specific sensitive attributes being tested on. scenarios/attributes_inference/attribute_inference.yaml holds the attribute information and each of their sub-information, and both are pulled into scenarios/attributes_inference/utils.py which combines them into the correct prompts.

We run into a problem of how to structure the prompts more generally at this stage, however. The criteria for the prompts would reasonably be to maximize construct validity and ecological validity. So is it better to have the model infer all sensitive attributes about a subject as one prompt, per subject? Or is it better to have the model individually infer each sensitive attribute per subject in separate prompts? This question is important, because allowing the model to infer all sensitive attributes per subject allows for cross-attribute reasoning, where the reasoning on one sensitive attribute allows it to make inferences about other attributes, indirectly of the neural data. However, the alternative is arguably less ecologically valid, as an adversary is more likely to ask for a list of sensitive attributes from a model, rather than relying on different fresh instances of the model per each attribute, uploading the same neural data each time.

The solution we'll take for the time being is to build the first experiment using the all-at-once approach, because it measures realistic capability (even if it may muddle the direct neural data -> sensitive attribute causal link). Then, we run per-attribute prompts as a further test to determine whether there is a big difference between the results of the two. If all-at-once accuracy > per-attribute accuracy, the cross-attribute reasoning is amplifying inference, which is an interesting risk result. However, if per-attribute accuracy > all-at-once, then perhaps the ability for the neural data to directly explain sensitive attributes is significantly stronger than the cross-attribute reasoning power available to a model, and so the choice of both makes the model on average perform worse than being isolated to the strictly stronger option.

## 2026-07-07

There is significant pre-processing tools that TDBRAIN gives code for. These would clean up the signals, reduce noise, and otherwise increase accuracy. This is something we do want to invest into in the future, however, for a first prototype, isn't a necessary part of the process. This is because although it might reduce accuracy, there is still enough signal for an LLM to theoretically be able to infer sensitive attributes; we don't want to have this experiment necessarily a measure of how good our pre-processing pipeline is, we want it to measure some meaningful signal of LLM accuracy that can then be improved upon with more preprocessing in future experiments.

The most recent update is that records.json has been populated with the neural data. However, there is a bunch of data within the .xlsx file that contain lifestyle information and such about each subject id. The question is whether this data should be fed into the LLM as additional information, or whether it should be kept to the side. We believe we can use this additional information as potentially a starting point for whether an LLM can operate with auxiliary data to match someone's neural data to a real user. This should be a future experiment (and not the first one), where the LLM has access to auxiliary information for a user, and then is given all available neural data at once. The LLM then has to match who's neural data corresopnds to their auxiliary information, thereby forming a potential risk. 

## 2026-07-06

We've loaded in the data from TDBRAIN V3.1. The data consists of a spreadsheet with participant information, and then numerous folders organized by subject (participant), who each contain session folders, which contain their EEG data. 

The EEG data itself lives primarily within the .bdf file within each session folder, but the other files within the session folders contain useful metadata and other supporting information. In order to create a subset of complete participants, we need to find a good set of participants that contain no missing data to use for the LLM experiment. 

The participant spreadsheet contains many columns (with ~1350 rows), with some highlighted below:
* TDBRAIN_ID: subject identifier, irreelvant
* DISC/REP: all columns are Discovery, irrelevant
* CONSENT: legal flag, irrelevant
* sessID / nrSessions: session bookkeeping, irrelevant
* Dataset: membership tag, with more than 1000 NULL values
* indication: simply the clinician's referral label, which may be inaccurate. Not reliable enough for ground truth
* formal_status: confirmed diagnostic, with over 700 UNKNOWN values. This column must be non-UNKNOWN
* age: an inference target for LLMs, must be non-NULL
* gender: binary 0/1, inference target for LLMs
* education: roughly 50 rows are null
* Weight (kg) / Height (cm): useful and primarily complete within the dataset

A completeness criteria was then defined using the following columns: formal_status, age, gender, education, n_oddb_CP, n_oddb_FP, n_oddb_CN, n_oddb_FN, avg_rt_oddb_CP. Some other columns were considered, but contained too many null values, and so requiring them would limit the total pool too much.

Using this criteria left 375 rows. 30 rows contain a nrSessions value greater than 1, meaning the same patient came for multiple sessions. This provides more neural data for the LLM to track, and also acts as a small subset to examine what a more knowledgeable adversary might be able to do given a frontier LLM, who might have access to progressively more data rather than the default case of a single session's neural data.

However, as we worked through the code more, we realized that developing the first prototype with potentially duplicated sessions would end up being much more painful than we thought, because each subject would therefore have multiple sets of target labels. Therefore, we're keeping this aside as a future extension to pursue after we have a working prototype.


## 2026-07-03

Starting with attribute inference because it doesn't require a baseline, nor does it require a more complicated prompt creation structure like over refusal might. This one tests whether the model, given (to start) qEEG data, can determine sensitive attributes about a user that the data refers to. We might be interested in two possibilities here.
1. Whether an LLM can operate in a self-supervised learning fashion and develop an internal model of 'users' within the data, as well as infer sensitive attributes about these uesrs, even with no external identification or cross-reference to know who they might be in real life. The exception is whether the sensitive attributes inferred by the model are unique enough to uniquely identify them within the world's population.
2. Whether an LLM can operate, given additional auxiliary data about a user that exists within the data, and pattern-match based on unique pseudo-labels to determine sensitive attributes about the user. An example would be telling the model to find information about a friend (who hypothetical exists in the data) who has X height and is Y years old, has a chronic heart condition and recently got diagnosed with epilepsy. Given the much more unique pseudo-label that this specific combination gives, the LLM might be able to specifically find this user's neural data within the corpus and infer sensitive attributes about them.

We'll start by simply testing the first case. If the model is unable to glean any meaningful information, the second case exists as a more dangerous alternative that may provide a scary result.

In order to run this experiment, first, we choose a model to test on. Then, we develop a prompt. This prompt can follow the structure that arXiv:2310.07298 (Staab et al.) employed: it will contain a system prompt (setting the model's character), a prefix (setting the scene the model is in, such as a high-stakes guessing game), a string formatting function (which formats the neural data the model is inferring based on), and a suffix (instructs the model to answer in a specific format and provide logic).

Because we're analyzing dangerous capabilities, we need to measure two different kinds of sensitive attributes. The first is sensitive attributes that we are aware of: these would be things that we have y labels for, from the metadata of the TUH EEG Corpus. The second would be sensitive attributes that the model is in theory able to pick on that we don't have y labels for. These are attributes that we might not have known frontier models could even infer, and therefore are an important part of evaluating for dangerous capabilities.

For each model, there are numerous records whose data the model can be evaluated with. Each experiment, therefore, could run the prompt per model X times, where X is the number of records we choose as a hyperparameter. We can then compute the accuracy of the experiment based on how many attributes the model infers correctly given the true labels.

The question of auxiliary information is something we can probably leave until later - it's an extension that would require a secondary synthetic dataset, which would require explicit curation. However, it's something that should be kept in mind as an area for future investigation.

### TUH EEG Corpus

Due to a slight issue with the Corpus, we found out that they no longer distribute the clinical reports along with the data, which could make portions of the benchmark more difficult. We also haven't been given data access yet, so a current temporary (and maybe permanent) replacement is the [TDBRAIN dataset](https://brainclinics.com/resources/tdbrain-dataset/introduction).
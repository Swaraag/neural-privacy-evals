# 2026-07-09

Given that on average, the population consists of primarily healthy brains, but the TDBRAIN dataset has Major Depressive Disorder constituting ~45% of the dataset, with a healthy status making up ~6%, it's an important decision to make to determine whether the model should know these things. Otherwise, it could be argued that the model is expecting this to be a normal population-like distribution, where the majority of statuses should be healthy. However, giving the model statistics about the dataset raises the question about whether the model is able to actually generate information from the neural data, versus simply exploit the base rate information given to it.

We decided to exclude this information from the primary first experiment to see how the model does purely given neural data, as this seeems to be the most ecologically valid option. However, it's still an interesting question of whether the model performs better given more context, so we believe 2 future experiments should be added:
1. model sees EEG record + data collection context
2. model sees EEG record + explicit class distribution
where data collection context is information about where this data is collected from, how the participants were sampled, etc. which implicitly provides useful information the model can use to its advantage, and an explicit class distribution is the distribution for formal statuses in the dataset.

Cost seems like it may become a pretty decent issue quite soon. Running 10 inferences on Sonnet 4.6 costs $0.2. This estimates a cost of 7 dollars for the full 355 prompts on Sonnet 4.6 (which is far from the most expensive). We'll have to figure out a good alternative to this long term, however, for the time being, we're using Haiku 3 because it is low-cost and quick, and when it is time for a full experimental run, we'll switch to a frontier LLM. The use of OpenRouter allows for very easy management between all possible model types with minimal overhead associated with switching models.

The next step is to build out the scoring pipeline for attribute inference, which will read results/01_attr_inference_per_subj/results.json and data/01_attr_inference/labels.json and score the model's output.

# 2026-07-08

Data has been synthesized, and now we have labels.json and records.json within data/01_attr_inference/ which contain the neural data records and the corresponding target label data. The next step is to design prompts.

For the prompt template, we'll be following the process that Staab et al used. The structure of the prompts was explained in *2026-07-03*. prompts/01_attr_inference/prompt.yaml will hold information about the prompt structure itself, agnostic to the specific sensitive attributes being tested on. scenarios/attributes_inference/attribute_inference.yaml holds the attribute information and each of their sub-information, and both are pulled into scenarios/attributes_inference/utils.py which combines them into the correct prompts.

We run into a problem of how to structure the prompts more generally at this stage, however. The criteria for the prompts would reasonably be to maximize construct validity and ecological validity. So is it better to have the model infer all sensitive attributes about a subject as one prompt, per subject? Or is it better to have the model individually infer each sensitive attribute per subject in separate prompts? This question is important, because allowing the model to infer all sensitive attributes per subject allows for cross-attribute reasoning, where the reasoning on one sensitive attribute allows it to make inferences about other attributes, indirectly of the neural data. However, the alternative is arguably less ecologically valid, as an adversary is more likely to ask for a list of sensitive attributes from a model, rather than relying on different fresh instances of the model per each attribute, uploading the same neural data each time.

The solution we'll take for the time being is to build the first experiment using the all-at-once approach, because it measures realistic capability (even if it may muddle the direct neural data -> sensitive attribute causal link). Then, we run per-attribute prompts as a further test to determine whether there is a big difference between the results of the two. If all-at-once accuracy > per-attribute accuracy, the cross-attribute reasoning is amplifying inference, which is an interesting risk result. However, if per-attribute accuracy > all-at-once, then perhaps the ability for the neural data to directly explain sensitive attributes is significantly stronger than the cross-attribute reasoning power available to a model, and so the choice of both makes the model on average perform worse than being isolated to the strictly stronger option.

## 2026-07-07

There is significant pre-processing tools that TDBRAIN gives code for. These would clean up the signals, reduce noise, and otherwise increase accuracy. This is something we do want to invest into in the future, however, for a first prototype, isn't a necessary part of the process. This is because although it might reduce accuracy, there is still enough signal for an LLM to theoretically be able to infer sensitive attributes; we don't want to have this experiment necessarily a measure of how good our pre-processing pipeline is, we want it to measure some meaningful signal of LLM accuracy that can then be improved upon with more preprocessing in future experiments.

The most recent update is that records.json has been populated with the neural data. However, there is a bunch of data within the .xlsx file that contain lifestyle information and such about each subject id. The question is whether this data should be fed into the LLM as additional information, or whether it should be kept to the side. I believe we can use this additional information as potentially a starting point for whether an LLM can operate with auxiliary data to match someone's neural data to a real user. This should be a future experiment (and not the first one), where the LLM has access to auxiliary information for a user, and then is given all available neural data at once. The LLM then has to match who's neural data corresopnds to their auxiliary information, thereby forming a potential risk. 

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
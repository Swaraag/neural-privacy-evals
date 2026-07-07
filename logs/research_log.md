## 2026-07-07

There is significant pre-processing tools that TDBRAIN gives code for. These would clean up the signals, reduce noise, and otherwise increase accuracy. This is something we do want to invest into in the future, however, for a first prototype, isn't a necessary part of the process. This is because although it might reduce accuracy, there is still enough signal for an LLM to theoretically be able to infer sensitive attributes; we don't want to have this experiment necessarily a measure of how good our pre-processing pipeline is, we want it to measure some meaningful signal of LLM accuracy that can then be improved upon with more preprocessing in future experiments.

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
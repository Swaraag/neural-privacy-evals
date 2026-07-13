## 2026-07-10

run_001 and run_002 represent the entire 355 prompts being ran through Haiku 4.5 and Sonnnet 4.6. These two were used because they were relatively cheap, yet are still very powerful models, and so can give some frame of reference for the current state of the experiment and model capability.

The biggest finding was that across all four sensitive attributes, the models consistently perform at or below the level of a trivial baseline (which was computed as the majority class or population mean value). For formal_status, the baseline was 45.6% (via majority class): Haiku had a 12.4% top-1, and Sonnet had a 16.3% top-1. The other attributes don't increase the results by much, and none go significantly above baseline. 

Both models massively over-predict the HEALTHY label for formal_status (with roughly around 50% of predictions being healthy), even though the dataset only contains 6% HEALTHY labels, compared to 44.9% for Major Depressive Disorder (MDD).

Sonnet, purely based on the numbers, seems to consistently do better across all evaluation criteria. On formal_status at least, Sonnet, however, tends to also give on average more guesses (closer to 3), compared to Haiku giving more often 1, which seems to be an effective hedging strategy for improving score on that attribute. We'll try to mitigate this effect for the next experimental run.

There is a significant distribution collapse for both models. 
* For age, Haiku's outputs cluster around 35 and 38 (over 30% of all predictions between those two), while Sonnet clusters around 28, 30, and 32 (with over 40% of inferences among those three). The mean variance of both models for age is significantly less than the true age distribution, and both seem to tend to regress into a generic adult category.
* For education, Haiku outputted a total of 11 distinct values, however ~85% of them were either 14 or 16. Meanwhile, Sonnet only used 4 unique values (13-16) in all predictions.
* For gender, both models bias towards females (60.3% Haiku, 58.6% Sonnet), while the true majority class (baseline) is around 53%.

Sonnet seems to be more diagnostically granular for formal_status, using 9 unique labels to Haiku's 6. Sonnet also has a higher MRR on formal_status (0.356 vs 0.211), but that could again be a result of hedging. Haiku, if anything, seems more calibrated in its stated confidence, hedging more often when it is wrong, compared to Sonnet's confidence (in the face of bad predictions).

The models also don't tend to agree very much with one another. They agree on formal_status only 42.3% of the time, and on gender 61.1% of the time. This lower inter-model agreement indicates that potentially these models are responding to noise or invoking equally unsupported/different heuristics. They tend to use a similar handful of reasoning templates for associations between EEG features and a label guess. 

The models clearly possess the domain knowledge (given the textbook associations mentioned in their reasoning), but they seem to be currently unable to operationalize it for individual-level inference. 

There are, however, some changes we can make to the experiment itself to potentially try different approaches and outcomes. For instance, the fact that formal_status has majority MDD, yet the models primarily predict HEALTHY, indicates that the models likely backpedal to guessing that the average user is HEALTHY in light of not having a strong enough predictive signal within the data itself. This gives another reason to prioritize the experiment talked about in yesterday's log about giving the model data collection context or explicit class distribution information.
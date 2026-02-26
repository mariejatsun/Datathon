# Datathon
### Data used
Data used is the Spaced Repetition dataset from Duolingo. To run the code, the csv file should be included in the folder. The data is not included in the repository, but is available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME

### Project Overview
This repository builds a personalized Duolingo Wrapped-style profile using spaced repetition data.
The pipeline consists of four analytical components and one visualization module.

#### 1. Learning Profile (Memory Parameters)
*Betas.py*
- Estimates per-user memory model parameters using linear regression (log(precall​)=β0​+β1​log(δ)+β2​log(history_seen), formula based on:https://github.com/duolingo/halflife-regression/blob/master/settles.acl16.pdf):
    - β0 → baseline recall
    - β1 → forgetting rate (time decay)
    - β2 → learning effect (practice effect)
- output: betas_df_eval.csv (Contains betas, percentiles, R², p-values, and model diagnostics per user)

*learningprofile1.py*
- Clusters users (KMeans, k=3) based on β parameters.
- Profiles
   - Repetition Builder
   - steady retainer
   - fast burner
- Includes silhouette validation.
- Output: *user_cluster_profiles.csv*

#### 2. Cognitive Focus (Vocabulary vs Grammar)
*word_classifier.py*
- Classifies lexemes into:
    - Word class (Verb, Noun, etc.)
    - Vocabulary vs Grammar
- Aggregates per-user statistics and determines dominant focus.
- Output: *user_profiles.csv*

#### 3. Learning Age
*learning_age.py *
- Computes a composite performance score and maps it to a Learning Age (18–80) using exponential scaling.
- Includes bootstrap confidence intervals and age brackets
- Output: *user_learning_age.csv*

#### 4. Wrapped Interface (Visualization)
*figma.py*
- Generates a 5-slide personalized Wrapped experience:
    1. Intro
    2. Memory percentiles (based on percentiles beta1 and beta2)
    3. Vocabulary/Grammar focus
    4. Learning Age
    5. Cognitive cluster type
- Inputs: *betas_df_eval.csv, user_profiles.csv, user_learning_age.csv, user_cluster_profiles.csv*
- Output:*app_slides_update/slides.gif*
  

### FINAL RESULTS: Each user receives:
- Memory performance percentiles
- Cognitive learning type
- Vocabulary vs Grammar preference
- Learning Age
- Animated Wrapped-style summary GIF

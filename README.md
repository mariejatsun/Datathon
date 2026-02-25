# Datathon
### Project Overview
This project aims to identify 3 profiles for each Duolingo user, and presents them in a Duolingo Wrapped styled manner

### Data used
Data used is the Spaced Repetition dataset from Duolingo. To run the code, the csv file should be included in the folder. The data is not included in the repository, but is available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME

### Repository contents
- Learning profile/
    - betas.py extracts parameters and outputs betas_df_eval.csv
    - learningprofile1.py uses these parameters to cluster into 3 distinct profiles, outputs user_cluster_profiles

- Cognitive focus/
    - word_classifier.py classifies lexemes and give vocab/grammar preference for each user, outputs user_profiles.csv

- Learning age/
    - learning_age.py computes learning age and outputs user_learning_age.csv

- Wrapped/
    - figma.py builds dynamic interface for presenting profiles in Wrapped styled manner
    - app_slides and app_slides_update contain backbone of interface design
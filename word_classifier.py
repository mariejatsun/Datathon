"""
Duolingo Spaced Repetition Dataset — Word Classification & User Profile Pipeline
=================================================================================
Classifies each lexeme_string into:
  - word_class   : Verb | Noun | Adjective | Pronoun | Determiner |
                   Prep/Conj | Adverb | Other
  - vocab_grammar: Vocabulary | Grammar | Other

Then aggregates per user into a profile DataFrame ready for downstream analysis.
"""

import time
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1.  CLASSIFICATION MAPS
# ---------------------------------------------------------------------------

POS_TO_CLASS = {
    'vblex':   'Verb',
    'vbser':   'Verb',
    'vbhaver': 'Verb',
    'vbdo':    'Verb',
    'vbmod':   'Verb',
    'vaux':    'Verb',
    'n':       'Noun',
    'np':      'Noun',
    'adj':     'Adjective',
    'prn':     'Pronoun',
    'prpers':  'Pronoun',
    'det':     'Determiner',
    'pr':      'Prep/Conj',
    'cnjcoo':  'Prep/Conj',
    'cnjsub':  'Prep/Conj',
    'cnjadv':  'Prep/Conj',
    'adv':     'Adverb',
    'preadv':  'Adverb',
    'num':     'Other',
    'ij':      'Other',
    'sent':    'Other',
}

# Vocabulary = open-class words. Grammar = closed-class / functional words.
CLASS_TO_VOCABGRAMMAR = {
    'Verb':       'Vocabulary',
    'Noun':       'Vocabulary',
    'Adjective':  'Vocabulary',
    'Adverb':     'Vocabulary',
    'Pronoun':    'Grammar',
    'Determiner': 'Grammar',
    'Prep/Conj':  'Grammar',
    'Other':      'Other',
}


# ---------------------------------------------------------------------------
# 2.  CLASSIFICATION  (fully vectorised)
# ---------------------------------------------------------------------------

def add_classification_columns(df, lexeme_col='lexeme_string'):
    """
    Adds 'pos_raw', 'word_class', and 'vocab_grammar' columns.
    Fully vectorised — no Python-level row iteration.
    """
    df = df.copy()

    # Extract POS: pull content inside [...], take first tag before '.'
    df['pos_raw'] = (df[lexeme_col]
                     .str.extract(r'\[([^\]]+)\]')[0]
                     .str.split('.').str[0]
                     .fillna('unknown'))

    # .map does C-level dict lookup across the whole column at once
    df['word_class']    = df['pos_raw'].map(POS_TO_CLASS).fillna('Other')
    df['vocab_grammar'] = df['word_class'].map(CLASS_TO_VOCABGRAMMAR).fillna('Other')

    return df


# ---------------------------------------------------------------------------
# 3.  VOCAB/GRAMMAR LABEL HELPER
# ---------------------------------------------------------------------------

def _dominant_vg_label(ratio, vocab_threshold=0.65, grammar_threshold=0.35):
    if pd.isna(ratio):
        return 'Unknown'
    if ratio >= vocab_threshold:
        return 'Vocabulary'
    elif ratio <= grammar_threshold:
        return 'Grammar'
    else:
        return 'Balanced'


# ---------------------------------------------------------------------------
# 4.  USER PROFILE BUILDER  (fully vectorised)
# ---------------------------------------------------------------------------

def build_user_profiles(df, user_col='user_id', recall_col='p_recall'):
    assert user_col in df.columns, f"Column '{user_col}' not found."
    missing = {'word_class', 'vocab_grammar'} - set(df.columns)
    assert not missing, f"Run add_classification_columns() first. Missing: {missing}"

    has_recall = recall_col in df.columns

    word_classes = ['Verb', 'Noun', 'Adjective', 'Pronoun',
                    'Determiner', 'Prep/Conj', 'Adverb', 'Other']
    vg_classes   = ['Vocabulary', 'Grammar', 'Other']

    df = df.copy()

    # Vectorised lemma extraction
    df['_lemma'] = (df['lexeme_string']
                    .str.split('/', n=1).str[1]
                    .str.split('[', n=1).str[0])

    # Counts & proportions
    total_obs = df.groupby(user_col).size().rename('total_observations')

    wc_counts = (df.pivot_table(index=user_col, columns='word_class',
                                aggfunc='size', fill_value=0)
                   .reindex(columns=word_classes, fill_value=0))
    wc_counts.columns = [f'n_{c.replace("/", "_")}' for c in word_classes]

    wc_pct = wc_counts.div(total_obs, axis=0)
    wc_pct.columns = [c.replace('n_', 'pct_') for c in wc_counts.columns]

    vg_counts = (df.pivot_table(index=user_col, columns='vocab_grammar',
                                aggfunc='size', fill_value=0)
                   .reindex(columns=vg_classes, fill_value=0))
    vg_counts.columns = [f'n_{c.lower()}' for c in vg_classes]

    vg_pct = vg_counts.div(total_obs, axis=0)
    vg_pct.columns = [c.replace('n_', 'pct_') for c in vg_counts.columns]

    # Recall
    if has_recall:
        wc_recall = (df.groupby([user_col, 'word_class'])[recall_col]
                       .mean().unstack('word_class')
                       .reindex(columns=word_classes))
        wc_recall.columns = [f'recall_{c.replace("/", "_")}' for c in word_classes]

        vg_recall = (df.groupby([user_col, 'vocab_grammar'])[recall_col]
                       .mean().unstack('vocab_grammar')
                       .reindex(columns=vg_classes))
        vg_recall.columns = [f'recall_{c.lower()}' for c in vg_classes]

        overall_recall = df.groupby(user_col)[recall_col].mean().rename('recall_overall')

    # Unique lemmas
    lemma_counts = (df.groupby([user_col, 'word_class'])['_lemma']
                      .nunique().unstack('word_class')
                      .reindex(columns=word_classes, fill_value=0))
    lemma_counts.columns = [f'lemmas_{c.replace("/", "_")}' for c in word_classes]

    # Assemble
    parts = [total_obs, wc_counts, wc_pct, vg_counts, vg_pct]
    if has_recall:
        parts += [wc_recall, vg_recall, overall_recall]
    parts.append(lemma_counts)

    profile_df = pd.concat(parts, axis=1)

    # Summary flags
    pct_wc_cols = [f'pct_{c.replace("/", "_")}' for c in word_classes]
    profile_df['dominant_class'] = (profile_df[pct_wc_cols]
                                    .idxmax(axis=1)
                                    .str.replace('pct_', '', regex=False))

    n_vocab   = profile_df['n_vocabulary']
    n_grammar = profile_df['n_grammar']
    total_vg  = n_vocab + n_grammar
    profile_df['vocab_grammar_ratio'] = np.where(total_vg > 0, n_vocab / total_vg, np.nan)
    profile_df['dominant_vg'] = profile_df['vocab_grammar_ratio'].apply(_dominant_vg_label)

    return profile_df


# ---------------------------------------------------------------------------
# 5.  NORMALISATION  (for clustering)
# ---------------------------------------------------------------------------

def normalise_profiles(profile_df, feature_cols=None):
    from sklearn.preprocessing import MinMaxScaler

    if feature_cols is None:
        feature_cols = [c for c in profile_df.columns
                        if c.startswith('pct_') or c.startswith('recall_')]

    scaler = MinMaxScaler()
    normed = profile_df[feature_cols].copy()
    normed = normed.dropna(axis=1, how='all')
    normed[normed.columns] = scaler.fit_transform(normed.fillna(0))
    return normed


# ---------------------------------------------------------------------------
# 6.  RUN
# ---------------------------------------------------------------------------

t0 = time.time()

print("Loading CSV...")
df = pd.read_csv(r'C:\Users\Desmi\Desktop\Schoolwerk\learning_traces.13m.csv')
print(f"  Loaded {len(df):,} rows, {df['user_id'].nunique():,} unique users ({time.time()-t0:.1f}s)")

print("Classifying lexemes...")
t1 = time.time()
df = add_classification_columns(df)
print(f"  Done ({time.time()-t1:.1f}s)")

print("Building user profiles...")
t2 = time.time()
profiles = build_user_profiles(df)
print(f"  Done — {len(profiles):,} user profiles ({time.time()-t2:.1f}s)")

print("Normalising...")
t3 = time.time()
normed = normalise_profiles(profiles)
print(f"  Done ({time.time()-t3:.1f}s)")

print("Saving to CSV...")
profiles.to_csv('user_profiles.csv')
normed.to_csv('user_profiles_normalised.csv')
print(f"  Saved. Total time: {time.time()-t0:.1f}s")

Title: Cyrillic Morphological Induction Grand Challenge 2026

URL Source: https://www.kaggle.com/competitions/cyrillic-morphological-induction-grand-challenge/overview/evaluation

Markdown Content:
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.

[](https://www.kaggle.com/sejoonchang)
Sejoon Chang  · Community Prediction Competition · 6 days to go

Infer the morphology of a synthetic Cyrillic language

![Image 1](https://www.kaggle.com/competitions/166974/images/header)

## Overview

The goal of this competition is to predict inflected forms in a synthetic Cyrillic language from Russian lemmas, grammatical information, and provided examples. Can you learn an unfamiliar morphological system well enough to generalize to forms you have never seen before?

Start

2 days ago

Close

6 days to go

### Description

Most NLP tasks are built around languages with existing dictionaries, corpora, and years of linguistic research. In this competition, the target language is synthetic and was created for the challenge  
 Your task is to predict the correct target-language form for each test example using the provided Russian lemma, grammatical information, dialect label, and other features. Labeled training and development data are provided to help you learn the relationship between the source information and the target forms.

This competition is designed to test how well a system can learn and generalize within an unfamiliar linguistic setting.

### Evaluation

Submissions are evaluated using a weighted combination of Exact Match and Character Error Rate (CER):

$$
\text{Score} = 0.9 \times \text{Weighted Exact Match} + 0.1 \times \left(\right. 1 - \text{Weighted CER} \left.\right)
$$

## Submission File

For each `id` in the test set, predict the corresponding `form_vvz`. The file should contain a header and have the following format:

```
id,form_vvz
1,а
2,а
3,а
etc.
```

See `sample_submission.csv` for the required format.

### Prizes

Total Prizes Available: $200

*   **1st Place**-$200
*   **2nd Place** - None
*   **3rd Place** - None

### Citation

Sejoon Chang. Cyrillic Morphological Induction Grand Challenge 2026. https://www.kaggle.com/competitions/cyrillic-morphological-induction-grand-challenge, 2026. Kaggle.

## Competition Host

Sejoon Chang

## Prizes & Awards

$200

Does not award Points or Medals

## Participation

24 Entrants

15 Participants

15 Teams

76 Submissions

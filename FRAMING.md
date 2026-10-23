# Framing: Rawaj Pay mule detection

## Decision
The core operational decision is whether to flag an instant transfer as a suspected "mule" account for manual review by the fraud analysis team. The five analysts act on this decision, and their daily capacity limits how many cases can be reviewed. False positives waste analyst time, while false negatives cost the business in undetected fraud.

## Metric
The primary evaluation metric is Expected Cost, which assigns a financial weight to False Positives and False Negatives to find a threshold that minimizes total business loss. Accuracy is inappropriate here because the dataset is highly imbalanced; predicting everything as "not a mule" would yield high accuracy but fail the business goal.

## Baseline
Without the model, the baseline is likely a simple heuristic rule-based system, such as automatically flagging transfers above a certain amount or from historically risky channels. The ML model must beat this baseline by demonstrating a lower overall expected cost.

## Classical ML or LLM
Classical ML (like LightGBM or XGBoost) is the right choice. We are dealing with structured, numerical tabular data that requires ultra-low latency inference for instant transfers. LLMs are too slow, expensive, and fundamentally unsuited for this type of quantitative thresholding task.
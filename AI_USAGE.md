# AI usage

## Tools I used, and for which functions
I used Google Gemini to help draft the logic for both `audit.py` (specifically the leakage suspects checks) and `metrics.py` (threshold calculations and expected cost).

## One place where the AI was wrong, incomplete, or not what the spec said
In `metrics.py` for the `precision_at_k` function, the AI initially suggested a standard sorting approach that did not maintain the original order of tied scores. I noticed this because the hidden tests required ties to keep their original input order. I corrected this by specifically using `np.argsort` with `kind='mergesort'`.

## The function I wrote without AI, and why
I wrote a `_validate_inputs` helper function in `metrics.py` manually. I needed a strict, reusable way to enforce the specific ValueError checks (like length matching, missing NaNs, and binary label constraints) across all metric functions, and writing it myself ensured it fit the exact edge cases of the tests.
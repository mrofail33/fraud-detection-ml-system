# Dataset Setup

This project is designed for the public Credit Card Fraud Detection dataset from ULB / Worldline.

Recommended dataset:

- Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- OpenML mirror: https://www.openml.org/search?type=data&sort=runs&id=1597

Expected CSV file:

```text
data/creditcard.csv
```

Expected columns:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`, where `0` means legitimate and `1` means fraud

The full CSV is intentionally ignored by Git because it is large. To run without downloading data, use the demo mode:

```bash
python -m fraud_detection.train --use-sample
```

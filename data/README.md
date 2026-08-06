# Data

The competition dataset is **not** redistributed in this repository. It remains subject to the
WiDS Datathon / Kaggle competition rules.

To run the pipeline, download the files from the
[competition page](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26) and place
them here:

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

`src/config.py` picks this directory up automatically when `/kaggle/input/...` is absent, so
the same code runs locally and on Kaggle without edits.

The tests in `tests/` generate synthetic data with the same schema, so you can verify the
install without any of these files.

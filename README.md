# Document Rectification

**Download dataset**

```bash
kaggle d download sharmaharsh/form-understanding-noisy-scanned-documentsfunsd -p .data
unzip .data/form-understanding-noisy-scanned-documentsfunsd.zip -d .data
```

**Examine the data**
```bash
poetry run data
```

**Run training**
```bash
poetry run train

poetry run tensorboard --logdir .logs
```

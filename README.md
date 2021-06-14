# Document Rectification

**Download dataset**

Auto-downloading the dataset upon initial usage with `kaggle` CLI.

To download manually:
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
poetry run wb_server # or
poetry run tb_server

poetry run train
```

**To run `jupyter` notebook add custom poetry kernel add custom kernel.**
[SRC - Add Jupyter kernel with poetry](https://docs.pymedphys.com/contrib/other/add-jupyter-kernel.html)
```bash
poetry run python -m ipykernel install --user --name document-rectification-kernel
```

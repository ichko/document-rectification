# Document Rectification

Experiments on document rectification

**Task**
- Input an image with a document on it - skewed and/or over a background
- Output only the document with straightened lines

**Approach**
- Get dataset with scanned documents
- Generate transformations of the documents placed over a complex background
- Train an STN (spatial transformer network) to consume complex image and rectify it
    - simplest architecture - complex image -> stn -> scanned image
    - auto-encoder based architecture
        - complex image -> encoder (short vector) -> cir
        - scanned image -> encoder (short vector) -> sir
        - minimize cosine diff between high level representations (cir ~ sir)
        - sir -> decoder -> minimize reconstruction loss with scanned image
    - GAN based architecture
        - complex image -> generator (stn) -> same size image (cig)
        - train discriminator to distinguish between scanned and not scanned images
        - force generator to rectify images as to be in the same distribution scanned images come from


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

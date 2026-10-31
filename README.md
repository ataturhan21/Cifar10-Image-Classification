# CIFAR-10 Image Classification

A TensorFlow/Keras notebook for the Kaggle CIFAR-10 competition. It loads archived images, trains a convolutional classifier, and generates predictions aligned with the sample submission.

**[Open the notebook](cifar10-image-classification.ipynb)** · **[Dependencies](requirements.txt)** · **[MIT license](LICENSE)**

## Model and workflow

- Extract `train.7z` and `test.7z`; read labels from `trainLabels.csv`.
- Load 32 × 32 images with OpenCV and normalize pixel values to 0–1.
- Split training images into 80% training and 20% validation with `random_state=42`.
- Use 3 convolutional blocks with 32, 64, and 128 filters, batch normalization, pooling, dropout, and L2 regularization.
- Train with augmentation, Adam, early stopping, and learning-rate reduction. The current training call uses batch size 128 and a maximum of 5 epochs.
- Export `submission.csv` with `id` and `label` columns.

## Run on Kaggle

1. Create a notebook and attach the [CIFAR-10 competition data](https://www.kaggle.com/competitions/cifar-10/data).
2. Import `cifar10-image-classification.ipynb`.
3. Confirm these inputs in **Loading the Dataset**:

   - `/kaggle/input/cifar-10/trainLabels.csv`
   - `/kaggle/input/cifar-10/sampleSubmission.csv`
   - `/kaggle/input/cifar-10/train.7z`
   - `/kaggle/input/cifar-10/test.7z`

4. Confirm that extraction writes to `/kaggle/working/train` and `/kaggle/working/test`.
5. Run the cells in order and download `submission.csv`.

The current loader holds image arrays in memory. Check available RAM and storage before extracting and loading the full competition data.

## Run locally

```bash
git clone https://github.com/ataturhan21/Cifar10-Image-Classification.git
cd Cifar10-Image-Classification
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt notebook
jupyter notebook cifar10-image-classification.ipynb
```

Download the competition data separately and replace the Kaggle input and extraction paths. Data archives and trained weights are not included. Saved metadata records Python 3.10.14; dependency versions are currently unpinned.

## Prediction integrity

Training and prediction share the same explicit class mapping. Test images are loaded in `sampleSubmission.csv` ID order instead of filesystem order. Missing images, unknown labels, mismatched image counts, and unexpected prediction shapes raise errors before a misleading CSV is written.

## Regression checks

```bash
python -m unittest discover -s tests -v
```

The tests exercise the notebook's actual loading and submission functions with synthetic inputs. They require NumPy and pandas, without TensorFlow training, OpenCV installation, or the competition dataset.

The existing notebook outputs are historical. The tests verify label and row alignment, not a newly measured model accuracy.

## License

[MIT](LICENSE).

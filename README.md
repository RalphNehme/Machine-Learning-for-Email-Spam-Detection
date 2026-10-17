# Machine Learning for Email Spam Detection

A spam classifier built as a small neural network written from scratch in NumPy (no deep learning framework). It is trained on the UCI Spambase dataset and predicts whether an email is spam (1) or not (0).

## Results

On 921 held-out test emails (20% stratified split):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9327 |
| Precision | 0.9239 |
| Recall    | 0.9036 |
| F1 score  | 0.9136 |

The notebook also plots a confusion matrix and the training loss and accuracy over the epochs.

## How it works

1. **Data:** `spambase.data` has 4,601 emails (1,813 spam, 2,788 non-spam), each described by 57 numeric features: word frequencies, character frequencies such as `!` and `$`, and capital-letter run lengths. Feature names are read from `spambase.names`.
2. **Preprocessing:** an 80/20 stratified train/test split, then standardization with `StandardScaler` fit on the training set only.
3. **Model:** a feed-forward network with 57 inputs, one hidden layer of 64 sigmoid units and a single sigmoid output. Weights use Xavier initialization and both layers have biases.
4. **Training:** full-batch gradient descent with binary cross-entropy loss and hand-written backpropagation, 1,000 epochs at a learning rate of 0.5.
5. **Evaluation:** a 0.5 threshold on the output probability, scored with scikit-learn metrics.

## Running it

Requirements: Python 3 with `numpy`, `pandas`, `scikit-learn`, `matplotlib` and Jupyter. All of these come with Anaconda.

```bash
git clone https://github.com/RalphNehme/Machine-Learning-for-Email-Spam-Detection.git
cd Machine-Learning-for-Email-Spam-Detection
jupyter notebook ML_Project.ipynb
```

Run the cells in order from the repository folder, since the notebook loads the data files by relative path.

## Repository contents

| File | Description |
|------|-------------|
| `ML_Project.ipynb` | The notebook: data loading, model, training and evaluation |
| `spambase.data` | Spambase dataset, one email per row, label in the last column |
| `spambase.names` | Feature names for the dataset |
| `spambase.DOCUMENTATION` | Original dataset documentation |
| `easy_ham/` | 2,551 raw non-spam emails from the SpamAssassin public corpus (not used by the current notebook) |

## Dataset

Spambase was created by Mark Hopkins, Erik Reeber, George Forman and Jaap Suermondt at Hewlett-Packard Labs (1999) and is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase). Its non-spam emails come from one person's work and personal mail, so a model trained on it is closer to a personalized filter than a general-purpose one.

## License

[MIT](LICENSE)

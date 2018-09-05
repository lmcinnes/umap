"""
UMAP as a Feature Extraction Technique for Classification
---------------------------------------------------------

The following script shows how UMAP can be used as a feature extraction
technique to improve the accuracy on a classification task. It also shows
how UMAP can be integrated in standard scikit-learn pipelines.

The first step is to create a dataset for a classification task, which is
performed with the function ``sklearn.datasets.make_classification``. The
dataset is then split into a training set and a test set using the
``sklearn.model_selection.train_test_split`` function.

Second, a linear SVM is fitted on the training set. To choose the best
hyperparameters automatically, a gridsearch is performed on the training set.
The performance of the model is then evaluated on the test set with the
accuracy metric.

 Third, the previous step is repeated with a slight modification: UMAP is
 used as a feature extraction technique. This small change results in a
 substantial improvement compared to the model where raw data is used.
"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from umap import UMAP


# Make a toy dataset
X, y = make_classification(n_samples=1000, n_features=300, n_informative=250,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           random_state=1212)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Classification with a linear SVM
svc = LinearSVC(dual=False, random_state=123)
params_grid = {"C": [10**k for k in range(-3, 4)]}
clf = GridSearchCV(svc, params_grid)
clf.fit(X_train, y_train)
print("Accuracy on the test set with raw data: {:.3f}".format(
    clf.score(X_test, y_test)))

# Transformation with UMAP followed by classification with a linear SVM
umap = UMAP(random_state=456)
pipeline = Pipeline([("umap", umap),
                     ("svc", svc)])
params_grid_pipeline = {"umap__n_neighbors": [5, 20],
                        "umap__n_components": [15, 25, 50],
                        "svc__C": [10**k for k in range(-3, 4)]}

clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
clf_pipeline.fit(X_train, y_train)
print("Accuracy on the test set with UMAP transformation: {:.3f}".format(
    clf_pipeline.score(X_test, y_test)))

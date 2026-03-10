import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline, FeatureUnion

from ..umap_ import UMAP


def test_get_feature_names_out():
    X, _ = make_classification(n_samples=30, n_features=10, random_state=42)
    umap = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        n_epochs=200,
        random_state=42,
        n_components=3,
    ).fit(X)
    # get_feature_names_out should not care about passed features.
    features_names_in = [f"feature{i}" for i in range(10)]
    feature_names_out = umap.get_feature_names_out(input_features=features_names_in)
    expected = ["umap0", "umap1", "umap2"]
    np.testing.assert_array_equal(feature_names_out, expected)


def test_get_feature_names_out_default():
    X, _ = make_classification(n_samples=30, n_features=10, random_state=42)
    umap = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        n_epochs=200,
        random_state=42,
        n_components=3,
    ).fit(X)
    # get_feature_names_out should generate feature names in a certain format if no names are passed.
    default_result = umap.get_feature_names_out()
    expected_default_result = ["umap0", "umap1", "umap2"]
    np.testing.assert_array_equal(default_result, expected_default_result)


def test_get_feature_names_out_multicomponent():
    # The output length should be equal to the number of components UMAP generates.
    X, _ = make_classification(n_samples=30, n_features=10, random_state=42)
    umap = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        n_epochs=200,
        random_state=42,
        n_components=9,
    ).fit(X)
    result_umap = umap.get_feature_names_out()
    expected_umap_result = [f"umap{i}" for i in range(9)]
    assert len(result_umap) == 9
    np.testing.assert_array_equal(result_umap, expected_umap_result)



def test_get_feature_names_out_featureunion():
    X, _ = make_classification(n_samples=30, n_features=10, random_state=42)
    pipeline = Pipeline(
        [
            (
                "umap_pipeline",
                FeatureUnion(
                    [
                        ("umap1", UMAP(n_components=2)),
                        ("umap2", UMAP(n_components=3)),
                    ]
                ),
            )
        ]
    )

    pipeline.fit(X)
    feature_names = pipeline.get_feature_names_out()
    expected_feature_names = np.array(
        [
            "umap1__umap0",
            "umap1__umap1",
            "umap2__umap0",
            "umap2__umap1",
            "umap2__umap2",
        ]
    )
    np.testing.assert_array_equal(feature_names, expected_feature_names)

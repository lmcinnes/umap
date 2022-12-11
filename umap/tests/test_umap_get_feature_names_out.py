import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline, FeatureUnion

from ..umap_ import UMAP


def test_get_feature_names_out_passthrough():
    umap = UMAP()
    # get_feature_names_out should return same names if feature are passed in directly.
    example_passthrough = ['feature1', 'feature2']
    passthrough_result = umap.get_feature_names_out(feature_names_out=example_passthrough)
    assert example_passthrough == passthrough_result


def test_get_feature_names_out_default():
    umap = UMAP()
    # get_feature_names_out should generate feature names in a certain format if no names are passed.
    default_result = umap.get_feature_names_out()
    expected_default_result = ["umap_component_1", "umap_component_2"]
    assert default_result == expected_default_result


def test_get_feature_names_out_multicomponent():
    # The output length should be equal to the number of components UMAP generates.
    umap10 = UMAP(n_components=10)
    result_umap10 = umap10.get_feature_names_out()
    expected_umap10_result = [f"umap_component_{i+1}" for i in range(10)]
    assert len(result_umap10) == 10
    assert result_umap10 == expected_umap10_result


def test_get_feature_names_out_featureunion():
    X, _ = make_classification(n_samples=10)
    pipeline = Pipeline(
        [
            (
                "umap_pipeline",
                FeatureUnion(
                    [
                        ("umap1", UMAP()),
                        ("umap2", UMAP(n_components=3)),
                    ]
                ),
            )
        ]
    )

    pipeline.fit(X)
    feature_names = pipeline.get_feature_names_out()
    expected_feature_names = np.array(["umap1__umap_component_1", "umap1__umap_component_2", "umap2__umap_component_1",
                                       "umap2__umap_component_2", "umap2__umap_component_3"])
    np.testing.assert_array_equal(feature_names, expected_feature_names)

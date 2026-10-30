"""Shared case definitions for the umap-learn 0.5.x compatibility fixtures.

This module must not import ``umap`` at module level: the fixture generator
loads it alongside an installed umap-learn 0.5.x, while the regression test
loads it alongside the current package.
"""

import numpy as np

FIXTURE_FILE = "compat_0_5_final.npz"
N_EPOCHS = 50
N_SAMPLES = 600
N_TRANSFORM = 40


def _datasets():
    import scipy.sparse
    from sklearn.datasets import load_digits

    digits = load_digits()
    data = digits.data[:N_SAMPLES].astype(np.float32)
    target = digits.target[:N_SAMPLES]
    query = digits.data[N_SAMPLES : N_SAMPLES + N_TRANSFORM].astype(np.float32)
    sparse = scipy.sparse.csr_matrix(np.where(data > 8, data, 0))
    sparse_query = scipy.sparse.csr_matrix(np.where(query > 8, query, 0))
    return data, target, query, sparse, sparse_query


def build_cases():
    """Return ``{name: (fit_data, target, transform_data, kwargs)}``."""
    import numba

    @numba.njit()
    def l1_metric(x, y):
        return np.sum(np.abs(x - y))

    data, target, query, sparse, sparse_query = _datasets()
    return {
        "spectral": (data, None, query, dict(init="spectral")),
        "pca": (data, None, query, dict(init="pca")),
        "random": (data, None, query, dict(init="random")),
        "tswspectral": (data, None, query, dict(init="tswspectral")),
        "three_components": (data, None, query, dict(n_components=3)),
        "nndescent": (
            data,
            None,
            query,
            dict(force_approximation_algorithm=True),
        ),
        "sparse_cosine": (sparse, None, sparse_query, dict(metric="cosine")),
        "sparse_nndescent": (
            sparse,
            None,
            sparse_query,
            dict(metric="cosine", force_approximation_algorithm=True),
        ),
        "custom_metric": (data, None, None, dict(metric=l1_metric)),
        "haversine_output": (data, None, query, dict(output_metric="haversine")),
        "densmap": (data, None, None, dict(densmap=True, output_dens=True)),
        "supervised": (data, target, query, dict(target_weight=0.5)),
        "mix_ratio": (data, None, query, dict(set_op_mix_ratio=0.5)),
    }


def run_case(umap_module, fit_data, target, transform_data, kwargs, extra_kwargs):
    """Fit one case and return a dict of named result arrays."""
    model = umap_module.UMAP(
        n_epochs=N_EPOCHS, random_state=42, **kwargs, **extra_kwargs
    ).fit(fit_data, target)
    results = {"embedding": model.embedding_}
    if transform_data is not None:
        results["transform"] = model.transform(transform_data)
    if kwargs.get("output_dens"):
        results["rad_orig"] = model.rad_orig_
        results["rad_emb"] = model.rad_emb_
    return results


def run_aligned_case(umap_module, extra_kwargs):
    """Fit and update an AlignedUMAP model on overlapping digits slices."""
    from sklearn.datasets import load_digits

    data = load_digits().data.astype(np.float32)
    slices = [data[i * 200 : (i + 2) * 200] for i in range(3)]
    relations = [{k: v for v, k in enumerate(range(200, 400))} for _ in range(2)]
    model = umap_module.AlignedUMAP(
        n_epochs=N_EPOCHS, random_state=42, **extra_kwargs
    ).fit(slices, relations=relations)
    results = {f"embedding_{i}": e.copy() for i, e in enumerate(model.embeddings_)}
    model.update(data[800:1000], relations={v: v for v in range(200)})
    results["updated"] = model.embeddings_[-1]
    return results

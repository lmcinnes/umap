"""The datasets the map is built for.

Each dataset has a folder beneath ``DATA_ROOT`` holding its data, its
nearest-neighbor graph, its runs and everything built from them. Adding a
dataset means writing a loader, which returns the data, an integer class
label per point and the class names, and a palette maker, which gives a
colour per class, and listing them in ``DATASETS``.
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("UMAP_PARAM_MAP_DIR", REPO_ROOT / "experiments" / "parameter_map"))
# Neighbors in the shared k-NN graph; each run prunes it to its n_neighbors.
K_MAX = 512


@dataclass(frozen=True)
class Dataset:
    name: str
    label: str  # in the viewer's drop-down
    title: str  # in the viewer's side panel
    metric: str  # of the input space
    size: int  # parameter settings in its map: a power of four
    loader: Callable[["Dataset"], tuple]  # -> (data, labels, class names)
    palette_maker: Callable[[list], list]  # class names -> RGB colours in 0-255

    @property
    def dir(self):
        return DATA_ROOT / self.name

    def load(self):
        """Return (data, labels), loading and caching them the first time."""
        data_path, label_path = self.dir / "data.npy", self.dir / "labels.npy"
        if not data_path.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            data, labels, names = self.loader(self)
            np.save(data_path, np.ascontiguousarray(data, dtype=np.float32))
            np.save(label_path, np.asarray(labels, dtype=np.int32))
            with open(self.dir / "label_names.json", "w") as f:
                json.dump(list(names), f)
        return np.load(data_path), np.load(label_path)

    def palette(self):
        """Return an (n_classes, 3) array of colours in 0-255, one per label."""
        path = self.dir / "palette.json"
        if not path.exists():
            self.load()
            with open(self.dir / "label_names.json") as f:
                colours = self.palette_maker(json.load(f))
            with open(path, "w") as f:
                json.dump([[int(v) for v in colour] for colour in colours], f)
        with open(path) as f:
            return np.array(json.load(f), dtype=np.float64)

    def knn(self):
        """Return the exact K_MAX-nearest-neighbor graph as (indices,
        distances), each point first in its own row."""
        index_path = self.dir / f"knn_indices_{K_MAX}.npy"
        dist_path = self.dir / f"knn_dists_{K_MAX}.npy"
        if not index_path.exists():
            from sklearn.neighbors import NearestNeighbors

            data, _ = self.load()
            nn = NearestNeighbors(
                n_neighbors=K_MAX, algorithm="brute", metric=self.metric, n_jobs=-1
            )
            dists, indices = nn.fit(data).kneighbors(data)
            # Duplicate points can displace a point from the head of its own row.
            rows = np.arange(len(data))
            for row in np.flatnonzero(indices[:, 0] != rows):
                hits = np.flatnonzero(indices[row] == row)
                pos = hits[0] if hits.size else K_MAX - 1
                indices[row, 1 : pos + 1] = indices[row, :pos]
                dists[row, 1 : pos + 1] = dists[row, :pos]
                indices[row, 0], dists[row, 0] = row, 0.0
            np.save(index_path, indices.astype(np.int32))
            np.save(dist_path, dists.astype(np.float32))
        return np.load(index_path), np.load(dist_path)


def _download(url, path):
    """Fetch ``url`` to ``path`` once, never leaving a partial file."""
    import urllib.request

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def _glasbey(n):
    """``n`` distinct colours, clear of near-black and of tints too pale to
    read on white."""
    import glasbey

    colours = glasbey.create_palette(n, lightness_bounds=(25, 75), as_hex=False)
    return [[round(255 * c) for c in colour[:3]] for colour in colours]


# -- Fashion-MNIST ------------------------------------------------------------

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]  # fmt: skip
# Matplotlib's tab10, one colour per class.
TAB10 = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207],
]  # fmt: skip


def load_fashion_mnist(dataset):
    from sklearn.datasets import fetch_openml

    fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    return fmnist.data, fmnist.target.astype(np.int32), FASHION_CLASSES


# -- 20 Newsgroups ------------------------------------------------------------

# Posts embedded with nomic-embed-text. The groups fall in six broad
# categories; labels follow this order, and the palette gives each category
# a block of related colours.
NEWSGROUPS_SOURCE = (
    "https://huggingface.co/datasets/lmcinnes/20newsgroups_embedded"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
NEWSGROUP_CATEGORIES = {
    "computers": [
        "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware", "comp.windows.x",
    ],
    "sports": ["rec.sport.baseball", "rec.sport.hockey"],
    "automobiles": ["rec.autos", "rec.motorcycles"],
    "politics and religion": [
        "talk.politics.guns", "talk.politics.mideast", "talk.politics.misc",
        "talk.religion.misc", "alt.atheism", "soc.religion.christian",
    ],
    "science": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "other": ["misc.forsale"],
}  # fmt: skip


def load_newsgroups(dataset):
    import pyarrow.parquet as pq

    source = _download(NEWSGROUPS_SOURCE, dataset.dir / "raw" / "20newsgroups_embedded.parquet")
    table = pq.read_table(source, columns=["newsgroup", "embedding"])
    order = [group for groups in NEWSGROUP_CATEGORIES.values() for group in groups]
    labels = [order.index(name) for name in table.column("newsgroup").to_pylist()]
    return np.array(table.column("embedding").to_pylist()), labels, order


def newsgroups_palette(names):
    import glasbey

    blocks = [len(groups) for groups in NEWSGROUP_CATEGORIES.values()]
    colours = glasbey.create_block_palette(blocks, lightness_bounds=(25, 75), as_hex=False)
    return [[round(255 * c) for c in colour[:3]] for colour in colours]


# -- C. elegans embryogenesis --------------------------------------------------

# Packer et al. 2019, GEO GSE126954: a genes x cells count matrix and cell
# annotations. Preprocessed as is usual: cells passing the authors' QC,
# counts per 10,000, log1p, the most variable genes, then principal components.
CELEGANS_SOURCE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126954/suppl/"
CELEGANS_ANNOTATION = "GSE126954_cell_annotation.csv.gz"
CELEGANS_COUNTS = "GSE126954_gene_by_cell_count_matrix.txt.gz"
CELEGANS_GENES = 2000
CELEGANS_COMPONENTS = 50
UNANNOTATED = "unannotated"


def load_celegans(dataset):
    """Cells labelled by cell type: commonest first, unannotated last."""
    import pandas as pd
    import scipy.io
    from sklearn.decomposition import PCA

    raw = {name: _download(CELEGANS_SOURCE + name, dataset.dir / "raw" / name)
           for name in (CELEGANS_ANNOTATION, CELEGANS_COUNTS)}  # fmt: skip
    cells = pd.read_csv(raw[CELEGANS_ANNOTATION], index_col=0)
    counts = scipy.io.mmread(raw[CELEGANS_COUNTS]).T.tocsr()  # cells x genes
    assert counts.shape[0] == len(cells), (counts.shape, len(cells))
    keep = cells["passed_initial_QC_or_later_whitelisted"].astype(bool).to_numpy()
    cells, counts = cells[keep], counts[keep].astype(np.float32)

    totals = np.asarray(counts.sum(axis=1)).ravel()
    counts = counts.multiply(1e4 / totals[:, None]).tocsr()
    counts.data = np.log1p(counts.data)
    mean = np.asarray(counts.mean(axis=0)).ravel()
    variance = np.asarray(counts.multiply(counts).mean(axis=0)).ravel() - mean**2
    expression = counts[:, np.argsort(variance)[-CELEGANS_GENES:]].toarray()
    pcs = PCA(CELEGANS_COMPONENTS, svd_solver="randomized", random_state=0).fit_transform(
        expression - expression.mean(axis=0)
    )

    types = cells["cell.type"].fillna(UNANNOTATED)
    names = [t for t in types.value_counts().index if t != UNANNOTATED]
    if (types == UNANNOTATED).any():
        names.append(UNANNOTATED)
    return pcs, [names.index(t) for t in types], names


def celegans_palette(names):
    """Distinct colours for cell types; grey for unannotated cells."""
    typed = [name for name in names if name != UNANNOTATED]
    return _glasbey(len(typed)) + ([[190, 190, 190]] if len(typed) < len(names) else [])


DATASETS = {
    d.name: d
    for d in [
        Dataset(
            "fashion_mnist", "Fashion-MNIST", "Fashion-MNIST (70,000 points)",
            "euclidean", 16384, load_fashion_mnist, lambda names: TAB10,
        ),
        Dataset(
            "20newsgroups", "20 Newsgroups",
            "20 Newsgroups, nomic-embed-text embeddings (18,170 posts)",
            "cosine", 4096, load_newsgroups, newsgroups_palette,
        ),
        Dataset(
            "celegans", "C. elegans embryo",
            "C. elegans embryogenesis (Packer et al. 2019), 50 principal components",
            "euclidean", 4096, load_celegans, celegans_palette,
        ),
    ]
}  # fmt: skip

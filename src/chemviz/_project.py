"""Placing new molecules on an existing map (see ``Map.project``)."""

import numpy as np
import pandas as pd

from chemviz.maps import Map, _drop_invalid, _encode

_INPUTS = {"ecfp": "SMILES", "minhash": "uint32 MinHash vectors", "fingerprints": "binary fingerprints"}


def project(ref: Map, data, smiles=None, n_jobs=None) -> Map:
    model = ref.model
    if model is None:
        raise ValueError("This map has no model (it was saved with model=False): "
                         "new molecules cannot be placed on it")
    edges = np.empty((0, 2), dtype=np.int64)
    if ref.method == "pca":
        from chemviz._pca import _transform

        pos, frame, column, invalid = _transform(model, data, smiles, n_jobs)
        new_model = model
    else:
        hashes, frame, column, invalid, encoding = _encode(data, smiles, model["n_permutations"],
                                                           model["radius"], n_jobs)
        ref_hashes = model["hashes"]
        if encoding != model["encoding"] or hashes.shape[1] != ref_hashes.shape[1]:
            raise ValueError(f"This map was made from {_INPUTS[model['encoding']]}; "
                             "give new molecules the same way")
        keep, frame = _drop_invalid(frame, invalid)
        if keep is not None:
            hashes = hashes[keep]
        ref_pos = np.column_stack([ref.x, ref.y])
        if ref.method == "tmap":
            from chemviz._tmap import _transform

            pos, edges = _transform(ref_pos, ref.edges, ref_hashes, hashes, model)
        elif ref.method == "umap":
            from chemviz._umap import _transform

            pos = _transform(ref_pos, ref_hashes, hashes, model)
        else:
            from chemviz._tsne import _transform

            pos = _transform(ref_pos, ref_hashes, hashes, model)
        new_model = {**model, "hashes": np.concatenate([ref_hashes, hashes])}

    if ref.smiles is not None and column is not None and column != ref.smiles:
        frame = frame.rename(columns={column: ref.smiles})
    old = ref.data if "projected" in ref.data.columns else ref.data.assign(projected=False)
    both_ranges = isinstance(old.index, pd.RangeIndex) and isinstance(frame.index, pd.RangeIndex)
    data = pd.concat([old, frame.assign(projected=True)], ignore_index=both_ranges)
    return Map(np.concatenate([ref.x, pos[:, 0]]).astype(np.float32),
               np.concatenate([ref.y, pos[:, 1]]).astype(np.float32),
               np.concatenate([ref.edges, edges]).astype(np.int64),
               data, ref.invalid, ref.smiles or column, ref.method, new_model)

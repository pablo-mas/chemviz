# Gallery

Interactive maps of the same 20,000 molecules, a random sample of
[ZINC250k](https://github.com/the-matter-lab/chemical_vae), made by
[`make_gallery.py`](https://github.com/pablo-mas/chemviz/blob/main/docs/make_gallery.py)
at every build of this documentation. Each map is the HTML file chemviz writes with
`m.save("map.html")`.

Hover over a point to see the molecule, change the colouring in the panel (QED, logP,
synthetic accessibility), search a ZINC row, add filters, and select with a click or
Shift + drag. Use the ☰ button to hide the panel, or open a map in its own tab for more room.

```{raw} html
<style>
.chemviz-map { width: 100%; height: 600px; border: 1px solid var(--pst-color-border); border-radius: 4px; }
</style>
```

## TMAP

The tree links each molecule to a close neighbour: follow a branch to see a series of
related molecules, and its side branches for their variations.

```{raw} html
<iframe class="chemviz-map" src="maps/tmap.html" loading="lazy" title="TMAP of 20,000 ZINC molecules"></iframe>
<p><a href="maps/tmap.html" target="_blank">Open in a new tab</a></p>
```

## UMAP

Similar molecules gather in clusters; distances between clusters mean little.

```{raw} html
<iframe class="chemviz-map" src="maps/umap.html" loading="lazy" title="UMAP of 20,000 ZINC molecules"></iframe>
<p><a href="maps/umap.html" target="_blank">Open in a new tab</a></p>
```

## t-SNE

Neighbourhoods are kept best: zoom into a region to find closely related molecules.

```{raw} html
<iframe class="chemviz-map" src="maps/tsne.html" loading="lazy" title="t-SNE of 20,000 ZINC molecules"></iframe>
<p><a href="maps/tsne.html" target="_blank">Open in a new tab</a></p>
```

## PCA

A linear projection: colour by logP to see how it varies along the axes.

```{raw} html
<iframe class="chemviz-map" src="maps/pca.html" loading="lazy" title="PCA of 20,000 ZINC molecules"></iframe>
<p><a href="maps/pca.html" target="_blank">Open in a new tab</a></p>
```

## Placing new molecules

A TMAP of 18,000 of the molecules, on which the other 2,000 were then placed with
`Map.project` (see {doc}`projection`). They are coloured by the `projected` column; filter
on it to show them alone.

```{raw} html
<iframe class="chemviz-map" src="maps/projection.html" loading="lazy" title="2,000 molecules placed on a TMAP of 18,000"></iframe>
<p><a href="maps/projection.html" target="_blank">Open in a new tab</a></p>
```

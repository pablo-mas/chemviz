# Saving maps

A map can be saved in two ways, chosen by the file extension:

```python
m.save("map.html")       # the viewer: to look at and share, no Python needed
m.save("map.chemviz")    # the map itself: to reload in Python without recomputing it
m = cv.load("map.chemviz")
```

Save a `.chemviz` file whenever computing the map took a while: reloading a map of a
million molecules takes about a second, and the reloaded map can be recoloured, plotted,
exported to the viewer again or receive new molecules ({doc}`projection`).

## What a .chemviz file keeps

- the coordinates, the tree edges (TMAP) and the method;
- the data with its column types: numbers, booleans, text, categories, nullable integers,
  dates, and the index (including multi-level indexes);
- the positions of the molecules that could not be parsed (`m.invalid`);
- the **model**: what {meth}`Map.project <chemviz.Map.project>` needs to place new
  molecules, that is the method's parameters and the molecules' MinHash vectors (for PCA,
  its two axes).

Columns of other types (Python objects, lists) are saved as text, with a warning.

## Size

The MinHash vectors take 512 bytes per molecule and do not compress, so they make up most
of the file: 139 MB for ZINC250k. Without them, the same map takes 11 MB, but new molecules
can no longer be placed on it:

```python
m.save("map.chemviz", model=False)
```

## Safety

A `.chemviz` file is a zip archive of NumPy arrays and a JSON description. It is read
without pickle, so loading a file from someone else cannot run code on your machine, unlike
pickled objects.

## From the command line

Every subcommand can write a `.chemviz` file next to its other outputs:

```bash
chemviz tmap library.csv -o library.chemviz -o library.html
```

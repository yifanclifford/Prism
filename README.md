# Projection regularized item similarity model (Prism)

Implementation of Prism for our paper published in sigIR 2017: Top-N Recommendation with High-Dimensional Side Information via Locality Preserving Projection

The code is reimplemented in pytorch (>=1.0.0) with python (>=3.6).

We provide a sample dataset ''music'': #user=188, #item=900, #rating=2716, #feature=3322. The music dataset is originally from Amazon review data (http://jmcauley.ucsd.edu/data/amazon/).

You can simply run the code by:
```python
python3 prism.py
```

For detailed settings of running, please command following:
```python
python3 prism.py -h
```


# KD Trees

<p align="center">
  <img src = "../assets/img/kd_trees.png" alt="KD Trees">
  <br>
  <small><i>Image source: https://www.astroml.org/book_figures/chapter2/fig_kdtree_example.html</i></small>
</p>

KD Trees is a data structure built over KNNs to reduce inference time. Instead of comparing a test point with every data point, KD-Trees organize the data so we can skip many unnecessary comparisons.

The main idea is to split the data space into two parts. When a new point is given, we first check which part it belongs to. We search that part first. We only search the other part if it might contain a closer neighbor than the ones we have already found. By repeating this splitting process, the data forms a tree structure. This allows us to quickly narrow down the search and reduces the time needed to find nearest neighbors.

<p align="center">
  <img src = "../assets/img/india_map.jpg" alt="India Map">
  <br>
  <small><i>Image source: https://www.mapsofindia.com/zonal/</i></small>
</p>

Imagine you are standing in a large crowd and want to find who your nearest neighbors are. The standard kNN approach would be to ask every person where they live. A smarter approach is to first divide the crowd into groups. For example, ask everyone from southern India to raise their hands. If you also live in southern India, your nearest neighbors are more likely to be in this group, so you check them first.

There are cases where this may not be enough. For example, if you live in Telangana, some people from Maharashtra might actually be closer to you. In such situations, you also need to check the other group. Even in the worst case, you may still need to check everyone, which is no worse than the original kNN approach. But in most cases, this method saves a lot of work by avoiding unnecessary checks. This is similar to hit-miss concept in caching. There are a few instances where you will have to go to main memory, but for many cases, cache memory will provide you a faster inference.

## Limitations

Similar to KNNs, KD trees also suffer from curse of dimensionality. However, unlike KNNs which work for low intrinsic dimensionality, KD trees can not even work for low intrinsic dimensionality. KD-Trees divide the space using axis-aligned splits (splits along one feature at a time). In high dimensions, even if the data actually lies on a low-dimensional manifold, these axis-aligned splits do not align well with the true shape of the data. As a result, nearby points often end up in different partitions. For such cases, we can use ball trees. 
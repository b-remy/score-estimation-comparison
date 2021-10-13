# Neural Score Estimation Comparison for Generative Modeling

How models generalize? (The higher the log likelihood, the closer to the data distribution)
![](images/generalization_error_distance.png)

Methods comparison at commit 3b6a122c81bdddf08871e90fd6c6ad6d0973f03e.

See notebook in branch *normalizing_flows*

![](images/methods_comparison.png)

First results at commit 13dc9331c78c7091ee6faf5e18f6e8cbc1c6b21f
(for one draw)

**Tow moons dataset (L=2)**

*At large scale*

![](images/error_comparison_two_moons_2.png)

*At close scale*

![](images/error_comparison_moons_zoomed.png)

**Two gaussians dataset (L=2)**

![](images/error_comparison_blobs_2.png)

**Swiss roll dataset (L=2)**

![](images/error_comparison_swiss_roll_2.png)

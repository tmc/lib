# Common-domain oracle

Generate with Python 3.12 and clean NVlabs/CuTe commit
111253d17e2f0f8631f43999b43ac4afa5954b04:

```
PYTHONPATH="$CUTE" python3.12 common_domain_oracles.py > common_domain_oracles.txt
```

CUTE is the pinned checkout root containing pycute. The retained output was
generated with Python 3.12.13. Python 3.14 raises NameError in this reference's
prefix_product. No reference implementation is copied into the Go package.

The table normalizes Python tuple spaces and trailing commas to Go layout
notation. Broad generated pairs check symmetry and that common size divides
the input-size gcd; they do not claim equality. For example, contiguous shapes
(2,3) and (3,2) both have size6 but the reference returns (1,):(0,).

# simple_image_recon_py

Python module (using pybind11) to access the [simple image reconstruction library](https://github.com/berndpfrommer/simple_image_recon_lib).

# How to build
```
git clone https://github.com/berndpfrommer/simple_image_recon_py.git
cd simple_image_recon_py
git clone https://github.com/berndpfrommer/simple_image_recon_lib.git
mkdir build
cd build
cmake .. && make
```

# How to try it out

You must start python inside the ``build`` directory to import the library!

```python
from _simple_image_recon import SimpleImageRecon
from matplotlib import pyplot as plt

recon = SimpleImageRecon(width, height, 30, 2, 0.6)
recon.update(my_structured_event_array)
plt.imshow(recon.get_state()['L'], cmap='gray')
plt.show()
```

Make sure the x, y  coordinates are within the sensor resolution. There is no sanity check!


# How to convert float event arrays to structured array:

```python
from numpy.lib.recfunctions import unstructured_to_structured
float_events = np.load('my_events.npy')
events = unstructured_to_structured(
    float_events[:, (0, 1, 3, 2)], dtype=np.dtype([('x', 'u2'), ('y', 'u2'), ('p', 'i1'), ('t', 'i4')]))
```
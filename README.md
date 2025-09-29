# GDSLayout 0.1.1

GDSLayout is a Python library for designing chips’ layout. Built upon the GDSFactory package, GDSLayout introduces a range of novel structures within its `device` directory. These additions include components such as **high-Q resonators**, **photonic crystals**, and **mechanical structures**.

A key feature of GDSLayout is its modular and configurable approach to design. Each structural unit is defined by a dedicated YAML configuration file. This allows for a streamlined workflow where users can generate complex layouts by retrieving the corresponding structural units. The library supports various layout methodologies, including the generation of single-device, dual-device, and multi-device layouts to produce the final chip design.

## Quick Start

Here’s a example to get you started.

Create `YourProject\config\1.yaml` under your project directory. 

```yaml
device:
  type: spiral
  straight_length: 650
  left_length: 400
  right_length: 10
  width: 1.5
  num_turns: 2
  spiral_gap: 20
  inner_gap: 220
  p: 0.8
  translation: 0.025
  resolution: 1
coupler:
  type: out_point
  length: 150
  width: 1.0
  height: 20
  coupling_length: 0
  Rc: 261
  Rc0: 261
  extend_length: 210
  ring_down: null
  resolution: 1
gap: 0.6
layer:
- 2
- 0
```

Next, you can create the `YourProject\config_edge_coupler.yaml` configuring the coupler type. (Note: To find out the bending coupler parameters below, you can test the values under `\example\bending_coupler.ipynb`)

```yaml
in:
  file: false
  type: bending
  bending_height: 10
  angle: 22.5
  length: 50.2900715898245
  width: 0.8
out:
  file: false 
  type: straight
  width: 0.8
```

Then write the `YourProject\gds.py` script to draw the layout.

```python
import gdsfactory as gf
from gdslayout.components import Cluster
import kfactory as kf

opts = kf.save_layout_options()
opts.dbu = 1e-4

Chip=gf.Component("Chip")

l_chip=2500
w_chip=2500

rect = gf.components.rectangle(size = (l_chip, w_chip), centered=True, layer=(100,2))
dicing_bot = Chip <<rect

w_side = 100
l_side = 100
width_tot = w_chip - 2 * w_side
length_tot = l_chip - 2 * l_side
chip_layout = gf.components.rectangle(size=(width_tot, length_tot), centered=True, layer=(100, 0))
chip_layout_ref = Chip << chip_layout

dicing_bot.move((l_chip/2-w_side, -w_chip/2+l_side))
chip_layout_ref.move((width_tot/2, -length_tot/2))

tot_drift=[0, 0]

y_drift = l_side
cluster = Cluster('cluster', './config')

distance = 55
y_border_dist = distance/2
text_status = True

x_auto = False
y_auto = False
#=================================== PART I =====================================

cluster.load([1])
Chip, y_len = cluster.build_layout(Chip, x_border=[100,100], y_border=[0, y_border_dist], length_tot=width_tot, y_drift=y_drift, edge_coupler_distance=distance, tot_drift=tot_drift, text=text_status, auto=(x_auto, y_auto))
y_drift += y_len

Chip_ref = gf.Component()
Chip_ref << Chip
Chip_ref.move((60, 60))

file_name = 'test'

Chip_ref.write_gds(file_name + '.gds', save_options=opts)
```

the `YourProject\test.gds` displays the single-device layout.

![test1](img/test1.png)

With the modification of the line

```python
cluster.load([1, 2])
```

The dual-device layout looks like

![test2](img/test2.png)

And the multi-device layout looks like

![test3](img/test3.png)

We provide multiple convenience in your design process.

**Layout**: You can generate multiple clusters by repeating following process.

```python
# Example:
#=================================== PART I =====================================

cluster.load([1, 2, 3, 4, 5, 6])
Chip, y_len = cluster.build_layout(Chip, x_border=[600,600], y_border=[0, y_border_dist], length_tot=width_tot, y_drift=y_drift, edge_coupler_distance=distance, tot_drift=tot_drift, text=text_status, auto=(x_auto, y_auto))
cluster._curvature()
y_drift += y_len

#=================================== PART II =====================================

cluster.load([7, 8])
Chip, y_len = cluster.build_layout(Chip, x_border=[600,600], y_border=[y_border_dist, y_border_dist], length_tot=width_tot, y_drift=y_drift, edge_coupler_distance=distance, tot_drift=tot_drift, text=text_status, auto=(x_auto, y_auto))
cluster._curvature()
y_drift += y_len

cluster.load([9, 10])
Chip, y_len = cluster.build_layout(Chip, x_border=[600,600], y_border=[y_border_dist, y_border_dist], length_tot=width_tot, y_drift=y_drift, edge_coupler_distance=distance, tot_drift=tot_drift, text=text_status, auto=(x_auto, y_auto))
cluster._curvature()
y_drift += y_len

cluster.load([11, 12])
Chip, y_len = cluster.build_layout(Chip, x_border=[600,600], y_border=[y_border_dist, y_border_dist], length_tot=width_tot, y_drift=y_drift, edge_coupler_distance=distance, tot_drift=tot_drift, text=text_status, auto=(x_auto, y_auto))
cluster._curvature()
y_drift += y_len

# ...
# You can continue on the process to put all devices in the layout.
```

![test4](img/test4.png)

**Coupler**: We provide mutiple configurations for your coupler choice.

1. No coupler at all.

```yaml
file: false 
type: straight
width: 0.8
```

2. Bending coupler to minimize the port reflection.
3. Commercial coupler region provided by the fabrication factory. (Using LiGenTech’s coupler as an example)

```yaml
file: ./AN800_LIGENTEC_BlackBoxes_v8.5.1.gds
offset: 10
length: 400
width: 0.8
cellname: "ligentecInvertedTaper_w0.8BB"
port_len: 5
```

**Device**: We provide several new devices, including the high-Q resonators, photonic crystals, and mechanical structures. These devices are open to further development and customization. For example:

```python
# Example: Futher customized Archimedean curve, trigonometric dependence of width variation of center connection waveguide part. You can check the code at /example/device_modification.ipynb

from gdslayout.device import register_device
from gdslayout.device.archimedean import SpiralArchimedean1
import numpy as np

def trigonometric_transition_function(t, y1, y2, N):
        A = (y2 - y1) / 2
        b = (y2 + y1) / 2
        return -A * np.cos(2*np.pi*t*N)+b

@register_device("spiral_archimedean1_varwidth")
class SpiralArchimedean1VarWidth(SpiralArchimedean1):
    def _compute_width_profile(self, base_width):
        N = self.points_orig.shape[0]
        w = np.full(N, float(base_width), dtype=float)
        start_id = len(self.points2)
        end_id = start_id + len(self.point_mid)
        w[start_id:end_id] = trigonometric_transition_function(
            np.linspace(0, 1, end_id - start_id), base_width, 4*base_width, 2
        )
        return w
```

![test5](img/test5.png)

**Simulation:** Our library allows for easy integration of simulation workflows. We have developed several projects, backend by ANSYS Lumerical FDTD (`lumapi`) and other simulation engines, with more details to be announced shortly.

![test6](img/test6.png)

"""
Device package for photonic component design and fabrication.

This package provides classes and functions for creating various photonic devices
including cavities, couplers, and complete structures from YAML configurations.

Main Classes:
    Structure: Main class for building complete photonic structures from YAML configs

Cavity Functions:
    ring: Create ring resonator components
    race_track: Create racetrack resonator components  
    spiral: Create spiral resonator components

Coupler Functions:
    point_coupler: Create point couplers
    symmetric_pulley_coupler: Create symmetric pulley couplers
    out_point_coupler: Create output point couplers

Usage:
    from device import Structure
    
    # Load structure from config file
    structure = Structure.from_config_id("config", "1")
    component = structure.build()
    structure.plot()
    structure.save_gds("output.gds")
"""
import gdsfactory as gf

from .basics import anchor_arc, straight

# Import cavity functions
from .cavity import (
    ring,
    race_track, 
    spiral
)

# Import coupler functions
from .coupler import (
    point_coupler,
    symmetric_pulley_coupler,
    out_point_coupler,
    port_coupler
)

# Import photonic crystal functions
from .phc import (
    PhC_optomechanics1,
    PhC_optomechanics2
)

from .mechanics import (
    optomechanical_mass
)

pdk = gf.get_active_pdk()
pdk.cells["anchor_arc"] = anchor_arc
pdk.cells["straight"] = straight

# Define what gets imported with "from device import *"
__all__ = [    
    # Cavity functions
    'ring',
    'race_track',
    'spiral',
    
    # Coupler functions
    'point_coupler',
    'symmetric_pulley_coupler', 
    'out_point_coupler',
    'port_coupler',

    'PhC_optomechanics1',
    'PhC_optomechanics2',
    'optomechanical_mass'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Photonic device design and fabrication tools"

"""
Visualization module for GaiaChat.
"""

from .plots import (
    create_hr_diagram,
    create_sky_map,
    create_velocity_plot,
    create_toomre_diagram,
    create_proper_motion_plot,
    create_interactive_hr_diagram,
    create_interactive_velocity_plot,
    get_plot_function
)

__all__ = [
    'create_hr_diagram',
    'create_sky_map', 
    'create_velocity_plot',
    'create_toomre_diagram',
    'create_proper_motion_plot',
    'create_interactive_hr_diagram',
    'create_interactive_velocity_plot',
    'get_plot_function'
]

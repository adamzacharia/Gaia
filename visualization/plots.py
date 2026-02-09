"""
GaiaChat Visualization Module

Provides astronomical visualizations for Gaia data including:
- HR (Color-Magnitude) Diagrams
- Sky maps in Galactic coordinates
- Velocity plots (V_R vs V_phi, Toomre diagrams)
- Proper motion fields
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go


# Set up matplotlib style for professional plots
plt.style.use('dark_background')
COLORS = {
    'primary': '#00D4AA',      # Cyan-green
    'secondary': '#FF6B6B',    # Coral
    'accent': '#4ECDC4',       # Teal
    'highlight': '#FFE66D',    # Yellow
    'background': '#1a1a2e',   # Dark blue
    'grid': '#333355'          # Subtle grid
}


def create_hr_diagram(
    df: pd.DataFrame,
    title: str = "Hertzsprung-Russell Diagram",
    color_by: Optional[str] = None
) -> Figure:
    """
    Create an HR (Color-Magnitude) diagram.
    
    Args:
        df: DataFrame with 'bp_rp' (color) and 'phot_g_mean_mag' (magnitude)
        title: Plot title
        color_by: Optional column to color points by
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Calculate absolute magnitude if parallax available
    if 'parallax' in df.columns:
        # M = m - 5*log10(d) + 5 = m + 5*log10(parallax) - 10
        valid = df['parallax'] > 0
        df_valid = df[valid].copy()
        df_valid['abs_mag'] = (
            df_valid['phot_g_mean_mag'] + 
            5 * np.log10(df_valid['parallax']) - 10
        )
        y_col = 'abs_mag'
        y_label = 'Absolute G Magnitude'
    else:
        df_valid = df.copy()
        y_col = 'phot_g_mean_mag'
        y_label = 'Apparent G Magnitude'
    
    # Handle color mapping
    if color_by and color_by in df_valid.columns:
        scatter = ax.scatter(
            df_valid['bp_rp'],
            df_valid[y_col],
            c=df_valid[color_by],
            cmap='plasma',
            s=10,
            alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(
            df_valid['bp_rp'],
            df_valid[y_col],
            c=COLORS['primary'],
            s=10,
            alpha=0.7
        )
    
    ax.set_xlabel('BP - RP (Color Index)', fontsize=12, color='white')
    ax.set_ylabel(y_label, fontsize=12, color='white')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    
    # Invert y-axis (brighter = smaller magnitude)
    ax.invert_yaxis()
    
    # Style
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    
    # Add annotations for stellar types
    ax.annotate('Main Sequence', xy=(1.0, 5), color=COLORS['accent'], fontsize=9)
    ax.annotate('Giants', xy=(1.5, 0), color=COLORS['secondary'], fontsize=9)
    ax.annotate('White Dwarfs', xy=(0.0, 12), color=COLORS['highlight'], fontsize=9)
    
    plt.tight_layout()
    return fig


def create_sky_map(
    df: pd.DataFrame,
    title: str = "Sky Distribution (Galactic Coordinates)",
    projection: str = "mollweide"
) -> Figure:
    """
    Create a sky map in Galactic coordinates.
    
    Args:
        df: DataFrame with 'l' (Galactic longitude) and 'b' (latitude)
        title: Plot title
        projection: Map projection ('mollweide', 'aitoff', 'hammer')
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(
        figsize=(12, 6), 
        subplot_kw={'projection': projection},
        facecolor=COLORS['background']
    )
    ax.set_facecolor(COLORS['background'])
    
    # Convert to radians, shift longitude to [-180, 180]
    l_rad = np.radians(df['l'].values)
    l_rad = np.where(l_rad > np.pi, l_rad - 2*np.pi, l_rad)
    b_rad = np.radians(df['b'].values)
    
    # Color by distance if available
    if 'distance_kpc' in df.columns:
        colors = df['distance_kpc'].values
        scatter = ax.scatter(
            -l_rad, b_rad,  # Negative for conventional orientation
            c=colors,
            cmap='viridis',
            s=5,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Distance (kpc)', shrink=0.5)
    else:
        ax.scatter(
            -l_rad, b_rad,
            c=COLORS['primary'],
            s=5,
            alpha=0.6
        )
    
    ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Add Galactic center marker
    ax.scatter([0], [0], c=COLORS['secondary'], s=100, marker='*', 
               label='Galactic Center', zorder=10)
    ax.legend(loc='lower right', facecolor=COLORS['background'])
    
    plt.tight_layout()
    return fig


def create_velocity_plot(
    df: pd.DataFrame,
    title: str = "Galactocentric Velocities"
) -> Figure:
    """
    Create V_R vs V_phi velocity plot.
    
    This is useful for identifying kinematic substructures like
    stellar streams and accreted populations.
    
    Args:
        df: DataFrame with 'V_R' and 'V_phi' columns
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    if 'V_R' not in df.columns or 'V_phi' not in df.columns:
        raise ValueError("DataFrame must contain V_R and V_phi columns")
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Color by total velocity if available
    if 'v_total' in df.columns:
        scatter = ax.scatter(
            df['V_R'],
            df['V_phi'],
            c=df['v_total'],
            cmap='plasma',
            s=20,
            alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label='Total Velocity (km/s)')
    else:
        ax.scatter(
            df['V_R'],
            df['V_phi'],
            c=COLORS['primary'],
            s=20,
            alpha=0.7
        )
    
    ax.set_xlabel('V$_R$ (km/s) - Radial', fontsize=12, color='white')
    ax.set_ylabel('V$_\\phi$ (km/s) - Azimuthal', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=0, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['grid'], linestyle='--', alpha=0.5)
    
    # Mark regions
    ax.axhline(y=220, color=COLORS['accent'], linestyle=':', alpha=0.5, 
               label='Solar V_φ (~220 km/s)')
    ax.axhline(y=0, color=COLORS['secondary'], linestyle=':', alpha=0.5,
               label='No rotation (halo)')
    
    # Disk annotation
    ax.annotate('Disk stars\n(prograde)', xy=(0, 200), 
                color=COLORS['accent'], fontsize=9, ha='center')
    ax.annotate('Retrograde\n(accreted)', xy=(0, -100), 
                color=COLORS['secondary'], fontsize=9, ha='center')
    
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.legend(loc='upper right', facecolor=COLORS['background'])
    
    plt.tight_layout()
    return fig


def create_toomre_diagram(
    df: pd.DataFrame,
    title: str = "Toomre Diagram"
) -> Figure:
    """
    Create a Toomre diagram showing V_perp vs V_phi.
    
    The Toomre diagram shows perpendicular velocity 
    (sqrt(V_R^2 + V_z^2)) vs azimuthal velocity,
    useful for separating disk and halo populations.
    
    Args:
        df: DataFrame with V_R, V_phi, V_z columns
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    required = ['V_R', 'V_phi', 'V_z']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame must contain columns: {required}")
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Calculate perpendicular velocity
    V_perp = np.sqrt(df['V_R']**2 + df['V_z']**2)
    
    # Subtract solar motion to center on LSR
    V_phi_lsr = df['V_phi'] - 220  # Approximate solar Vphi
    
    scatter = ax.scatter(
        V_phi_lsr,
        V_perp,
        c=COLORS['primary'],
        s=20,
        alpha=0.7
    )
    
    ax.set_xlabel('V$_\\phi$ - V$_{LSR}$ (km/s)', fontsize=12, color='white')
    ax.set_ylabel('$\\sqrt{V_R^2 + V_z^2}$ (km/s)', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    
    # Add velocity circles
    for v in [50, 100, 150, 200, 250]:
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        x = v * np.cos(theta)
        y = v * np.sin(theta)
        y = np.abs(y)  # Only upper half
        ax.plot(x, y, color=COLORS['grid'], linestyle='--', alpha=0.3)
        ax.annotate(f'{v}', xy=(0, v), color=COLORS['grid'], 
                   fontsize=8, alpha=0.5)
    
    # Annotate regions
    ax.annotate('Thin Disk', xy=(0, 30), color=COLORS['accent'], 
               fontsize=10, ha='center')
    ax.annotate('Thick Disk', xy=(0, 70), color=COLORS['highlight'], 
               fontsize=10, ha='center')
    ax.annotate('Halo', xy=(-150, 180), color=COLORS['secondary'], 
               fontsize=10, ha='center')
    
    ax.set_xlim(-350, 150)
    ax.set_ylim(0, 300)
    
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    
    plt.tight_layout()
    return fig


def create_proper_motion_plot(
    df: pd.DataFrame,
    title: str = "Proper Motion Distribution"
) -> Figure:
    """
    Create a proper motion vector plot.
    
    Args:
        df: DataFrame with 'pmra' and 'pmdec' columns
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Calculate total proper motion for coloring
    pm_total = np.sqrt(df['pmra']**2 + df['pmdec']**2)
    
    scatter = ax.scatter(
        df['pmra'],
        df['pmdec'],
        c=pm_total,
        cmap='plasma',
        s=20,
        alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label='Total PM (mas/yr)')
    
    ax.set_xlabel('μ$_α$cos(δ) (mas/yr)', fontsize=12, color='white')
    ax.set_ylabel('μ$_δ$ (mas/yr)', fontsize=12, color='white')
    ax.set_title(title, fontsize=14, color='white', fontweight='bold')
    
    ax.axhline(y=0, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax.axvline(x=0, color=COLORS['grid'], linestyle='--', alpha=0.5)
    
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    
    plt.tight_layout()
    return fig


def create_interactive_hr_diagram(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive HR diagram using Plotly.
    
    Args:
        df: DataFrame with 'bp_rp' and 'phot_g_mean_mag' columns
        
    Returns:
        Plotly Figure
    """
    # Calculate absolute magnitude
    df_valid = df[df['parallax'] > 0].copy()
    df_valid['abs_mag'] = (
        df_valid['phot_g_mean_mag'] + 
        5 * np.log10(df_valid['parallax']) - 10
    )
    
    fig = px.scatter(
        df_valid,
        x='bp_rp',
        y='abs_mag',
        color='distance_kpc' if 'distance_kpc' in df_valid.columns else None,
        hover_data=['source_id', 'ra', 'dec', 'parallax'],
        labels={
            'bp_rp': 'BP - RP (Color)',
            'abs_mag': 'Absolute G Magnitude',
            'distance_kpc': 'Distance (kpc)'
        },
        title='Interactive HR Diagram'
    )
    
    fig.update_yaxes(autorange='reversed')  # Brighter at top
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e'
    )
    
    return fig


def create_interactive_velocity_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive velocity plot using Plotly.
    
    Args:
        df: DataFrame with V_R, V_phi columns
        
    Returns:
        Plotly Figure
    """
    if 'V_R' not in df.columns or 'V_phi' not in df.columns:
        raise ValueError("DataFrame must contain V_R and V_phi columns")
    
    fig = px.scatter(
        df,
        x='V_R',
        y='V_phi',
        color='v_total' if 'v_total' in df.columns else None,
        hover_data=['source_id', 'distance_kpc'] if 'distance_kpc' in df.columns else ['source_id'],
        labels={
            'V_R': 'V_R (km/s)',
            'V_phi': 'V_φ (km/s)',
            'v_total': 'Total Velocity (km/s)'
        },
        title='Galactocentric Velocities'
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=220, line_dash='dot', line_color='cyan', opacity=0.5,
                  annotation_text='Solar V_φ')
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e'
    )
    
    return fig


def get_plot_function(plot_type: str):
    """
    Get the appropriate plot function for a given plot type.
    
    Args:
        plot_type: One of 'hr_diagram', 'sky_map', 'velocity_plot', 
                   'toomre_diagram', 'proper_motion'
                   
    Returns:
        Plot function
    """
    plot_functions = {
        'hr_diagram': create_hr_diagram,
        'sky_map': create_sky_map,
        'velocity_plot': create_velocity_plot,
        'toomre_diagram': create_toomre_diagram,
        'proper_motion': create_proper_motion_plot
    }
    
    if plot_type not in plot_functions:
        raise ValueError(
            f"Unknown plot type: {plot_type}. "
            f"Available: {list(plot_functions.keys())}"
        )
    
    return plot_functions[plot_type]

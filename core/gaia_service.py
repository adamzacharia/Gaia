"""
Gaia Data Service Module

Provides interface to the Gaia Archive via TAP/ADQL queries.
Includes specialized queries for dark matter research and stellar streams.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from astroquery.gaia import Gaia
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.coordinates as coord

from .config import config


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    data: pd.DataFrame
    query: str
    row_count: int
    description: str


class GaiaService:
    """
    Service class for accessing Gaia DR3 data.
    
    Provides high-level methods for common astronomical queries,
    with special focus on stellar kinematics and dark matter research.
    """
    
    def __init__(self):
        """Initialize the Gaia service."""
        self.table = config.gaia_table
        self.max_results = config.max_query_results
        self.default_results = config.default_query_results
        
        # Set up Galactocentric frame (matching Astropy defaults for consistency)
        self.galcen_frame = Galactocentric()
        
        # Cache for last query results
        self._last_result: Optional[QueryResult] = None
    
    def execute_adql(self, query: str) -> QueryResult:
        """
        Execute an ADQL query against the Gaia Archive.
        
        Args:
            query: ADQL query string
            
        Returns:
            QueryResult with data and metadata
        """
        try:
            job = Gaia.launch_job(query)
            result = job.get_results()
            df = result.to_pandas()
            
            query_result = QueryResult(
                data=df,
                query=query,
                row_count=len(df),
                description=f"Query returned {len(df)} rows"
            )
            self._last_result = query_result
            return query_result
            
        except Exception as e:
            raise RuntimeError(f"ADQL query failed: {str(e)}")
    
    def search_cone(
        self, 
        ra: float, 
        dec: float, 
        radius: float = 1.0,
        limit: Optional[int] = None
    ) -> QueryResult:
        """
        Perform a cone search around a sky position.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees  
            radius: Search radius in degrees
            limit: Maximum number of results
            
        Returns:
            QueryResult with stars in the cone
        """
        limit = limit or self.default_results
        
        query = f"""
        SELECT TOP {limit}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, radial_velocity,
            phot_g_mean_mag, bp_rp,
            l, b
        FROM {self.table}
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        AND parallax IS NOT NULL
        AND parallax > 0
        ORDER BY parallax DESC
        """
        
        result = self.execute_adql(query)
        result.description = f"Cone search: {radius}° around (RA={ra}°, Dec={dec}°)"
        return result
    
    def search_solar_neighborhood(
        self, 
        distance_pc: float = 100,
        limit: Optional[int] = None
    ) -> QueryResult:
        """
        Search for stars in the solar neighborhood.
        
        Args:
            distance_pc: Maximum distance in parsecs
            limit: Maximum number of results
            
        Returns:
            QueryResult with nearby stars
        """
        limit = limit or self.default_results
        min_parallax = 1000.0 / distance_pc  # parallax in mas
        
        query = f"""
        SELECT TOP {limit}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, radial_velocity,
            phot_g_mean_mag, bp_rp,
            l, b, 
            ruwe
        FROM {self.table}
        WHERE parallax > {min_parallax}
        AND parallax_over_error > 10
        AND ruwe < 1.4
        ORDER BY parallax DESC
        """
        
        result = self.execute_adql(query)
        result.description = f"Solar neighborhood within {distance_pc} pc"
        return result
    
    def search_hypervelocity_stars(
        self,
        distance_kpc: float = 5.0,
        min_velocity_kms: float = 300,
        limit: Optional[int] = None
    ) -> QueryResult:
        """
        Search for hypervelocity star candidates.
        
        These are stars with total velocities high enough to potentially
        escape the Galaxy, often ejected from the Galactic center.
        
        Args:
            distance_kpc: Maximum distance in kiloparsecs
            min_velocity_kms: Minimum total velocity in km/s
            limit: Maximum results
            
        Returns:
            QueryResult with hypervelocity candidates
        """
        limit = limit or min(500, self.default_results)
        min_parallax = 1.0 / distance_kpc  # parallax in mas for kpc distance
        
        # We'll compute velocity after fetching; filter on high proper motion first
        # Rough cut: 300 km/s at ~1 kpc corresponds to ~63 mas/yr proper motion
        min_pm = 20  # mas/yr, conservative cut
        
        query = f"""
        SELECT TOP {limit}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, pmra_error, pmdec_error,
            radial_velocity, radial_velocity_error,
            phot_g_mean_mag, bp_rp,
            l, b
        FROM {self.table}
        WHERE parallax > {min_parallax}
        AND parallax_over_error > 5
        AND radial_velocity IS NOT NULL
        AND SQRT(pmra*pmra + pmdec*pmdec) > {min_pm}
        ORDER BY SQRT(pmra*pmra + pmdec*pmdec) DESC
        """
        
        result = self.execute_adql(query)
        
        # Add velocity calculations
        if len(result.data) > 0:
            result.data = self._add_galactic_velocities(result.data)
            # Filter by total velocity
            result.data = result.data[
                result.data['v_total'] > min_velocity_kms
            ].reset_index(drop=True)
            result.row_count = len(result.data)
        
        result.description = (
            f"Hypervelocity candidates within {distance_kpc} kpc, "
            f"v > {min_velocity_kms} km/s"
        )
        return result
    
    def search_stellar_stream(
        self,
        stream_name: str,
        limit: Optional[int] = None
    ) -> QueryResult:
        """
        Search for stars potentially belonging to known stellar streams.
        
        Supports: Nyx, Gaia-Sausage-Enceladus (GSE), Helmi, Sequoia
        
        Args:
            stream_name: Name of the stellar stream
            limit: Maximum results
            
        Returns:
            QueryResult with stream candidates
        """
        limit = limit or self.default_results
        stream_name = stream_name.lower()
        
        # Define stream selection criteria based on kinematics
        stream_criteria = self._get_stream_criteria(stream_name)
        
        if stream_criteria is None:
            raise ValueError(
                f"Unknown stream: {stream_name}. "
                f"Supported: Nyx, GSE, Gaia-Sausage-Enceladus, Helmi, Sequoia"
            )
        
        query = f"""
        SELECT TOP {limit}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, radial_velocity,
            phot_g_mean_mag, bp_rp,
            l, b
        FROM {self.table}
        WHERE parallax > 0.2
        AND parallax_over_error > 5
        AND radial_velocity IS NOT NULL
        {stream_criteria['adql_filter']}
        ORDER BY parallax DESC
        """
        
        result = self.execute_adql(query)
        
        # Add velocities and apply kinematic selection
        if len(result.data) > 0:
            result.data = self._add_galactic_velocities(result.data)
            
            # Apply velocity-based selection if defined
            if 'velocity_filter' in stream_criteria:
                result.data = stream_criteria['velocity_filter'](result.data)
                result.row_count = len(result.data)
        
        result.description = stream_criteria['description']
        return result
    
    def search_accreted_halo(
        self,
        retrograde_only: bool = False,
        limit: Optional[int] = None
    ) -> QueryResult:
        """
        Search for accreted halo stars.
        
        These are stars that likely originated from dwarf galaxies
        that merged with the Milky Way, identifiable by their kinematics.
        
        Args:
            retrograde_only: If True, only return retrograde orbit stars
            limit: Maximum results
            
        Returns:
            QueryResult with accreted halo candidates
        """
        limit = limit or self.default_results
        
        query = f"""
        SELECT TOP {limit * 2}
            source_id, ra, dec, parallax, parallax_error,
            pmra, pmdec, radial_velocity,
            phot_g_mean_mag, bp_rp,
            l, b
        FROM {self.table}
        WHERE parallax > 0.5
        AND parallax_over_error > 5
        AND radial_velocity IS NOT NULL
        AND ABS(b) > 30
        ORDER BY SQRT(pmra*pmra + pmdec*pmdec) DESC
        """
        
        result = self.execute_adql(query)
        
        if len(result.data) > 0:
            result.data = self._add_galactic_velocities(result.data)
            
            # Select halo stars: high velocity dispersion, low rotation
            # Halo selection: |V_phi| < 50 km/s or V_phi < -50 (counter-rotating)
            if retrograde_only:
                result.data = result.data[
                    result.data['V_phi'] < -50
                ].head(limit).reset_index(drop=True)
            else:
                result.data = result.data[
                    (np.abs(result.data['V_phi']) < 100) |
                    (result.data['V_phi'] < -50)
                ].head(limit).reset_index(drop=True)
            
            result.row_count = len(result.data)
        
        desc = "Accreted halo stars"
        if retrograde_only:
            desc += " (retrograde orbits only)"
        result.description = desc
        return result
    
    def _get_stream_criteria(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """Get selection criteria for known stellar streams."""
        
        streams = {
            "nyx": {
                "adql_filter": "AND b > -30 AND b < 30",  # Near disk plane
                "description": (
                    "Nyx stream candidates - prograde structure near the disk, "
                    "discovered by Necib et al. 2020"
                ),
                "velocity_filter": lambda df: df[
                    (df['V_phi'] > 100) &  # Prograde
                    (df['V_R'] > 100) & (df['V_R'] < 200)  # Radially biased
                ].reset_index(drop=True)
            },
            "gse": {
                "adql_filter": "",
                "description": (
                    "Gaia-Sausage-Enceladus candidates - major merger remnant "
                    "with highly radial orbits"
                ),
                "velocity_filter": lambda df: df[
                    (np.abs(df['V_phi']) < 50) &  # Low rotation
                    (np.abs(df['V_R']) > 150)  # Highly radial
                ].reset_index(drop=True)
            },
            "gaia-sausage-enceladus": {
                "adql_filter": "",
                "description": (
                    "Gaia-Sausage-Enceladus candidates - major merger remnant "
                    "with highly radial orbits"
                ),
                "velocity_filter": lambda df: df[
                    (np.abs(df['V_phi']) < 50) &
                    (np.abs(df['V_R']) > 150)
                ].reset_index(drop=True)
            },
            "helmi": {
                "adql_filter": "",
                "description": "Helmi stream candidates - tidally disrupted dwarf galaxy",
                "velocity_filter": lambda df: df[
                    (df['V_z'] > 150) | (df['V_z'] < -150)  # High vertical velocity
                ].reset_index(drop=True)
            },
            "sequoia": {
                "adql_filter": "",
                "description": "Sequoia candidates - retrograde merger remnant",
                "velocity_filter": lambda df: df[
                    (df['V_phi'] < -150)  # Strongly retrograde
                ].reset_index(drop=True)
            }
        }
        
        return streams.get(stream_name)
    
    def _add_galactic_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Galactocentric velocity components to dataframe.
        
        Computes V_R (radial), V_phi (azimuthal), V_z (vertical)
        in the Galactocentric frame.
        """
        if len(df) == 0:
            return df
        
        try:
            # Create SkyCoord from Gaia data
            coords = SkyCoord(
                ra=df['ra'].values * u.deg,
                dec=df['dec'].values * u.deg,
                distance=(1000 / df['parallax'].values) * u.pc,
                pm_ra_cosdec=df['pmra'].values * u.mas/u.yr,
                pm_dec=df['pmdec'].values * u.mas/u.yr,
                radial_velocity=df['radial_velocity'].values * u.km/u.s,
                frame='icrs'
            )
            
            # Transform to Galactocentric
            galcen = coords.transform_to(self.galcen_frame)
            
            # Get cylindrical velocities
            # v_R (radial, positive outward)
            # v_phi (azimuthal, positive in direction of Galactic rotation)
            # v_z (vertical, positive toward NGP)
            
            galcen_cyl = galcen.represent_as('cylindrical')
            vel_cyl = galcen.velocity.represent_as(
                coord.CylindricalDifferential,
                base=galcen_cyl
            )
            
            df = df.copy()
            df['V_R'] = vel_cyl.d_rho.to(u.km/u.s).value
            df['V_phi'] = (galcen_cyl.rho * vel_cyl.d_phi).to(
                u.km/u.s, equivalencies=u.dimensionless_angles()
            ).value
            df['V_z'] = vel_cyl.d_z.to(u.km/u.s).value
            
            # Total velocity
            df['v_total'] = np.sqrt(
                df['V_R']**2 + df['V_phi']**2 + df['V_z']**2
            )
            
            # Distance in kpc
            df['distance_kpc'] = 1.0 / df['parallax']
            
        except Exception as e:
            print(f"Warning: Could not compute velocities: {e}")
            # Add empty columns
            for col in ['V_R', 'V_phi', 'V_z', 'v_total', 'distance_kpc']:
                df[col] = np.nan
        
        return df
    
    def get_last_result(self) -> Optional[QueryResult]:
        """Return the most recent query result."""
        return self._last_result
    
    def build_custom_adql(
        self,
        columns: List[str],
        conditions: List[str],
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> str:
        """
        Build a custom ADQL query from components.
        
        Args:
            columns: List of column names to select
            conditions: List of WHERE conditions
            limit: Maximum results
            order_by: ORDER BY clause
            
        Returns:
            ADQL query string
        """
        limit = limit or self.default_results
        
        select_clause = ", ".join(columns) if columns else "*"
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT TOP {limit}
            {select_clause}
        FROM {self.table}
        WHERE {where_clause}
        """
        
        if order_by:
            query += f"\nORDER BY {order_by}"
        
        return query


# Global service instance
gaia_service = GaiaService()

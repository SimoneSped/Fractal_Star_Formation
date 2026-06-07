# General Imports
import os
import sys
import numpy as np

# Astronomy Specific Imports
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
import astropy.units as u
from astropy.io import fits


def remove_region(data, wcs, longitude_min, longitude_max, latitude_min, latitude_max):
    
    min_coord_remove = SkyCoord(longitude_min, latitude_min, frame='galactic', unit=u.deg)
    max_coord_remove = SkyCoord(longitude_max, latitude_max, frame='galactic', unit=u.deg)

    min_pixel_remove = wcs[0].world_to_pixel(min_coord_remove)
    max_pixel_remove = wcs[0].world_to_pixel(max_coord_remove)

    # Create a mask based on these galactic coordinates
    remove_mask = np.zeros(data.shape, dtype=bool)

    # Set True in the mask for the specified region based on pixel indices
    y_min, x_min = int(np.floor(min_pixel_remove[1])), int(np.floor(min_pixel_remove[0]))
    y_max, x_max = int(np.ceil(max_pixel_remove[1])), int(np.ceil(max_pixel_remove[0]))

    # Apply the mask to the region within the pixel boundaries
    remove_mask[y_min:y_max, x_max:x_min] = True
    
    # Apply the mask to the original data to set the region to NaN or another value
    data[remove_mask] = np.nan  # Replace with np.nan to exclude the data

    return data

from pathlib import Path

def derive_density_maps(
    input_path: str = "Lombardi_maps/planck_herschel.fits.gz",
    remove_regions: bool = True,
):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    planck_herschel_fits_file = (
        PROJECT_ROOT / "inputs" / Path(input_path)
    )

    if not planck_herschel_fits_file.exists():
        raise FileNotFoundError(
            f"FITS file not found: {planck_herschel_fits_file}"
        )

    print(f"Using: {planck_herschel_fits_file}")

    hdu_herschel_fits = fits.open(planck_herschel_fits_file)[0]
    
    # Constructt WCS and image data
    wcs = WCS(hdu_herschel_fits.header)

    image_data = hdu_herschel_fits.data[0]

    # Replace NaNs and Infs with some valid value, e.g., zero or the median
    tau = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Define constants
    # TO-DO: differentiate between Orion A and B 

    # Orion A: 206 ≤ l ≤ 217, −21 ≤ b ≤ −17
    # Orion B: 203 ≤ l ≤ 210, −17 ≤ b ≤ −12

    # Step 3: Apply the formula to calculate A_k (Lomabrdi et al)

    gamma_orion_A = 2640  # mag
    delta_orion_A = 0.012  # mag, Offset for Orion A

    gamma_orion_B = 3460  # mag
    delta_orion_B = -0.001  # mag, Offset for Orion B

    from astropy.wcs.utils import pixel_to_skycoord

    ny, nx = tau.shape
    y, x = np.mgrid[0:ny, 0:nx]
    coords = pixel_to_skycoord(x, y, wcs)

    l_min_A, l_max_A = 206, 217
    b_min_A, b_max_A = -21, -17

    l_min_B, l_max_B = 203, 210
    b_min_B, b_max_B = -17, -12

    # Build masks in Galactic coordinates
    mask_A = (coords.l.deg >= l_min_A) & (coords.l.deg <= l_max_A) & (coords.b.deg >= b_min_A) & (coords.b.deg <= b_max_A)
    mask_B = (coords.l.deg >= l_min_B) & (coords.l.deg <= l_max_B) & (coords.b.deg >= b_min_B) & (coords.b.deg <= b_max_B)

    A_k = np.zeros_like(tau)
    A_k = gamma_orion_A * tau + delta_orion_A
    A_k[mask_A] = gamma_orion_A * tau[mask_A] + delta_orion_A
    A_k[mask_B] = gamma_orion_B * tau[mask_B] + delta_orion_B

    # A_k to A_V
    A_V = A_k/0.112

    # N(H2)
    N_H2 = 0.93e21 * np.array(A_V , dtype=np.float64)

    wcs_2d = wcs.dropaxis(2)

    center_OA = SkyCoord(l=211.5*u.deg, b=-19*u.deg, frame='galactic')  # center of Orion A
    size_OA = (4*u.deg, 11*u.deg)  # (height in latitude, width in longitude)

    wcs_2d = wcs.dropaxis(2)

    cutout_OA = Cutout2D(
        data=N_H2,
        position=center_OA,
        size=size_OA,
        wcs=wcs_2d,             # your original WCS
        mode='partial',      # keeps WCS valid even if edges fall outside
        fill_value=np.nan
    )
    N_H2_OA = cutout_OA.data
    wcs_OA = cutout_OA.wcs

    center_OB = SkyCoord(206.5*u.deg, -14.5*u.deg, frame='galactic')
    size_OB = (5*u.deg, 7*u.deg)

    cutout_OB = Cutout2D(
        data=N_H2,
        position=center_OB,
        size=size_OB,
        wcs=wcs_2d,             # your original WCS
        mode='partial',      # keeps WCS valid even if edges fall outside
        fill_value=np.nan
    )

    N_H2_OB = cutout_OB.data
    wcs_OB = cutout_OB.wcs

    # Define regions to remove as a list of dictionaries
    regions_to_remove = [
        # Orion A regions
        {"longitude_min": 208, "longitude_max": 211, "latitude_min": -17.9, "latitude_max": -16.9},
        {"longitude_min": 214.5, "longitude_max": 216.5, "latitude_min": -18, "latitude_max": -17},
        {"longitude_min": 206, "longitude_max": 206.5, "latitude_min": -19.5, "latitude_max": -19},

        # Orion B regions
        {"longitude_min": 208, "longitude_max": 210, "latitude_min": -15, "latitude_max": -12},
        {"longitude_min": 209, "longitude_max": 210, "latitude_min": -16.4, "latitude_max": -15.5},
        {"longitude_min": 203, "longitude_max": 205, "latitude_min": -12.3, "latitude_max": -11.5},
        {"longitude_min": 204.5, "longitude_max": 205.5, "latitude_min": -12.2, "latitude_max": -11.5},
        {"longitude_min": 205.8, "longitude_max": 207, "latitude_min": -13.5, "latitude_max": -12.5},
    ]

    # Loop through each region and remove it from N_H2
    if remove_regions:
        for region in regions_to_remove:
            N_H2 = remove_region(
                N_H2,
                wcs,
                region["longitude_min"],
                region["longitude_max"],
                region["latitude_min"],
                region["latitude_max"]
            )

    N_H2 = np.nan_to_num(N_H2, nan=0.0, posinf=0.0, neginf=0.0)
    N_H2_OA = np.nan_to_num(N_H2_OA, nan=0.0, posinf=0.0, neginf=0.0)
    N_H2_OB = np.nan_to_num(N_H2_OB, nan=0.0, posinf=0.0, neginf=0.0)
    
    return N_H2, N_H2_OA, N_H2_OB, wcs, wcs_OA, wcs_OB

def convert_to_mass(N_H2):
    """
    Converts a column density map (N_H2) into a mass map (M_H2).
    This function calculates the mass map of molecular hydrogen (H2) 
    from the given column density map (N_H2). It takes into account 
    the pixel scale, distance to the object, and physical constants 
    to perform the conversion. Negative and invalid values in the 
    resulting mass map are masked out.
    Parameters:
    -----------
    N_H2 : array-like
        The column density map of molecular hydrogen (H2) in units 
        of particles per square centimeter.
    Returns:
    --------
    M_H2 : numpy.ndarray
        The mass map of molecular hydrogen (H2) in units of solar 
        masses per pixel. Negative and invalid values are replaced 
        with zero.
    """

    # calculate mass map
    pixel_scale = 0.00417
    distance = 412.0 # pc
    radians = 180.0/np.pi #conversion factor: rad in deg
    rad_per_px = pixel_scale/radians
    pc_per_px = np.sin(rad_per_px)*distance
    
    pc2_per_px = pc_per_px**2

    # Conversion factor: 1 parsec (pc) to centimeters (cm)
    cm_per_pc = 3.085677581e18 
    cm2_per_px = pc2_per_px * (cm_per_pc ** 2) 

    m_p = 1.67262192369e-27  # mass of proton (kg)
    
    # M_H2 = np.array(N_H2, dtype=np.float64) * m_p / 1.98847e30
    mu = 2.8 # mean molecular weight corrected for the helium abundance

    M_H2 = np.array(N_H2, dtype=np.float64) * mu * m_p / 1.98847e30

    # M_H2 = M_H2*area # check here!
    M_H2 = M_H2 * 8.56e33 # * cm2_per_px

    # Assuming M_H2_clean is already defined
    M_H2_no_nan = np.nan_to_num(M_H2, nan=0.0, posinf=0.0, neginf=0.0)

    # Masking out negative values
    M_H2_clean = np.where(M_H2_no_nan > 0, M_H2_no_nan, 0)

    return M_H2_clean


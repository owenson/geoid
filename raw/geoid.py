#!/usr/bin/env python3
import rasterio
import numpy as np
from rasterio.transform import rowcol


def get_geoid_height(geotiff_path, lat, lon):
    """
    Get geoid height from EGM2008 GeoTIFF for a specific lat/lon
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the EGM2008 GeoTIFF file
    lat : float
        Latitude in decimal degrees (-90 to 90)
    lon : float
        Longitude in decimal degrees (-180 to 180)
        
    Returns:
    --------
    float
        Geoid height in meters
    """
    try:
        # Open the GeoTIFF file
        with rasterio.open(geotiff_path) as src:
            # Convert lat/lon to pixel coordinates
            row, col = rowcol(src.transform, lon, lat)
            
            # Check if the coordinates are within bounds
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                raise ValueError(f"Coordinates ({lat}, {lon}) are outside the bounds of the GeoTIFF")
            
            # Read the geoid height value at that location
            # Most geoid GeoTIFFs have a single band
            window = ((row, row+1), (col, col+1))
            geoid_height = src.read(1, window=window)[0, 0]
            
            # Handle nodata values if they exist
            if geoid_height == src.nodata:
                raise ValueError(f"No data available at coordinates ({lat}, {lon})")
                
            return geoid_height
            
    except Exception as e:
        print(f"Error extracting geoid height: {e}")
        raise


def interpolated_geoid_height(geotiff_path, lat, lon):
    """
    Get bilinearly interpolated geoid height for a more precise result
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the EGM2008 GeoTIFF file
    lat : float
        Latitude in decimal degrees (-90 to 90)
    lon : float
        Longitude in decimal degrees (-180 to 180)
        
    Returns:
    --------
    float
        Interpolated geoid height in meters
    """
    with rasterio.open(geotiff_path) as src:
        # Get exact pixel coordinates (floating point)
        row, col = rowcol(src.transform, lon, lat)
        
        # Get the four surrounding pixels
        row_floor, col_floor = int(np.floor(row)), int(np.floor(col))
        
        # Make sure we're within bounds
        if (row_floor < 0 or row_floor >= src.height - 1 or 
            col_floor < 0 or col_floor >= src.width - 1):
            raise ValueError(f"Cannot interpolate at coordinates ({lat}, {lon}): too close to edge")
            
        # Get the fractional parts for interpolation weights
        row_frac = row - row_floor
        col_frac = col - col_floor
        
        # Read the 2x2 window of pixels surrounding our point
        window = ((row_floor, row_floor+2), (col_floor, col_floor+2))
        data = src.read(1, window=window)
        
        # Get the four corner values
        z00 = data[0, 0]  # Upper left
        z01 = data[0, 1]  # Upper right
        z10 = data[1, 0]  # Lower left
        z11 = data[1, 1]  # Lower right
        
        # Bilinear interpolation formula
        height = (z00 * (1 - row_frac) * (1 - col_frac) +
                  z01 * (1 - row_frac) * col_frac +
                  z10 * row_frac * (1 - col_frac) +
                  z11 * row_frac * col_frac)
                  
        return height


if __name__ == "__main__":
    # Example usage
    geotiff_path = "us_egm2008_europe.tif"
    lat = 51.8  # New York City latitude
    lon = -1.2  # New York City longitude
    
    try:
        # Get nearest-pixel height
        height = get_geoid_height(geotiff_path, lat, lon)
        print(f"Geoid height at ({lat}, {lon}): {height:.3f} meters")
        
        # Get interpolated height for higher precision
        interp_height = interpolated_geoid_height(geotiff_path, lat, lon)
        print(f"Interpolated geoid height: {interp_height:.3f} meters")
    except Exception as e:
        print(f"Error: {e}")

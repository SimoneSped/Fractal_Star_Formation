import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import perimeter, euler_number 

def compute_minkowski_functionals(data, threshold_min=1e20, threshold_max=1e22, thresholds=None):
    """
    Compute the standard Minkowski functionals (area, perimeter, Euler characteristic) 
    and fractal dimension for a given 2D dataset across a range of thresholds.

    Parameters:
    ----------
    data : numpy.ndarray
        A 2D array representing the map dataset for which Minkowski functionals are computed.
    threshold_min : float, optional
        The minimum threshold value for generating the binary masks. Default is 1e20.
    threshold_max : float, optional
        The maximum threshold value for generating the binary masks. Default is 1e22.
    thresholds : numpy.ndarray or None, optional
        An array of threshold values to use. If None, thresholds are generated 
        logarithmically between `threshold_min` and `threshold_max`.

    Returns:
    -------
    dict
        A dictionary containing the following keys:
        - "thresholds": The array of threshold values used.
        - "areas": The computed areas (v0) for each threshold.
        - "perimeters": The computed perimeters (v1) for each threshold.
        - "euler_chars": The computed Euler characteristics (v2) for each threshold.
        - "fractal_dimension": The computed fractal dimensions based on the log-log 
          relationship between area and perimeter.

    Notes:
    -----
    - Minkowski functionals are geometric measures used to describe the morphology 
      of structures in 2D or 3D datasets. 
    - The fractal dimension is derived from the relationship between the logarithm 
      of the perimeter and the logarithm of the area.
    - The relation D = (2 * log(perimeter)) / (log(area)) is used to compute the fractal dimension.
    """
    # If thresholds are not provided, generate them using a logarithmic scale
    if thresholds is None:
        thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Store Minkowski Functional values
    areas = []
    perimeters = []
    euler_chars = []

    # Process each threshold
    for threshold in thresholds:
        # Create binary mask
        mask = data >= threshold

        # Compute Area (v0)
        area = np.sum(mask)
        areas.append(area)

        # Compute Perimeter (v1)
        mask_perim = perimeter(mask)
        perimeters.append(mask_perim)

        # Compute Euler Characteristic (v2)
        euler_char = euler_number(mask)
        euler_chars.append(euler_char)

    # Convert to log scale for fractal dimension analysis
    log_areas = np.log10(areas)
    log_perimeters = np.log10(np.array(perimeters))

    # Fit a linear model to log_areas and log_perimeters to retrieve D
    # valid_indices = (log_areas > -np.inf) & (log_perimeters > -np.inf)
    # filtered_log_areas = log_areas[valid_indices]
    # filtered_log_perimeters = log_perimeters[valid_indices]

    # Perform linear regression
    coefficients = np.polyfit(log_perimeters, log_areas, 1)
    slope, intercept = coefficients
    # Calculate correlation coefficient between log_perimeters and log_areas
    corr_coef = np.corrcoef(log_perimeters, log_areas)[0, 1]
    # print(f"Correlation coefficient: {corr_coef:.4f}")

    # Fractal dimension D is twice the slope
    m = 2/slope
    # print(2/slope)

    D = (2 * log_perimeters) / (log_areas)

    # Estimate uncertainties
    sigma_A = np.sqrt(areas)  # rough Poisson error
    sigma_P = 0.5 * np.sqrt(perimeters)  # rough guess; adjust based on your system

    # Convert to np arrays
    areas = np.array(areas)
    perimeters = np.array(perimeters)

    # Partial derivatives
    dD_dP = 2 / (np.log(10) * perimeters * log_areas)
    dD_dA = -2 * log_perimeters / (np.log(10) * areas * log_areas**2)

    # Propagate uncertainties
    sigma_D = np.sqrt((dD_dP * sigma_P)**2 + (dD_dA * sigma_A)**2)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(log_perimeters, log_areas, label="Data Points", color="blue")
    # plt.plot(log_perimeters, slope * log_perimeters + intercept, label=f"Fit with D = {2/slope:.2f}", color="red")
    # plt.xlabel("log(P)", fontsize=20)
    # plt.ylabel("log(A)", fontsize=20)
    # # plt.title("Log-Log Plot of Areas vs Perimeters with Linear Fit")
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.show()

    return {
        "thresholds": thresholds,
        "areas": areas,
        "perimeters": perimeters,
        "euler_chars": euler_chars,
        "fractal_dimension": D,
        "sigma_D": sigma_D
    }

def horizontal_marching_minkowski_functionals(data, n_regions, threshold_min=6e21, threshold_max=2.5e22):
    """
    Calculates the Minkowski functionals for the map divided along the x-axis.

    Parameters:
        data (ndarray): 2D array of the map data.
        region_name (str): Name of the region (for labeling purposes).
        threshold_min (float): Minimum threshold value.
        threshold_max (float): Maximum threshold value.
        n_regions (int): Number of sections to divide into along the y-axis.

    Returns:
        avg_log_areas (ndarray): Averaged log areas over all regions.
        avg_log_perimeters (ndarray): Averaged log perimeters over all regions.
    """
    thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Get the shape of the data
    _, width = data.shape

    # Determine step size for slicing along the y-axis
    w_step = width // n_regions

    fractal_dimensions = []

    for i in range(n_regions):
        # Slice the region along the y-axis
        region = data[:, i * w_step:(i + 1) * w_step]

        fractal_dimensions.append([standard_minkowski_functionals(region, threshold_min=threshold_min, threshold_max=threshold_max)["fractal_dimension"]])

    return fractal_dimensions, thresholds

def vertical_marching_minkowski_functionals(data, n_regions, threshold_min=6e21, threshold_max=2.5e22):
    """
    Calculates the Minkowski functionals for the map divided along the y-axis.

    Parameters:
        data (ndarray): 2D array of the map data.
        region_name (str): Name of the region (for labeling purposes).
        threshold_min (float): Minimum threshold value.
        threshold_max (float): Maximum threshold value.
        n_regions (int): Number of sections to divide into along the y-axis.

    Returns:
        avg_log_areas (ndarray): Averaged log areas over all regions.
        avg_log_perimeters (ndarray): Averaged log perimeters over all regions.
    """
    thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Get the shape of the data
    height, _ = data.shape

    # Determine step size for slicing along the y-axis
    h_step = height // n_regions

    fractal_dimensions = []

    for i in range(n_regions):
        # Slice the region along the y-axis
        region = data[i * h_step:(i + 1) * h_step, :]

        fractal_dimensions.append([standard_minkowski_functionals(region, threshold_min=threshold_min, threshold_max=threshold_max)["fractal_dimension"]])

    return fractal_dimensions, thresholds

def plane_corrected_minkowski_functionals(data, threshold_min = 1e20, threshold_max=1e22, thresholds = None):

    if thresholds is None:
        thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Store Minkowski Functional values
    areas = []
    perimeters = []
    euler_chars = []

    # Process each threshold
    for threshold in thresholds:
        # Create binary mask
        mask = data >= threshold

        # Compute Area (v0)
        area = np.sum(mask)
        areas.append(area)

        # Compute Perimeter (v1) with correction for the plane dimension
        perim = perimeter(mask)/4
        perimeters.append(perim)

        # Compute Euler Characteristic (v2)
        euler_char = euler_number(mask)
        euler_chars.append(euler_char)

    # Convert to log scale for fractal dimension analysis
    log_areas = np.log10(areas)
    log_perimeters = np.log10(perimeters)

    D = 2*log_perimeters/log_areas

    return {
        "thresholds": thresholds,
        "areas": areas,
        "perimeters": perimeters,
        "euler_chars": euler_chars,
        "fractal_dimension": D
    }

def circle_corrected_minkowski_functionals(data, threshold_min = 1e20, threshold_max=1e22, thresholds = None):

    if thresholds is None:
        thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Store Minkowski Functional values
    areas = []
    perimeters = []
    euler_chars = []

    # Process each threshold
    for threshold in thresholds:
        # Create binary mask
        mask = data >= threshold

        # Compute Area (v0)
        area = np.sum(mask)
        areas.append(area)

        # Compute Perimeter (v1)
        perim = perimeter(mask)
        perimeters.append(perim)

        # Compute Euler Characteristic (v2)
        euler_char = euler_number(mask)
        euler_chars.append(euler_char)

    # Convert to log scale for fractal dimension analysis
    log_areas = np.log10(areas)
    log_perimeters = np.log10(perimeters)

    D = 2*(log_perimeters - np.log10(2*np.sqrt(np.pi)))/(log_areas)

    return {
        "thresholds": thresholds,
        "areas": areas,
        "perimeters": perimeters,
        "euler_chars": euler_chars,
        "fractal_dimension": D
    }

def intercept_corrected_minkowski_functionals(data, threshold_min=1e20, threshold_max=1e22, thresholds=None):
    if thresholds is None:
        thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), 100)

    # Store Minkowski Functional values
    areas = []
    perimeters = []
    euler_chars = []

    # Process each threshold
    for threshold in thresholds:
        # Create binary mask
        mask = data >= threshold

        # Compute Area (v0)
        area = np.sum(mask)
        areas.append(area)

        # Compute Perimeter (v1) with correction for the intercept
        perim = perimeter(mask)
        perimeters.append(perim)

        # Compute Euler Characteristic (v2)
        euler_char = euler_number(mask)
        euler_chars.append(euler_char)

    # Convert to log scale for fractal dimension analysis
    log_areas = np.log10(areas)
    log_perimeters = np.log10(perimeters)

    # Fit a linear model to obtain the intercept
    # Filter out zero or negative areas and perimeters to avoid log10 issues
    valid_indices = (log_areas > -np.inf) & (log_perimeters > -np.inf)
    filtered_log_areas = log_areas[valid_indices]
    filtered_log_perimeters = log_perimeters[valid_indices]

    # Fit a linear model to obtain the intercept
    coefficients = np.polyfit(filtered_log_perimeters, filtered_log_areas, 1)
    slope, intercept = coefficients

    # Plot log_areas vs log_perimeters and the linear fit
    plt.figure(figsize=(8, 6))
    plt.scatter(log_perimeters, log_areas, label="Data Points", color="blue")
    plt.plot(log_perimeters, slope * log_perimeters + intercept, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}", color="red")
    plt.xlabel("log(Perimeters)")
    plt.ylabel("log(Areas)")
    plt.title("Log-Log Plot of Areas vs Perimeters with Linear Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    log_areas = log_areas - intercept  # Adjust log_areas by the intercept

    # Plot log_areas vs log_perimeters and the linear fit
    plt.figure(figsize=(8, 6))
    plt.scatter(log_perimeters, log_areas, label="Data Points", color="blue")
    plt.plot(log_perimeters, slope * log_perimeters, label=f"Fit: y = {slope:.2f}x", color="red")
    plt.xlabel("log(Perimeters)")
    plt.ylabel("log(Areas)")
    plt.title("Log-Log Plot of Areas vs Perimeters with Linear Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    D = 2 * (log_perimeters) / (log_areas)
    # D = (log_perimeters) / (log_areas - np.abs(intercept)) # review this, but it seems right like this.

    return {
        "thresholds": thresholds,
        "areas": areas,
        "perimeters": perimeters,
        "euler_chars": euler_chars,
        "fractal_dimension": D
    }
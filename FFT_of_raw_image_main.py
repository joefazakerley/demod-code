# 28/11/24 
## Ford's set-up with MSE spectrum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.fft import fft, ifft, fftfreq, fft2, ifft2, fftshift
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import peak_local_max
from scipy.integrate import quad, simpson
from scipy.signal import hilbert, find_peaks
from scipy.signal.windows import tukey, hamming
from scipy.ndimage import gaussian_filter, median_filter
from scipy import interpolate
from scipy.stats import linregress
import math
from datetime import datetime
import pandas as pd
import cv2
from scipy.interpolate import CubicSpline, UnivariateSpline
import seaborn as sns

def read_image(filepath, width, height):
# Read raw image data
    with open(filepath, "rb") as image_file:
        raw_data = np.fromfile(image_file, dtype=np.uint8)
        
        # Ensure dimensions are correct
        if raw_data.size != width * height:
            raise ValueError(f"Size mismatch: Cannot reshape array of size {raw_data.size} into shape ({height}, {width})")
        
        # Reshape raw image data
        raw_image = raw_data.reshape((height, width))
        raw_image = np.array(raw_image)
    
    return raw_image

def apply_hanning_window_to_image(raw_image, rows_for_window, cols_for_window):

    # x_hann = np.arange(cols_for_window[0], cols_for_window[1], 1)
    # y_hann = np.arange(rows_for_window[0], rows_for_window[1], 1)

    hanning_image_x = np.hanning(cols_for_window[1] - cols_for_window[0])
    hanning_image_y = np.hanning(rows_for_window[1] - rows_for_window[0])

    hanning_window_image = np.outer(hanning_image_y, hanning_image_x)  # Create a 2D Hanning window

    raw_image = raw_image.astype(np.float64)

    raw_image[rows_for_window[0]:rows_for_window[1], cols_for_window[0]:cols_for_window[1]] *= hanning_window_image
    
    return raw_image

def apply_gaussian_window_to_image(image, sigma):

    filtered_image = gaussian_filter(image, sigma=sigma)

    return filtered_image

def pad_image(image, percentage_of_pad):
    """
    Pad the given image with zeros proportionally based on the image size.

    Parameters:
        image (numpy.ndarray): The input 2D image to be padded.
        row_padding_percent (float): Percentage of the image's rows to pad on top and bottom.
        col_padding_percent (float): Percentage of the image's columns to pad on left and right.

    Returns:
        numpy.ndarray: The zero-padded image.
    """
    # Get the original dimensions of the image
    rows, cols = image.shape

    # Calculate padding amounts based on the percentages
    row_padding = int(rows * percentage_of_pad / 100)
    col_padding = int(cols * percentage_of_pad / 100)

    # Apply padding using numpy.pad
    padded_image = np.pad(image, ((row_padding, row_padding), (col_padding, col_padding)), mode='constant', constant_values=0)

    height, width = padded_image.shape

    return padded_image, height, width, row_padding, col_padding


def crop_circle(R, circle_center, a, raw_image):

    h = 2*R/np.sqrt(a**2+1)
    l = a*h
    rows_rectangle = [int(circle_center[0] - h//2), int(circle_center[0] + h//2)]
    cols_rectangle = [int(circle_center[1] - l//2), int(circle_center[1] + l//2)]

    raw_image_cropped = raw_image[rows_rectangle[0]:rows_rectangle[1], cols_rectangle[0]:cols_rectangle[1]]

    return rows_rectangle, cols_rectangle, raw_image_cropped

def crop_into_square(R, circle_center, raw_image):

    rows_rectangle = [int(circle_center[0] - R), int(circle_center[0] + R)]
    cols_rectangle = [int(circle_center[1] - R), int(circle_center[1] + R)]

    raw_image_cropped = raw_image[rows_rectangle[0]:rows_rectangle[1], cols_rectangle[0]:cols_rectangle[1]]

    return rows_rectangle, cols_rectangle, raw_image_cropped

def fft_2d(image, width, height):
    # 2D fft of output intensity 
    fft_image = fft2(image)

    # Absolute value of fft
    abs_fft = np.abs(fft_image)

    noise_floor = np.mean(abs_fft[:, 250:width-250])
    noise_std = np.std(abs_fft[:, 250:width-250])

    # flattened_noise = abs_fft[:, 250:width-250].flatten()

    # sns.histplot(flattened_noise, bins=100, kde=True, color='blue', stat="density", label="Noise Distribution")
    # plt.show()

    # Shift the zero frequency component to the center of the spectrum
    fft_shifted = fftshift(fft_image)

    # Calculate the magnitude spectrum (already computed with abs_fft_pycrisp)
    abs_fft_shifted = np.abs(fft_shifted)

    # Define the frequency axes
    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    # Create the frequency grids
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Shift the grids to match the FFT shift
    freq_x_grid_shifted = fftshift(freq_x_grid)
    freq_y_grid_shifted = fftshift(freq_y_grid)

    return fft_image, abs_fft, noise_floor, noise_std, abs_fft_shifted, freq_x, freq_y, freq_x_grid_shifted, freq_y_grid_shifted

def find_peaks_fft(function, width, height):

    peaks = peak_local_max(function, min_distance=5, threshold_abs=.1e6, exclude_border=False)
    print('No. peaks:', np.shape(peaks)[0])

    peaks = peaks[peaks[:, 1] >= 10]
    peaks = peaks[peaks[:, 1] <= width-10]

    coords_pz_bottom = peaks[peaks[:, 0] <= 5]
    coords_pz_top = peaks[peaks[:,0] >= height-5]

    coords_pz1 = coords_pz_bottom[coords_pz_bottom[:,1] <= width//2][0]
    coords_pz2 = coords_pz_bottom[coords_pz_bottom[:,1] > width//2][0]

    coords_pz3 = coords_pz_top[coords_pz_top[:, 1] <= width//2][0]
    coords_pz4 = coords_pz_top[coords_pz_top[:, 1] > width//2][0]

    peaks = peaks[peaks[:, 0] > 5]
    peaks = peaks[peaks[:, 0] < height-5]

    # Add the coordinates for each pair
    coordinate_sums = np.sum(peaks, axis=1)

    # Sort indices by the sum in descending order
    sorted_indices = np.argsort(coordinate_sums)[::-1]

    coords_pp1 = peaks[sorted_indices[3]]
    coords_pp2 = peaks[sorted_indices[0]]

    coords_pm1 = peaks[sorted_indices[2]]
    coords_pm2 = peaks[sorted_indices[1]]

    all_bright_spot_coords = [coords_pz1, coords_pz2, coords_pz3, coords_pz4, coords_pp1, coords_pp2, coords_pm1, coords_pm2]

    return all_bright_spot_coords

def find_peaks_fft_45(function, width, height):

    peaks = peak_local_max(function, min_distance=5, threshold_abs=.5e6, exclude_border=True)
    print('No. peaks:', np.shape(peaks)[0])

    # considers peaks in the top left (tl) corner
    peaks_tl = peaks[peaks[:, 1] < width//2]
    peaks_tl = peaks_tl[peaks_tl[:, 0] > height//2]

    # sort peaks by column index
    peaks_tl = peaks_tl[np.argsort(peaks_tl[:, 1])]

    coords_pz1 = peaks_tl[1,:]
    coords_pp1 = peaks_tl[2,:]
    coords_pm1 = peaks_tl[0,:]

    # considers peaks in the bottom right (br) corner
    peaks_br = peaks[peaks[:, 1] > width//2]
    peaks_br = peaks_br[peaks_br[:, 0] < height//2]

    # sort peaks by column index
    peaks_br = peaks_br[np.argsort(peaks_br[:, 1])]

    coords_pz2 = peaks_br[1,:]
    coords_pp2 = peaks_br[0,:]
    coords_pm2 = peaks_br[2,:]

    all_bright_spot_coords = [coords_pz1, coords_pz2, coords_pp1, coords_pp2, coords_pm1, coords_pm2]

    return all_bright_spot_coords

def find_edge(image, edge_type, threshold, bright_spot_coords):
    """
    Find the edge coordinate in the FFT domain based on an absolute intensity threshold.

    Parameters:
        image (ndarray): The 2D array representing the image.
        edge_type (str): The direction of the edge ('right', 'left', 'top', 'bottom').
        threshold (float): The absolute threshold for determining the edge.
        bright_spot_coords (tuple): Coordinates of the bright spot (row, column).

    Returns:
        int: The coordinate of the detected edge, or None if the array edge is reached without satisfying the threshold.
    """
    if edge_type not in ['right', 'left', 'top', 'bottom']:
        raise ValueError("Invalid edge_type. Must be 'right', 'left', 'top', or 'bottom'.")

    # Initialize edge coordinate
    edge_coord = bright_spot_coords[1] if edge_type in ['right', 'left'] else bright_spot_coords[0]

    # Get image dimensions
    rows, cols = image.shape

    # Track consecutive pixels meeting the threshold
    consecutive_met = 0
    consecutive_threshold = 2  # Number of consecutive pixels needed to confirm the edge

    while True:
        # Get the current pixel intensity
        current_value = (
            image[bright_spot_coords[0], edge_coord]
            if edge_type in ['right', 'left']
            else image[edge_coord, bright_spot_coords[1]]
        )

        # Check if the threshold condition is met (absolute check)
        if current_value < threshold:
            consecutive_met += 1
        else:
            consecutive_met = 0

        # If threshold condition is met for consecutive pixels, stop
        if consecutive_met >= consecutive_threshold:
            break

        # Move in the specified direction
        if edge_type == 'right':
            edge_coord += 1
            if edge_coord >= cols:
                edge_coord = None
                break
        elif edge_type == 'left':
            edge_coord -= 1
            if edge_coord < 0:
                edge_coord = None
                break
        elif edge_type == 'top':
            edge_coord += 1
            if edge_coord >= rows:
                edge_coord = None
                break
        elif edge_type == 'bottom':
            edge_coord -= 1
            if edge_coord < 0:
                edge_coord = None
                break

    return edge_coord

def find_edge_grad(image, edge_type, bright_spot_coords, gradient_threshold=0):
    """
    Find the edge coordinate in the FFT domain when the gradient flattens out. Requires that two consecutive gradients are above/below the threshold.

    Parameters:
        image (ndarray): The 2D array representing the image.
        edge_type (str): The direction of the edge ('right', 'left', 'top', 'bottom').
        bright_spot_coords (tuple): Coordinates of the bright spot (row, column).
        gradient_threshold (float): The threshold for detecting when the gradient flattens out.

    Returns:
        int: The coordinate of the detected edge, or None if the array edge is reached.
    """
    if edge_type not in ['right', 'left', 'top', 'bottom']:
        raise ValueError("Invalid edge_type. Must be 'right', 'left', 'top', or 'bottom'.")


    rows, cols = image.shape

    if edge_type == 'right':
        edge_coord = bright_spot_coords[1] + 3
        while edge_coord < cols:
            gradient = image[bright_spot_coords[0], edge_coord] - image[bright_spot_coords[0], edge_coord - 1]
            gradient_2 = image[bright_spot_coords[0], edge_coord - 1] - image[bright_spot_coords[0], edge_coord - 2]
            gradient_3 = image[bright_spot_coords[0], edge_coord - 2] - image[bright_spot_coords[0], edge_coord - 3]
            #gradient_4 = image[bright_spot_coords[0], edge_coord - 3] - image[bright_spot_coords[0], edge_coord - 4]
            if gradient > gradient_threshold and gradient_2 > gradient_threshold and gradient_3 > gradient_threshold:# and gradient_4 > gradient_threshold:
                break
            edge_coord = edge_coord + 1
        if edge_coord == cols - 1:
            final_edge_coord = edge_coord
        else:
            final_edge_coord = edge_coord - 3
    elif edge_type == 'left':
        edge_coord = bright_spot_coords[1] - 3
        while edge_coord >= 0:
            gradient = image[bright_spot_coords[0], edge_coord + 1] - image[bright_spot_coords[0], edge_coord]
            gradient_2 = image[bright_spot_coords[0], edge_coord + 2] - image[bright_spot_coords[0], edge_coord + 1]
            gradient_3 = image[bright_spot_coords[0], edge_coord + 3] - image[bright_spot_coords[0], edge_coord + 2]
            #gradient_4 = image[bright_spot_coords[0], edge_coord + 4] - image[bright_spot_coords[0], edge_coord + 3]
            if gradient < gradient_threshold and gradient_2 < gradient_threshold and gradient_3 < gradient_threshold:# and gradient_4 < gradient_threshold:
                break
            edge_coord = edge_coord - 1
        if edge_coord == 0:
            final_edge_coord = edge_coord
        else:
            final_edge_coord = edge_coord + 3
    elif edge_type == 'top':
        edge_coord = bright_spot_coords[0] + 3
        while edge_coord < rows:
            gradient = image[edge_coord, bright_spot_coords[1]] - image[edge_coord - 1, bright_spot_coords[1]]
            gradient_2 = image[edge_coord -1, bright_spot_coords[1]] - image[edge_coord - 2, bright_spot_coords[1]]
            gradient_3 = image[edge_coord - 2, bright_spot_coords[1]] - image[edge_coord - 3, bright_spot_coords[1]]
            #gradient_4 = image[edge_coord - 3, bright_spot_coords[1]] - image[edge_coord - 4, bright_spot_coords[1]]
            if gradient > gradient_threshold and gradient_2 > gradient_threshold and gradient_3 > gradient_threshold:# and gradient_4 > gradient_threshold:
                break
            edge_coord = edge_coord + 1
        if edge_coord == rows - 1:
            final_edge_coord = edge_coord
        else:
            final_edge_coord = edge_coord - 3
    else: # bottom
        edge_coord = bright_spot_coords[0] - 3
        while edge_coord > 0:
            gradient = image[edge_coord + 1, bright_spot_coords[1]] - image[edge_coord, bright_spot_coords[1]]
            gradient_2 = image[edge_coord + 2, bright_spot_coords[1]] - image[edge_coord + 1, bright_spot_coords[1]]
            gradient_3 = image[edge_coord + 3, bright_spot_coords[1]] - image[edge_coord + 2, bright_spot_coords[1]]
            #gradient_4 = image[edge_coord + 4, bright_spot_coords[1]] - image[edge_coord + 3, bright_spot_coords[1]]
            if gradient < gradient_threshold and gradient_2 < gradient_threshold and gradient_3 < gradient_threshold:# and gradient_4 < gradient_threshold:
                break
            edge_coord = edge_coord - 1
        if edge_coord == 0:
            final_edge_coord = edge_coord
        else:
            final_edge_coord = edge_coord + 3
    
    return final_edge_coord


def calculate_edges(image, threshold, threshold_pz, all_bright_spot_coords, height):

    coords_pz1 = all_bright_spot_coords[0]
    coords_pz2 = all_bright_spot_coords[1]
    coords_pz3 = all_bright_spot_coords[2]
    coords_pz4 = all_bright_spot_coords[3]
    coords_pp1 = all_bright_spot_coords[4]
    coords_pp2 = all_bright_spot_coords[5]
    coords_pm1 = all_bright_spot_coords[6]
    coords_pm2 = all_bright_spot_coords[7]

    ### Plus zero

    ## PZ1
    # right edge
    r_pz1 = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    # left edge
    l_pz1 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    # top edge
    t_pz1_mean = (coords_pp1[0] + coords_pz1[0]) // 2
    t_pz1_threshold = find_edge(image, edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    if t_pz1_threshold < t_pz1_mean:
        t_pz1 = t_pz1_threshold
        print('Top PZ1: Threshold')
    else:
        t_pz1 = t_pz1_mean
        print('Top PZ1: Middle')

    # bottom edge
    b_pz1 = 0

    ## PZ2 
    # right edge
    r_pz2 = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    # left edge
    l_pz2 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    # top edge
    t_pz2_mean = (coords_pm2[0] + coords_pz2[0]) // 2
    t_pz2_threshold = find_edge(image, edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    if t_pz2_threshold < t_pz2_mean:
        t_pz2 = t_pz2_threshold
        print('Top PZ2: Threshold')
    else:
        t_pz2 = t_pz2_mean
        print('Top PZ2: Middle')

    # bottom edge
    b_pz2 = 0

    ## PZ3
    # right edge
    r_pz3 = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz3)

    # left edge
    l_pz3 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz3)

    # top edge
    t_pz3 = height

    # bottom edge
    b_pz3_mean = (coords_pm1[0] + coords_pz3[0]) // 2
    b_pz3_threshold = find_edge(image, edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz3)

    if b_pz3_threshold > b_pz3_mean:
        b_pz3 = b_pz3_threshold
        print('Bottom PZ3: Threshold')
    else:
        b_pz3 = b_pz3_mean
        print('Bottom PZ3: Middle')
    
    ## PZ4
    # right edge
    r_pz4 = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz4)

    # left edge
    l_pz4 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz4)

    # top edge
    t_pz4 = height

    # bottom edge
    b_pz4_mean = (coords_pp2[0] + coords_pz4[0]) // 2
    b_pz4_threshold = find_edge(image, edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz4)

    if b_pz4_threshold > b_pz4_mean:
        b_pz4 = b_pz4_threshold
        print('Bottom PZ4: Threshold')
    else:
        b_pz4 = b_pz4_mean
        print('Bottom PZ4: Middle')


    ### Plus plus

    ## PP1
    # right edge
    r_pp1 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pp1)

    # left edge
    l_pp1 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pp1)

    # top edge
    t_pp1 = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pp1)

    # bottom edge
    b_pp1_threshold = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pp1)

    if b_pp1_threshold == None:
        b_pp1 = t_pz1_mean
    else:
        b_pp1 = b_pp1_threshold

    ## PP2 
    # right edge
    r_pp2 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pp2)

    # left edge
    l_pp2 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pp2)

    # top edge
    t_pp2_threshold = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pp2)

    if t_pp2_threshold == None:
        t_pp2 = b_pz4_mean
    else:
        t_pp2 = t_pp2_threshold

    # bottom edge
    b_pp2 = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pp2)


    ### Plus minus

    ## PM1
    # right edge
    r_pm1 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pm1)

    # left edge
    l_pm1 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pm1)

    # top edge
    t_pm1_threshold = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pm1)

    if t_pm1_threshold == None:
        t_pm1 = b_pz3_mean
    else:
        t_pm1 = t_pm1_threshold

    # bottom edge
    b_pm1 = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pm1)

    ## PM2 
    # right edge
    r_pm2 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pm2)

    # left edge
    l_pm2 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pm2)

    # top edge
    t_pm2 = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pm2)

    # bottom edge
    b_pm2_threshold = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pm2)

    if b_pm2_threshold == None:
        b_pm2 = t_pz2_mean
    else:
        b_pm2 = b_pm2_threshold

    edge_coords = [b_pz1, t_pz1, l_pz1, r_pz1, b_pz2, t_pz2, l_pz2, r_pz2, b_pz3, t_pz3, l_pz3, r_pz3, b_pz4, t_pz4, l_pz4, r_pz4, 
                   b_pp1, t_pp1, l_pp1, r_pp1, b_pp2, t_pp2, l_pp2, r_pp2, b_pm1, t_pm1, l_pm1, r_pm1, b_pm2, t_pm2, l_pm2, r_pm2]

    edge_names = [
    'b_pz1', 't_pz1', 'l_pz1', 'r_pz1', 'b_pz2', 't_pz2', 'l_pz2', 'r_pz2',
    'b_pz3', 't_pz3', 'l_pz3', 'r_pz3', 'b_pz4', 't_pz4', 'l_pz4', 'r_pz4',
    'b_pp1', 't_pp1', 'l_pp1', 'r_pp1', 'b_pp2', 't_pp2', 'l_pp2', 'r_pp2',
    'b_pm1', 't_pm1', 'l_pm1', 'r_pm1', 'b_pm2', 't_pm2', 'l_pm2', 'r_pm2'
    ]

    # # Print each name with its corresponding value
    # for name, value in zip(edge_names, edge_coords):
    #     print(f"{name}: {value}")

    return edge_coords

def calculate_edges_45(image, threshold, threshold_pz, all_bright_spot_coords, height):

    coords_pz1 = all_bright_spot_coords[0]
    coords_pz2 = all_bright_spot_coords[1]
    coords_pp1 = all_bright_spot_coords[2]
    coords_pp2 = all_bright_spot_coords[3]
    coords_pm1 = all_bright_spot_coords[4]
    coords_pm2 = all_bright_spot_coords[5]

    ### Plus zero

    ## PZ1
    # right edge
    r_pz1 = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    # left edge
    l_pz1 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    # top edge
    t_pz1_threshold = find_edge(image, edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz1)
    
    if t_pz1_threshold == None:
        t_pz1 = height
        print('Top PZ1: Edge of image')
    else:
        t_pz1 = t_pz1_threshold

    # bottom edge
    b_pz1 = find_edge(image, edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz1)

    ## PZ2 
    # right edge
    r_pz2_threshold = find_edge(image, edge_type='right', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    if r_pz2_threshold == None:
        r_pz2 = width
        print('Right PZ2: Edge of image')
    else:
        r_pz2 = r_pz2_threshold

    # left edge
    l_pz2 = find_edge(image, edge_type='left', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    # top edge
    t_pz2 = find_edge(image, edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    # bottom edge
    b_pz2_threshold = find_edge(image, edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz2)

    if b_pz2_threshold == None:
        b_pz2 = 0
        print('Bottom PZ2: Edge of image')
    else:
        b_pz2 = b_pz2_threshold

    
    ### Plus plus

    ## PP1
    # right edge
    r_pp1 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pp1)

    # left edge
    l_pp1 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pp1)

    # top edge
    t_pp1_threshold = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pp1)

    if t_pp1_threshold == None:
        t_pp1 = height
    else:
        t_pp1 = t_pp1_threshold

    # bottom edge
    b_pp1 = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pp1)

    ## PP2 
    # right edge
    r_pp2 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pp2)

    # left edge
    l_pp2 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pp2)

    # top edge
    t_pp2 = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pp2)

    # bottom edge
    b_pp2_threshold = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pp2)

    if b_pp2_threshold == None:
        b_pp2 = 0
    else:
        b_pp2 = b_pp2_threshold


    ### Plus minus

    ## PM1
    # right edge
    r_pm1 = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pm1)

    # left edge
    l_pm1_threshold = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pm1)

    if l_pm1_threshold == None:
        l_pm1 = 0
    else:
        l_pm1 = l_pm1_threshold

    # top edge
    t_pm1 = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pm1)

    # bottom edge
    b_pm1 = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pm1)

    ## PM2 
    # right edge
    r_pm2_threshold = find_edge(image, edge_type='right', threshold=threshold, bright_spot_coords=coords_pm2)

    if r_pm2_threshold == None:
        r_pm2 = width
    else:
        r_pm2 = r_pm2_threshold

    # left edge
    l_pm2 = find_edge(image, edge_type='left', threshold=threshold, bright_spot_coords=coords_pm2)

    # top edge
    t_pm2 = find_edge(image, edge_type='top', threshold=threshold, bright_spot_coords=coords_pm2)

    # bottom edge
    b_pm2 = find_edge(image, edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pm2)

    edge_coords = [b_pz1, t_pz1, l_pz1, r_pz1, b_pz2, t_pz2, l_pz2, r_pz2, b_pp1, t_pp1, l_pp1, r_pp1, b_pp2, t_pp2, l_pp2, r_pp2, b_pm1, t_pm1, l_pm1, r_pm1, b_pm2, t_pm2, l_pm2, r_pm2]


    return edge_coords

def calculate_edges_45_grad(image, all_bright_spot_coords, height):

    coords_pz1 = all_bright_spot_coords[0]
    coords_pz2 = all_bright_spot_coords[1]
    coords_pp1 = all_bright_spot_coords[2]
    coords_pp2 = all_bright_spot_coords[3]
    coords_pm1 = all_bright_spot_coords[4]
    coords_pm2 = all_bright_spot_coords[5]

    ### Plus zero

    ## PZ1
    # right edge
    r_pz1 = find_edge_grad(image, edge_type='right', bright_spot_coords=coords_pz1)

    # left edge
    l_pz1 = find_edge_grad(image, edge_type='left', bright_spot_coords=coords_pz1)

    # make the windows symmetical around bright spot
    if np.abs(r_pz1 - coords_pz1[1]) < np.abs(l_pz1 - coords_pz1[1]):
        l_pz1 = 2*coords_pz1[1] - r_pz1
    else:
        r_pz1 = 2*coords_pz1[1] - l_pz1

    # top edge
    t_pz1 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pz1)

    # bottom edge
    b_pz1 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pz1)

    # make the windows symmetical around bright spot
    if np.abs(t_pz1 - coords_pz1[0]) < np.abs(b_pz1 - coords_pz1[0]):
        b_pz1 = 2*coords_pz1[0] - t_pz1
    else:
        t_pz1 = 2*coords_pz1[0] - b_pz1
             

    ## PZ2 
    # right edge
    r_pz2 = find_edge_grad(image, edge_type='right',  bright_spot_coords=coords_pz2)

    # left edge
    l_pz2 = find_edge_grad(image, edge_type='left',  bright_spot_coords=coords_pz2)

    # make the windows symmetical around bright spot
    if np.abs(r_pz2 - coords_pz2[1]) < np.abs(l_pz2 - coords_pz2[1]):
        l_pz2 = 2*coords_pz2[1] - r_pz2
    else:
        r_pz2 = 2*coords_pz2[1] - l_pz2

    # top edge
    t_pz2 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pz2)

    # bottom edge
    b_pz2 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pz2)

    # make the windows symmetical around bright spot
    if np.abs(t_pz2 - coords_pz2[0]) < np.abs(b_pz2 - coords_pz2[0]):
        b_pz2 = 2*coords_pz2[0] - t_pz2
    else:
        t_pz2 = 2*coords_pz2[0] - b_pz2


    ### Plus plus

    ## PP1
    # right edge
    r_pp1 = find_edge_grad(image, edge_type='right',  bright_spot_coords=coords_pp1)

    # left edge
    l_pp1 = find_edge_grad(image, edge_type='left',  bright_spot_coords=coords_pp1)

    # make the windows symmetical around bright spot
    if np.abs(r_pp1 - coords_pp1[1]) < np.abs(l_pp1 - coords_pp1[1]):
        l_pp1 = 2*coords_pp1[1] - r_pp1
    else:
        r_pp1 = 2*coords_pp1[1] - l_pp1

    # top edge
    t_pp1 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pp1)

    # bottom edge
    b_pp1 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pp1)

    # make the windows symmetical around bright spot
    if np.abs(t_pp1 - coords_pp1[0]) < np.abs(b_pp1 - coords_pp1[0]):
        b_pp1 = 2*coords_pp1[0] - t_pp1
    else:
        t_pp1 = 2*coords_pp1[0] - b_pp1
    
    
    ## PP2 
    # right edge
    r_pp2 = find_edge_grad(image, edge_type='right',  bright_spot_coords=coords_pp2)

    # left edge
    l_pp2 = find_edge_grad(image, edge_type='left',  bright_spot_coords=coords_pp2)

    # make the windows symmetical around bright spot
    if np.abs(r_pp2 - coords_pp2[1]) < np.abs(l_pp2 - coords_pp2[1]):
        l_pp2 = 2*coords_pp2[1] - r_pp2
    else:
        r_pp2 = 2*coords_pp2[1] - l_pp2

    # top edge
    t_pp2 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pp2)

    # bottom edge
    b_pp2 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pp2)

    # make the windows symmetical around bright spot
    if np.abs(t_pp2 - coords_pp2[0]) < np.abs(b_pp2 - coords_pp2[0]):
        b_pp2 = 2*coords_pp2[0] - t_pp2
    else:
        t_pp2 = 2*coords_pp2[0] - b_pp2


    ### Plus minus

    ## PM1
    # right edge
    r_pm1 = find_edge_grad(image, edge_type='right',  bright_spot_coords=coords_pm1)

    # left edge
    l_pm1 = find_edge_grad(image, edge_type='left',  bright_spot_coords=coords_pm1)

    # make the windows symmetical around bright spot
    if np.abs(r_pm1 - coords_pm1[1]) < np.abs(l_pm1 - coords_pm1[1]):
        l_pm1 = 2*coords_pm1[1] - r_pm1
    else:
        r_pm1 = 2*coords_pm1[1] - l_pm1

    # top edge
    t_pm1 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pm1)

    # bottom edge
    b_pm1 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pm1)

    # make the windows symmetical around bright spot
    if np.abs(t_pm1 - coords_pm1[0]) < np.abs(b_pm1 - coords_pm1[0]):
        b_pm1 = 2*coords_pm1[0] - t_pm1
    else:
        t_pm1 = 2*coords_pm1[0] - b_pm1

    ## PM2 
    # right edge
    r_pm2 = find_edge_grad(image, edge_type='right',  bright_spot_coords=coords_pm2)

    # left edge
    l_pm2 = find_edge_grad(image, edge_type='left',  bright_spot_coords=coords_pm2)

    # make the windows symmetical around bright spot
    if np.abs(r_pm2 - coords_pm2[1]) < np.abs(l_pm2 - coords_pm2[1]):
        l_pm2 = 2*coords_pm2[1] - r_pm2
    else:
        r_pm2 = 2*coords_pm2[1] - l_pm2

    # top edge
    t_pm2 = find_edge_grad(image, edge_type='top',  bright_spot_coords=coords_pm2)

    # bottom edge
    b_pm2 = find_edge_grad(image, edge_type='bottom',  bright_spot_coords=coords_pm2)

    # make the windows symmetical around bright spot
    if np.abs(t_pm2 - coords_pm2[0]) < np.abs(b_pm2 - coords_pm2[0]):
        b_pm2 = 2*coords_pm2[0] - t_pm2
    else:
        t_pm2 = 2*coords_pm2[0] - b_pm2

    edge_coords = [b_pz1, t_pz1, l_pz1, r_pz1, b_pz2, t_pz2, l_pz2, r_pz2, b_pp1, t_pp1, l_pp1, r_pp1, b_pp2, t_pp2, l_pp2, r_pp2, b_pm1, t_pm1, l_pm1, r_pm1, b_pm2, t_pm2, l_pm2, r_pm2]

    return edge_coords

def define_masks(fft_window, edge_coords, height, width):
    # Initialize empty masks with the same shape as the image
    mask_pz1 = np.zeros((height, width))
    mask_pz2 = np.zeros((height, width))
    mask_pz3 = np.zeros((height, width))
    mask_pz4 = np.zeros((height, width))
    mask_pp1 = np.zeros((height, width))
    mask_pp2 = np.zeros((height, width))
    mask_pm1 = np.zeros((height, width))
    mask_pm2 = np.zeros((height, width))

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pz3 = edge_coords[8]
    t_pz3 = edge_coords[9]
    l_pz3 = edge_coords[10]
    r_pz3 = edge_coords[11]
    b_pz4 = edge_coords[12]
    t_pz4 = edge_coords[13]
    l_pz4 = edge_coords[14]
    r_pz4 = edge_coords[15]
    b_pp1 = edge_coords[16]
    t_pp1 = edge_coords[17]
    l_pp1 = edge_coords[18]
    r_pp1 = edge_coords[19]
    b_pp2 = edge_coords[20]
    t_pp2 = edge_coords[21]
    l_pp2 = edge_coords[22]
    r_pp2 = edge_coords[23]
    b_pm1 = edge_coords[24]
    t_pm1 = edge_coords[25]
    l_pm1 = edge_coords[26]
    r_pm1 = edge_coords[27]
    b_pm2 = edge_coords[28]
    t_pm2 = edge_coords[29]
    l_pm2 = edge_coords[30]
    r_pm2 = edge_coords[31]

    if fft_window == 'hanning':

        hann_window_pz1_y = np.hanning(2*(t_pz1-b_pz1))  # 1D Hanning window
        hann_window_pz1_x = np.hanning(r_pz1-l_pz1)  # 1D Hanning window
        hann_window_pz1 = np.outer(hann_window_pz1_y, hann_window_pz1_x)[np.shape(hann_window_pz1_y)[0]//2:,:]  # Create a 2D Hanning window

        hann_window_pz2_y = np.hanning(2*(t_pz2-b_pz2))  # 1D Hanning window
        hann_window_pz2_x = np.hanning(r_pz2-l_pz2)  # 1D Hanning window
        hann_window_pz2 = np.outer(hann_window_pz2_y, hann_window_pz2_x)[np.shape(hann_window_pz2_y)[0]//2:,:]  # Create a 2D Hanning window

        hann_window_pz3_y = np.hanning(2*(t_pz3-b_pz3))  # 1D Hanning window
        hann_window_pz3_x = np.hanning(r_pz3-l_pz3)  # 1D Hanning window
        hann_window_pz3 = np.outer(hann_window_pz3_y, hann_window_pz3_x)[0:np.shape(hann_window_pz3_y)[0]//2,:]  # Create a 2D Hanning window

        hann_window_pz4_y = np.hanning(2*(t_pz4-b_pz4))  # 1D Hanning window
        hann_window_pz4_x = np.hanning(r_pz4-l_pz4)  # 1D Hanning window
        hann_window_pz4 = np.outer(hann_window_pz4_y, hann_window_pz4_x)[0:np.shape(hann_window_pz4_y)[0]//2,:]  # Create a 2D Hanning window

        hann_window_pp1_y = np.hanning(t_pp1-b_pp1)  # 1D Hanning window
        hann_window_pp1_x = np.hanning(r_pp1-l_pp1)  # 1D Hanning window
        hann_window_pp1 = np.outer(hann_window_pp1_y, hann_window_pp1_x)  # Create a 2D Hanning window

        hann_window_pp2_y = np.hanning(t_pp2-b_pp2)  # 1D Hanning window
        hann_window_pp2_x = np.hanning(r_pp2-l_pp2)  # 1D Hanning window
        hann_window_pp2 = np.outer(hann_window_pp2_y, hann_window_pp2_x)  # Create a 2D Hanning window

        hann_window_pm1_y = np.hanning(t_pm1-b_pm1)  # 1D Hanning window
        hann_window_pm1_x = np.hanning(r_pm1-l_pm1)  # 1D Hanning window
        hann_window_pm1 = np.outer(hann_window_pm1_y, hann_window_pm1_x)  # Create a 2D Hanning window

        hann_window_pm2_y = np.hanning(t_pm2-b_pm2)  # 1D Hanning window
        hann_window_pm2_x = np.hanning(r_pm2-l_pm2)  # 1D Hanning window
        hann_window_pm2 = np.outer(hann_window_pm2_y, hann_window_pm2_x)  # Create a 2D Hanning window

        # Applying Hanning window to each mask instead of using ones
        mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = hann_window_pz1
        mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = hann_window_pz2

        mask_pz3[b_pz3:t_pz3, l_pz3:r_pz3] = hann_window_pz3
        mask_pz4[b_pz4:t_pz4, l_pz4:r_pz4] = hann_window_pz4

        mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = hann_window_pp1
        mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = hann_window_pp2

        mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = hann_window_pm1
        mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = hann_window_pm2

    elif fft_window == 'tophat':

        # Applying Hanning window to each mask instead of using ones
        mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = 1
        mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = 1

        mask_pz3[b_pz3:t_pz3, l_pz3:r_pz3] = 1
        mask_pz4[b_pz4:t_pz4, l_pz4:r_pz4] = 1

        mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = 1
        mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = 1

        mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = 1
        mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = 1

    else:
        print('Define windowing!')
    # Combine masks for different regions
    mask_pz = mask_pz1 + mask_pz2 + mask_pz3 + mask_pz4 # (+0)
    mask_pp = mask_pp1 + mask_pp2 # (++)
    mask_pm = mask_pm1 + mask_pm2 # (+-)

    return mask_pz, mask_pp, mask_pm

def define_masks_45(fft_window, edge_coords, height, width, alpha):

    # Initialize empty masks with the same shape as the image
    mask_pz1 = np.zeros((height, width))
    mask_pz2 = np.zeros((height, width))
    mask_pp1 = np.zeros((height, width))
    mask_pp2 = np.zeros((height, width))
    mask_pm1 = np.zeros((height, width))
    mask_pm2 = np.zeros((height, width))

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pp1 = edge_coords[8]
    t_pp1 = edge_coords[9]
    l_pp1 = edge_coords[10]
    r_pp1 = edge_coords[11]
    b_pp2 = edge_coords[12]
    t_pp2 = edge_coords[13]
    l_pp2 = edge_coords[14]
    r_pp2 = edge_coords[15]
    b_pm1 = edge_coords[16]
    t_pm1 = edge_coords[17]
    l_pm1 = edge_coords[18]
    r_pm1 = edge_coords[19]
    b_pm2 = edge_coords[20]
    t_pm2 = edge_coords[21]
    l_pm2 = edge_coords[22]
    r_pm2 = edge_coords[23]

    if fft_window == 'hanning':

        hann_window_pz1_y = np.hanning(t_pz1-b_pz1)  # 1D Hanning window
        hann_window_pz1_x = np.hanning(r_pz1-l_pz1)  # 1D Hanning window
        hann_window_pz1 = np.outer(hann_window_pz1_y, hann_window_pz1_x)  # Create a 2D Hanning window

        hann_window_pz2_y = np.hanning(t_pz2-b_pz2)  # 1D Hanning window
        hann_window_pz2_x = np.hanning(r_pz2-l_pz2)  # 1D Hanning window
        hann_window_pz2 = np.outer(hann_window_pz2_y, hann_window_pz2_x)  # Create a 2D Hanning window

        hann_window_pp1_y = np.hanning(t_pp1-b_pp1)  # 1D Hanning window
        hann_window_pp1_x = np.hanning(r_pp1-l_pp1)  # 1D Hanning window
        hann_window_pp1 = np.outer(hann_window_pp1_y, hann_window_pp1_x)  # Create a 2D Hanning window

        hann_window_pp2_y = np.hanning(t_pp2-b_pp2)  # 1D Hanning window
        hann_window_pp2_x = np.hanning(r_pp2-l_pp2)  # 1D Hanning window
        hann_window_pp2 = np.outer(hann_window_pp2_y, hann_window_pp2_x)  # Create a 2D Hanning window

        hann_window_pm1_y = np.hanning(t_pm1-b_pm1)  # 1D Hanning window
        hann_window_pm1_x = np.hanning(r_pm1-l_pm1)  # 1D Hanning window
        hann_window_pm1 = np.outer(hann_window_pm1_y, hann_window_pm1_x)  # Create a 2D Hanning window

        hann_window_pm2_y = np.hanning(t_pm2-b_pm2)  # 1D Hanning window
        hann_window_pm2_x = np.hanning(r_pm2-l_pm2)  # 1D Hanning window
        hann_window_pm2 = np.outer(hann_window_pm2_y, hann_window_pm2_x)  # Create a 2D Hanning window

        # Applying Hanning window to each mask instead of using ones
        mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = hann_window_pz1
        mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = hann_window_pz2

        mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = hann_window_pp1
        mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = hann_window_pp2

        mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = hann_window_pm1
        mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = hann_window_pm2
    
    elif fft_window =='tukey':

        tukey_window_pz1_y = tukey(t_pz1-b_pz1, alpha=alpha)  # 1D Hanning window
        tukey_window_pz1_x = tukey(r_pz1-l_pz1, alpha=alpha)  # 1D Hanning window
        tukey_window_pz1 = np.outer(tukey_window_pz1_y, tukey_window_pz1_x)  # Create a 2D Hanning window

        tukey_window_pz2_y = tukey(t_pz2-b_pz2, alpha=alpha)  # 1D Hanning window
        tukey_window_pz2_x = tukey(r_pz2-l_pz2, alpha=alpha)  # 1D Hanning window
        tukey_window_pz2 = np.outer(tukey_window_pz2_y, tukey_window_pz2_x)  # Create a 2D Hanning window

        tukey_window_pp1_y = tukey(t_pp1-b_pp1, alpha=alpha)  # 1D Hanning window
        tukey_window_pp1_x = tukey(r_pp1-l_pp1, alpha=alpha)  # 1D Hanning window
        tukey_window_pp1 = np.outer(tukey_window_pp1_y, tukey_window_pp1_x)  # Create a 2D Hanning window

        tukey_window_pp2_y = tukey(t_pp2-b_pp2, alpha=alpha)  # 1D Hanning window
        tukey_window_pp2_x = tukey(r_pp2-l_pp2, alpha=alpha)  # 1D Hanning window
        tukey_window_pp2 = np.outer(tukey_window_pp2_y, tukey_window_pp2_x)  # Create a 2D Hanning window

        tukey_window_pm1_y = tukey(t_pm1-b_pm1, alpha=alpha)  # 1D Hanning window
        tukey_window_pm1_x = tukey(r_pm1-l_pm1, alpha=alpha)  # 1D Hanning window
        tukey_window_pm1 = np.outer(tukey_window_pm1_y, tukey_window_pm1_x)  # Create a 2D Hanning window

        tukey_window_pm2_y = tukey(t_pm2-b_pm2, alpha=alpha)  # 1D Hanning window
        tukey_window_pm2_x = tukey(r_pm2-l_pm2, alpha=alpha)  # 1D Hanning window
        tukey_window_pm2 = np.outer(tukey_window_pm2_y, tukey_window_pm2_x)  # Create a 2D Hanning window

        # Applying Hanning window to each mask instead of using ones
        mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = tukey_window_pz1
        mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = tukey_window_pz2

        mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = tukey_window_pp1
        mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = tukey_window_pp2

        mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = tukey_window_pm1
        mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = tukey_window_pm2

    elif fft_window == 'tophat':
        # Applying Hanning window to each mask instead of using ones
        mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = 1
        mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = 1

        mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = 1
        mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = 1

        mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = 1
        mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = 1

    else:
        print('Define windowing!')

    # Combine masks for different regions
    mask_pz = mask_pz1 + mask_pz2 # (+0)
    mask_pp = mask_pp1 + mask_pp2 # (++)
    mask_pm = mask_pm1 + mask_pm2 # (+-)
  
    return mask_pz, mask_pp, mask_pm

def apply_masks(mask_pz, mask_pp, mask_pm, fft_function):

    # Apply the Hanning-weighted masks to the FFT image
    masked_fft_pz = mask_pz * fft_function # (+0)
    masked_fft_pp = mask_pp * fft_function # (++)
    masked_fft_pm = mask_pm * fft_function # (+-)

    return masked_fft_pz, masked_fft_pp, masked_fft_pm


def cosine_2d_linear(coords, A, B, C, kx, ky, phi, D):
    x_coord, y_coord = coords
    return (A*x_coord + B*y_coord + C) * np.cos(kx * x_coord + ky * y_coord + phi) + D

def compute_amplitude_2d_hilbert(function, function_name, gaussian_pre_smoothing, median_post_smoothing):
    """
    Compute the amplitude envelope of a 2D wave function using the Hilbert transform.

    Args:
        function (ndarray): Real-valued 2D array representing the wave function.

    Returns:
        ndarray: Amplitude envelope of the input wave function.
    """
    
    # Pre-smoothing to reduce high-frequency noise
    if gaussian_pre_smoothing == 'Yes':
        function = gaussian_filter(function, sigma=1)

    if function_name == '+0':
        # Apply Hilbert transform along one axis (rows)
        analytic_signal_tot = hilbert(function, axis=1)  # Along columns

    elif function_name == '++' or '+-':
         # Apply Hilbert transform along one axis (rows)
        analytic_signal_tot = hilbert(function, axis=0)  # Along columns
        # Compute the amplitude envelope along the rows
        #amplitude_y = np.abs(analytic_signal_y)
        
        # Apply Hilbert transform again along the other axis (columns)
        #analytic_signal_tot = hilbert(amplitude_y, axis=0)  # Along rows

    else:
         print('Define function name')

    amplitude_tot = np.abs(analytic_signal_tot)
    
    if median_post_smoothing == 'Yes':
        amplitude_tot = median_filter(amplitude_tot, size=5)

    return amplitude_tot


    
def find_peaks_2d(function):
    """
    Find peaks in a 2D array row by row and populate an array with the same size
    as the input, with the values of the peaks at their indices and zeros elsewhere.

    Parameters:
        function (numpy.ndarray): 2D array where each row is treated as a 1D signal.

    Returns:
        peaks_array (numpy.ndarray): 2D array with peak values at peak indices and zeros elsewhere.
    """
    if function.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Create an array of zeros with the same shape as the input
    peaks_array = np.zeros_like(function)

    for row in range(function.shape[0]):
        # Find peaks for the current row
        peaks, _ = find_peaks(
            function[row, :], 
            height=None, 
            threshold=None, 
            distance=None, 
            prominence=None, 
            width=None, 
            wlen=None, 
            rel_height=0.5, 
            plateau_size=None
        )
        # Set the peak values at the found indices
        peaks_array[row, peaks] = function[row, peaks]

    return peaks_array

def interpolate_zeros_2d(arr):
    # Get the shape of the array
    rows, cols = arr.shape

    # Create a grid of coordinates (row, col)
    row_indices, col_indices = np.indices(arr.shape)

    # Flatten the arrays for easier processing
    coords = np.column_stack((row_indices.flatten(), col_indices.flatten()))
    values = arr.flatten()

    # Identify the non-zero indices and their corresponding values
    non_zero_indices = values != 0
    non_zero_coords = coords[non_zero_indices]
    non_zero_values = values[non_zero_indices]

    # If there are no non-zero values, return the original array
    if len(non_zero_coords) == 0:
        return arr

    # Interpolate the array using griddata for 2D interpolation
    interpolated_array = interpolate.griddata(
        non_zero_coords, non_zero_values, coords, method='cubic', fill_value=0
    )

    # Reshape the interpolated array to the original shape
    return interpolated_array.reshape(arr.shape)


def find_amplitudes(function_pz, function_pp, function_pm, method, gaussian_pre_smoothing, median_post_smoothing):

    if method == 'curve_fit':

        A_pp, Z_fitpp, zdatapp, fn_pp, params_1dpp, coords = cosine_fitting_2d(ifft_pp, cosine_modulation='linear')
        A_pz, Z_fitpz, zdata_pz, fn_pz, params_1dpz, coords = cosine_fitting_2d(ifft_pz, cosine_modulation='linear')
        A_pm, Z_fitpm, zdatapm, fn_pm, params_1dpm, coords = cosine_fitting_2d(ifft_pm, cosine_modulation='linear')
        
    elif method == 'hilbert':

        A_pz = compute_amplitude_2d_hilbert(function_pz, '+0', gaussian_pre_smoothing, median_post_smoothing)
        A_pp = compute_amplitude_2d_hilbert(function_pp, '++', gaussian_pre_smoothing, median_post_smoothing)
        A_pm = compute_amplitude_2d_hilbert(function_pm, '+-', gaussian_pre_smoothing, median_post_smoothing)

    elif method == 'find_peaks':
        A_pz = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pz)))
        A_pp = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pp)))
        A_pm = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pm)))
        
    else:
        raise(ValueError('Define modulation!'))
    
    return A_pz, A_pp, A_pm


def safe_divide(a, b):

    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def calculate_theta(A_pz, A_pp, A_pm, height, width):

    theta_out_1_pp = np.pi/4 - 0.5*np.arctan(2*safe_divide(np.abs(A_pp), np.abs(A_pz)))
    theta_out_1_pm = np.pi/4 - 0.5*np.arctan(2*safe_divide(np.abs(A_pm), np.abs(A_pz)))
    theta_out_1_arith = np.pi/4 - 0.5*np.arctan(safe_divide(np.abs(A_pp) + np.abs(A_pm), np.abs(A_pz)))

    theta_out_00 = theta_out_1_pp[height//2, width//2]

    print(f'Theta out (0, 0): {np.rad2deg(theta_out_00):.3f}° ({np.rad2deg(np.min(theta_out_1_pp)):.3f}° - {np.rad2deg(np.max(theta_out_1_pp)):.3f}°)')
    print(f'Std theta_1 central lineout: {np.std(theta_out_1_pp[height//2, :])}')

    theta_out_2 = np.pi/4 - 0.5*np.arctan(np.sqrt(4*safe_divide(np.abs(A_pp)*np.abs(A_pm), np.abs(A_pz)**2)))

    theta_out_00 = theta_out_1_arith[height//2, width//2]

    print(f'Uncalibrated central theta = {np.rad2deg(theta_out_00):.3f}°')
    print(f'Std theta_2 central lineout: {np.std(theta_out_1_arith[height//2, :])}')

    return theta_out_1_pp, theta_out_1_pm, theta_out_1_arith, theta_out_2



def plotting(coords, original_function, fitted_function, function_name, params, x, y_for_lineout):

    plt.subplot(2,2,1)
    plt.title(f'Original function: {function_name}')
    plt.imshow(original_function, extent=[coords[0][0, 0], coords[0][0, -1], coords[1][0, 0], coords[1][-1, 0]], origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Intensity')

    plt.subplot(2,2,2)
    plt.title('Fitted 2D Cosine Function')
    plt.imshow(fitted_function, extent=[coords[0][0, 0], coords[0][0, -1], coords[1][0, 0], coords[1][-1, 0]], origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Intensity')

    plt.subplot(2,2,3)
    plt.title('Difference')
    plt.imshow(original_function - fitted_function, extent=[coords[0][0, 0], coords[0][0, -1], coords[1][0, 0], coords[1][-1, 0]], origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Intensity')

    # total_rows = original_function.shape[0]
    # middle_row = coords[1][0, 0] + total_rows//2
    # 
    x_fit = np.linspace(np.min(x), np.max(x), 10*len(x))
    lineout = cosine_1d(x_fit, y_for_lineout, *params)

    plt.subplot(2,2,4)
    plt.title(f'Lineout (central row)')
    # plt.plot(coords[0][total_rows // 2, :], original_function[total_rows//2, :], label='Original')
    plt.plot(x, original_function[y_for_lineout, :], label='Original')
    plt.plot(x_fit, lineout, label='Fit')
    plt.legend()
    
    plt.show()

def plotting_hilbert(original_function, hilbert_amplitude, function_name,  cols_for_lineout, rows_for_lineout):

    plt.close()
    plt.subplot(2,2,1)
    plt.title(f'IFFT: {function_name}')
    plt.imshow(original_function, origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.colorbar(label='Intensity')
     
    plt.subplot(2,2,2)
    plt.imshow(hilbert_amplitude, origin='lower')
    plt.colorbar(label='Amplitude Envelope')
    plt.title('Amplitude Envelope of 2D Wave')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 2, 3)
    plt.title(f'Lineout (rows: {rows_for_lineout})')

    # Loop over rows for lineout
    for i, r in enumerate(rows_for_lineout):  # enumerate gives index and value
        plt.plot(original_function[r, :], label=f'Original IFFT (row {r})')
        plt.plot(hilbert_amplitude[r, :], label=f'Hilbert (row {r})')
    plt.xlabel('x pixel')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title(f'Lineout (columns: {cols_for_lineout})')

    # Loop over columns for lineout
    for i, c in enumerate(cols_for_lineout):
        plt.plot(original_function[:, c], label=f'Original IFFT (col {c})')
        plt.plot(hilbert_amplitude[:, c], label=f'Hilbert (col {c})')
    plt.xlabel('y pixel')
    plt.legend()

    plt.tight_layout()

def plotting_interp(original_function, peaks_interp, function_name,  col_for_lineout, row_for_lineout):

    plt.close()
    plt.subplot(2,2,1)
    plt.title(f'Original function: {function_name}')
    plt.imshow(original_function, origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.colorbar(label='Intensity')
     
    plt.subplot(2,2,2)
    plt.imshow(peaks_interp, origin='lower')
    plt.colorbar(label='Interpolated value')
    plt.title('Interpolated amplitude of 2D Wave')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 2, 3)
    plt.title(f'Lineout (rows: {row_for_lineout})')

    # Loop over rows for lineout
    for i, r in enumerate(row_for_lineout):  # enumerate gives index and value
        plt.plot(original_function[r, :], label=f'Original (row {r})')
        plt.plot(peaks_interp[r, :], label=f'Interpolated (row {r})')
    plt.xlabel('x pixel')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title(f'Lineout (columns: {col_for_lineout})')

    # Loop over columns for lineout
    for i, c in enumerate(col_for_lineout):
        plt.plot(original_function[:, c], label=f'Original (col {c})')
        plt.plot(peaks_interp[:, c], label=f'Interpolated (col {c})')
    plt.xlabel('y pixel')
    plt.legend()

    plt.tight_layout()

def plot_image(image):
    
    plt.imshow(image, origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    #plt.colorbar()

def plot_image_with_crop(image, rows_rectangle, cols_rectangle):

    crop_rectangle = patches.Rectangle(
        (cols_rectangle[0], rows_rectangle[0]),  # Bottom-left corner
        cols_rectangle[1] - cols_rectangle[0],  # Width
        rows_rectangle[1] - rows_rectangle[0],  # Height
        edgecolor='red', facecolor='none', linewidth=1
    )

    plt.imshow(image, origin='lower')
    plt.gca().add_patch(crop_rectangle)

def plot_image_with_circle_boundary(image, circle_center, radius):

    square = patches.Rectangle(
        (circle_center[1]-radius, circle_center[0]-radius),  # Bottom-left corner
        2*radius,  # Width
        2*radius,  # Height
        edgecolor='red', facecolor='none', linewidth=1
    )

    # Add a circular boundary
    circle = patches.Circle(
        (circle_center[1], circle_center[0]),  # Center of the circle
        radius,  # Radius of the circle
        edgecolor='blue', facecolor='none', linewidth=1
    )

    plt.imshow(image, origin='lower')
    plt.gca().add_patch(square)
    plt.gca().add_patch(circle)

def plot_shifted_fft(to_keep, fft_image, width, height):

    plt.close()

    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    x_to_keep = [int((0.5-to_keep/2)*width), int((0.5+to_keep/2)*width)]
    y_to_keep = [int((0.5-to_keep/2)*height), int((0.5+to_keep/2)*height)]
    
    abs_fft_shifted = np.abs(fftshift(fft_image))

    plt.imshow(np.log(abs_fft_shifted + 1)[y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]], extent=(freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]))
    # Add axis labels and title
    plt.xlabel('x freq. (cycles/pix.)')
    plt.ylabel('y freq. (cycles/pix.)')

    # Add colorbar with vertical label
    cbar = plt.colorbar(label=r'$log(\hat{I}+1)$', pad=0.02)
    cbar.ax.set_ylabel(r'$log(\hat{I}+1)$', labelpad=15)

def plot_fft(fft_image, edge_coords):

    plt.close()

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pz3 = edge_coords[8]
    t_pz3 = edge_coords[9]
    l_pz3 = edge_coords[10]
    r_pz3 = edge_coords[11]
    b_pz4 = edge_coords[12]
    t_pz4 = edge_coords[13]
    l_pz4 = edge_coords[14]
    r_pz4 = edge_coords[15]
    b_pp1 = edge_coords[16]
    t_pp1 = edge_coords[17]
    l_pp1 = edge_coords[18]
    r_pp1 = edge_coords[19]
    b_pp2 = edge_coords[20]
    t_pp2 = edge_coords[21]
    l_pp2 = edge_coords[22]
    r_pp2 = edge_coords[23]
    b_pm1 = edge_coords[24]
    t_pm1 = edge_coords[25]
    l_pm1 = edge_coords[26]
    r_pm1 = edge_coords[27]
    b_pm2 = edge_coords[28]
    t_pm2 = edge_coords[29]
    l_pm2 = edge_coords[30]
    r_pm2 = edge_coords[31]

    # FFT with masking rectangles shown
    bottom_left_pz1 = [b_pz1 - 0.5, l_pz1 - 0.5]
    bottom_left_pz2 = [b_pz2 - 0.5, l_pz2 - 0.5]
    bottom_left_pz3 = [b_pz3 - 0.5, l_pz3 - 0.5]
    bottom_left_pz4 = [b_pz4 - 0.5, l_pz4 - 0.5]
    bottom_left_pp1 = [b_pp1 - 0.5, l_pp1 - 0.5]
    bottom_left_pp2 = [b_pp2 - 0.5, l_pp2 - 0.5]
    bottom_left_pm1 = [b_pm1 - 0.5, l_pm1 - 0.5]
    bottom_left_pm2 = [b_pm2 - 0.5, l_pm2 - 0.5]


    bottom_lefts = np.array([bottom_left_pz1, bottom_left_pz2, bottom_left_pz3, bottom_left_pz4, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])

    widths = np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pz3 - l_pz3), (r_pz4 - l_pz4), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = np.array([(t_pz1 - b_pz1), (t_pz2 - b_pz2), (t_pz3 - b_pz3), (t_pz4 - b_pz4), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

    rectangles = []
    for i in range(len(widths)):
        rectangle = patches.Rectangle((bottom_lefts[i]),  widths[i],  heights[i],  edgecolor='red', facecolor='none', linewidth=1)
        rectangles.append(rectangle)
    
    abs_fft = np.abs(fft_image)

    plt.title('FFT')
    plt.imshow(np.log(abs_fft+1), origin='lower', label='log(FFT+1)')
    plt.colorbar()

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

def plot_fft_45(fft_image, edge_coords):

    #plt.close()

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pp1 = edge_coords[8]
    t_pp1 = edge_coords[9]
    l_pp1 = edge_coords[10]
    r_pp1 = edge_coords[11]
    b_pp2 = edge_coords[12]
    t_pp2 = edge_coords[13]
    l_pp2 = edge_coords[14]
    r_pp2 = edge_coords[15]
    b_pm1 = edge_coords[16]
    t_pm1 = edge_coords[17]
    l_pm1 = edge_coords[18]
    r_pm1 = edge_coords[19]
    b_pm2 = edge_coords[20]
    t_pm2 = edge_coords[21]
    l_pm2 = edge_coords[22]
    r_pm2 = edge_coords[23]

    # FFT with masking rectangles shown
    bottom_left_pz1 = [b_pz1 - 0.5, l_pz1 - 0.5]
    bottom_left_pz2 = [b_pz2 - 0.5, l_pz2 - 0.5]
    bottom_left_pp1 = [b_pp1 - 0.5, l_pp1 - 0.5]
    bottom_left_pp2 = [b_pp2 - 0.5, l_pp2 - 0.5]
    bottom_left_pm1 = [b_pm1 - 0.5, l_pm1 - 0.5]
    bottom_left_pm2 = [b_pm2 - 0.5, l_pm2 - 0.5]


    bottom_lefts = np.array([bottom_left_pz1, bottom_left_pz2, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])

    widths = np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = np.array([(t_pz1 - b_pz1), (t_pz2 - b_pz2), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

    rectangles = []
    for i in range(len(widths)):
        rectangle = patches.Rectangle((bottom_lefts[i]),  widths[i],  heights[i],  edgecolor='red', facecolor='none', linewidth=1)
        rectangles.append(rectangle)
    
    abs_fft = np.abs(fft_image)

    plt.title('FFT')
    plt.imshow(np.log(abs_fft+1), origin='lower', label='log(FFT+1)')
    plt.colorbar()

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])


def plot_masked_fft(masked_fft_pz, masked_fft_pp, masked_fft_pm):

    plt.close()
    plt.subplot(2,2,1)
    plt.title('FFTs')
    plt.imshow(np.log(np.abs(masked_fft_pz)+1), origin='lower')
    plt.title('Masked FFT +0')

    plt.subplot(2,2,2)
    plt.imshow(np.log(np.abs(masked_fft_pp)+1), origin='lower')
    plt.title('Masked FFT ++')

    plt.subplot(2,2,3)
    plt.imshow(np.log(np.abs(masked_fft_pm)+1), origin='lower')
    plt.title('Masked FFT +-')

def plot_shifted_fft_with_windows(abs_fft_shifted, width, height, edge_coords):

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pz3 = edge_coords[8]
    t_pz3 = edge_coords[9]
    l_pz3 = edge_coords[10]
    r_pz3 = edge_coords[11]
    b_pz4 = edge_coords[12]
    t_pz4 = edge_coords[13]
    l_pz4 = edge_coords[14]
    r_pz4 = edge_coords[15]
    b_pp1 = edge_coords[16]
    t_pp1 = edge_coords[17]
    l_pp1 = edge_coords[18]
    r_pp1 = edge_coords[19]
    b_pp2 = edge_coords[20]
    t_pp2 = edge_coords[21]
    l_pp2 = edge_coords[22]
    r_pp2 = edge_coords[23]
    b_pm1 = edge_coords[24]
    t_pm1 = edge_coords[25]
    l_pm1 = edge_coords[26]
    r_pm1 = edge_coords[27]
    b_pm2 = edge_coords[28]
    t_pm2 = edge_coords[29]
    l_pm2 = edge_coords[30]
    r_pm2 = edge_coords[31]

    # Define the frequency axes
    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    # Create the frequency grids
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Shift the grids to match the FFT shift
    freq_x_grid_shifted = fftshift(freq_x_grid)
    freq_y_grid_shifted = fftshift(freq_y_grid)

    plt.imshow(np.log(abs_fft_shifted + 1), extent=(np.min(freq_x), np.max(freq_x), np.min(freq_y), np.max(freq_y)), origin='lower') # [y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]] freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]
    # Add axis labels and title
    plt.xlabel('x freq. (cycles/pix.)', fontsize=16)
    plt.ylabel('y freq. (cycles/pix.)', fontsize=16)

    # Add colorbar with vertical label
    cbar = plt.colorbar(label=r'$log(\hat{I}+1)$', pad=0.02)
    cbar.ax.set_ylabel(r'$log(\hat{I}+1)$', rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=14) 

    # Customize ticks
    plt.xticks(np.arange(-.1, .15, 0.05), fontsize=14)
    plt.yticks(fontsize=14)

    # FFT with masking rectangles shown
    bottom_left_pz1 = [freq_y_grid_shifted[-b_pz1 + height//2][0], freq_x_grid_shifted[0][l_pz1  + width//2]]
    bottom_left_pz2 = [freq_y_grid_shifted[-b_pz2 + height//2][0], freq_x_grid_shifted[0][l_pz2- width//2]]
    bottom_left_pz3 = [freq_y_grid_shifted[b_pz3 - height//2][0], freq_x_grid_shifted[0][l_pz3  + width//2]]
    bottom_left_pz4 = [freq_y_grid_shifted[b_pz4 - height//2][0], freq_x_grid_shifted[0][l_pz4- width//2]]
    bottom_left_pp1 = [freq_y_grid_shifted[b_pp1 + height//2][0], freq_x_grid_shifted[0][l_pp1 + width//2]]
    bottom_left_pp2 =  [freq_y_grid_shifted[b_pp2 - height//2][0], freq_x_grid_shifted[0][l_pp2 - width//2]]
    bottom_left_pm1 = [freq_y_grid_shifted[b_pm1 - height//2][0], freq_x_grid_shifted[0][l_pm1 + width//2]]
    bottom_left_pm2 = [freq_y_grid_shifted[b_pm2 + height//2][0], freq_x_grid_shifted[0][l_pm2 - width//2]]

    bottom_lefts = np.array([bottom_left_pz1, bottom_left_pz2, bottom_left_pz3, bottom_left_pz4, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])
    
    widths = ((np.max(freq_x) - np.min(freq_x))/width) * np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pz3 - l_pz3), (r_pz4 - l_pz4), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = ((np.max(freq_y) - np.min(freq_y))/height) * np.array([(t_pz1 - b_pz1), (t_pz2 - b_pz2), (t_pz3 - b_pz3), (t_pz4 - b_pz4), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

    rectangles = []
    for i in range(len(widths)):
        rectangle = patches.Rectangle((bottom_lefts[i]),  widths[i],  heights[i],  edgecolor='red', facecolor='none', linewidth=1)
        rectangles.append(rectangle)
    

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

    # Set axis limits
    plt.xlim(-0.13, 0.13) 
    plt.ylim(-0.08, 0.08)

    plt.tight_layout()

def plot_shifted_fft_with_windows_45(abs_fft_shifted, width, height, edge_coords):

    b_pz1 = edge_coords[0]
    t_pz1 = edge_coords[1]
    l_pz1 = edge_coords[2]
    r_pz1 = edge_coords[3]
    b_pz2 = edge_coords[4]
    t_pz2 = edge_coords[5]
    l_pz2 = edge_coords[6]
    r_pz2 = edge_coords[7]
    b_pp1 = edge_coords[8]
    t_pp1 = edge_coords[9]
    l_pp1 = edge_coords[10]
    r_pp1 = edge_coords[11]
    b_pp2 = edge_coords[12]
    t_pp2 = edge_coords[13]
    l_pp2 = edge_coords[14]
    r_pp2 = edge_coords[15]
    b_pm1 = edge_coords[16]
    t_pm1 = edge_coords[17]
    l_pm1 = edge_coords[18]
    r_pm1 = edge_coords[19]
    b_pm2 = edge_coords[20]
    t_pm2 = edge_coords[21]
    l_pm2 = edge_coords[22]
    r_pm2 = edge_coords[23]

    # Define the frequency axes
    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    # Create the frequency grids
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Shift the grids to match the FFT shift
    freq_x_grid_shifted = fftshift(freq_x_grid)
    freq_y_grid_shifted = fftshift(freq_y_grid)

    plt.imshow(np.log(abs_fft_shifted + 1), extent=(np.min(freq_x), np.max(freq_x), np.min(freq_y), np.max(freq_y)), origin='lower') # [y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]] freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]
    # Add axis labels and title
    plt.xlabel('x freq. (cycles/pix.)', fontsize=16)
    plt.ylabel('y freq. (cycles/pix.)', fontsize=16)

    # Add colorbar with vertical label
    cbar = plt.colorbar(label=r'$log(\hat{I}+1)$', pad=0.02)
    cbar.ax.set_ylabel(r'$log(\hat{I}+1)$', rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # Customize ticks
    plt.xticks(np.arange(-.1, .15, 0.05), fontsize=14)
    plt.yticks(fontsize=14)

    # FFT with masking rectangles shown
    bottom_left_pz1 = [freq_y_grid_shifted[b_pz1 - height//2][0], freq_x_grid_shifted[0][l_pz1  + width//2]]
    bottom_left_pz2 = [freq_y_grid_shifted[b_pz2 + height//2][0], freq_x_grid_shifted[0][l_pz2 - width//2]]
    bottom_left_pp1 = [freq_y_grid_shifted[b_pp1 - height//2][0], freq_x_grid_shifted[0][l_pp1 + width//2]]
    bottom_left_pp2 =  [freq_y_grid_shifted[b_pp2 + height//2][0], freq_x_grid_shifted[0][l_pp2 - width//2]]
    bottom_left_pm1 = [freq_y_grid_shifted[b_pm1 - height//2][0], freq_x_grid_shifted[0][l_pm1 + width//2]]
    bottom_left_pm2 = [freq_y_grid_shifted[b_pm2 + height//2][0], freq_x_grid_shifted[0][l_pm2 - width//2]]

    bottom_lefts = np.array([bottom_left_pz1, bottom_left_pz2, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])
    
    widths = ((np.max(freq_x) - np.min(freq_x))/width) * np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = ((np.max(freq_y) - np.min(freq_y))/height) * np.array([(t_pz1 - b_pz1), (t_pz2 - b_pz2), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

    rectangles = []
    for i in range(len(widths)):
        rectangle = patches.Rectangle((bottom_lefts[i]),  widths[i],  heights[i],  edgecolor='red', facecolor='none', linewidth=1)
        rectangles.append(rectangle)
    

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

    # Set axis limits
    plt.xlim(-0.13, 0.13) 
    plt.ylim(-0.08, 0.08)

    plt.tight_layout()

def plot_all_thetas(theta_out_1_pp, theta_out_1_pm, theta_out_1_arith, theta_out_2, edge_cols, edge_rows):

    plt.close()

    extent = (edge_cols[0], edge_cols[1], edge_rows[0], edge_rows[1])

    plt.subplot(2,2,1)
    plt.imshow(np.rad2deg(theta_out_1_pp), extent=extent, origin='lower')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $2|A(+,+)|/|A(+,0)|$')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(np.rad2deg(theta_out_1_pm), extent=extent, origin='lower')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $2|A(+,-)|/|A(+,0)|$')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(np.rad2deg(theta_out_1_arith), extent=extent, origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $(|A(+,+)|+|A(+,-)|)/|A(+,0)|$')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(np.rad2deg(theta_out_2), extent=extent, origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $(4|A(+,+)||A(+,-)|)/|A(+,0)|^2$')
    plt.colorbar()

def contour_plot(theta, circle_center, R, crop_image, input_angle):

    if crop_image == 'No':
        x_min = int(circle_center[1] - R / np.sqrt(2))
        x_max = int(circle_center[1] + R / np.sqrt(2))

        x = np.arange(x_min, x_max, 1)

        y_min = int(circle_center[0] - R / np.sqrt(2))
        y_max = int(circle_center[0] + R / np.sqrt(2))

        y = np.arange(y_min, y_max, 1)

        X, Y = np.meshgrid(x, y)

        offset_angle = np.rad2deg(theta)-input_angle
        
        contour = plt.contourf(X, Y, offset_angle[y_min:y_max, x_min:x_max], levels=20)
    
    elif crop_image == 'Yes':
        contour = plt.contourf(np.rad2deg(theta), levels=20)

    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    
    cbar = plt.colorbar(contour)
    cbar.set_label('Offset angle (degrees)')  # Adding label to the colorbar

def plot_theta_central_row(width, angle_name, thetas, theta_calibrated, radius_of_circle, circle_center, cols_lim):

    plt.close()

    theta_out_1_pp = thetas[0]
    theta_out_1_pm = thetas[1]
    theta_out_1_arith = thetas[2]
    theta_out_2 = thetas[3]

    x = np.arange(0, width, 1)

    # y_linear_fit = linear_fit_coefficients[0]*x + linear_fit_coefficients[1]

    # theta_calib = np.rad2deg(theta_out_1_arith)[circle_center[0], :] + (angle_name-154.5) - linear_fit_coefficients[1] - linear_fit_coefficients[0]*x

    # as proportion of diamater
    rows_to_plot_prop = [0.25, 0.75]
    rows_to_plot = [int(radius_of_circle*(2*rows_to_plot_prop[0]-1) + circle_center[0]), int(radius_of_circle*(2*rows_to_plot_prop[1]-1) + circle_center[0])]
    distance_to_edge_circle_from_center = [np.sqrt(np.abs(1-4*rows_to_plot_prop[0]**2))*radius_of_circle, np.sqrt(np.abs(1-4*rows_to_plot_prop[1]**2))*radius_of_circle]

    theta_min_plot = -150
    theta_max_plot = 150

    # plt.plot(x, np.rad2deg(theta_out_1_pp[height//2, :]), label=fr'$\theta_1pp$')
    # plt.plot(x, np.rad2deg(theta_out_1_pm[height//2, :]), label=fr'$\theta_1pm$')
    plt.plot([x[0], x[-1]], [angle_name-154.5, angle_name-154.5], label='Input')

    plt.plot(x, np.rad2deg(theta_out_1_arith[rows_to_plot[0], :]), label=fr'$\theta$ ({rows_to_plot_prop[0]}D)', linestyle='-', color='green')
    plt.plot(x, np.rad2deg(theta_out_1_arith[circle_center[0], :]), label=fr'$\theta$ (0.5D)', linestyle='-', color='blue')
    # plt.plot(x, np.rad2deg(theta_out_2[height//2, :]), label=fr'$\theta_2$')
    
    plt.plot(x, np.rad2deg(theta_out_1_arith[rows_to_plot[1], :]), label=fr'$\theta$ ({rows_to_plot_prop[1]}D)', linestyle='-', color='orange')

    # aplt.plot(x, y_linear_fit, label=fr'Linear fit to $\theta_1ar$')

    plt.plot(x, theta_calibrated[circle_center[0], :], label=fr'Calibrated $\theta$ (0.5D)', linestyle='--', color='blue')
    plt.plot(x, theta_calibrated[rows_to_plot[0], :], label=fr'Calibrated $\theta$ ({rows_to_plot_prop[0]}D)', linestyle='--', color='green')
    plt.plot(x, theta_calibrated[rows_to_plot[1], :], label=fr'Calibrated $\theta$ ({rows_to_plot_prop[1]}D)', linestyle='--', color='orange')

    # plt.plot([circle_center[1]-radius_of_circle, circle_center[1]-radius_of_circle], [theta_min_plot, theta_max_plot], linestyle='-', color='black', label='Edge of circle')
    # plt.plot([circle_center[1]+radius_of_circle, circle_center[1]+radius_of_circle], [theta_min_plot, theta_max_plot], linestyle='-', color='black')

    # plt.plot([circle_center[1]+distance_to_edge_circle_from_center[0], circle_center[1]+distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray', label='Edge of circle (for 0.25/0.75 D)')
    # plt.plot([circle_center[1]-distance_to_edge_circle_from_center[0], circle_center[1]-distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray')

    plt.plot([cols_lim[0], cols_lim[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='black', label='Interpolation limit')
    plt.plot([cols_lim[1], cols_lim[1]], [theta_min_plot, theta_max_plot], linestyle='-', color='black')

    # plt.plot([circle_center[1]+distance_to_edge_circle_from_center[0], circle_center[1]+distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray', label='Edge of circle (for 0.25/0.75 D)')
    # plt.plot([circle_center[1]-distance_to_edge_circle_from_center[0], circle_center[1]-distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray')

    plt.xlabel('x pixel')
    plt.ylabel('Angle (degrees)')
    plt.title('Row lineouts (uncalibrated and calibrated)')
    plt.legend()
    plt.ylim([angle_name-154.5-5, angle_name-154.5+10])
    plt.xlim([circle_center[1]-R-100, circle_center[1]+R+100])


def plot_theta_cols(height, theta_calibrated, angle_name, thetas, radius_of_circle, circle_center, rows_lim):

    plt.close()

    theta_out_1_arith = thetas[2]

    y = np.arange(0, height, 1)


    # as proportion of diamater
    cols_to_plot_prop = [0.25, 0.75]
    cols_to_plot = [int(radius_of_circle*(2*cols_to_plot_prop[0]-1) + circle_center[1]), int(radius_of_circle*(2*cols_to_plot_prop[1]-1) + circle_center[1])]
    distance_to_edge_circle_from_center = [np.sqrt(np.abs(1-4*cols_to_plot_prop[0]**2))*radius_of_circle, np.sqrt(np.abs(1-4*cols_to_plot_prop[1]**2))*radius_of_circle]
    
    theta_min_plot = -150
    theta_max_plot = 150
    
    plt.plot([y[0], y[-1]], [angle_name-154.5, angle_name-154.5], label='Input', color='red')
    
    plt.plot(y, np.rad2deg(theta_out_1_arith[:, cols_to_plot[0]]), label=fr'$\theta$ ({cols_to_plot_prop[0]}D)', linestyle='-', color='green')
    plt.plot(y, np.rad2deg(theta_out_1_arith[:, circle_center[1]]), label=fr'$\theta$ (0.5D)', linestyle='-', color='blue')
    plt.plot(y, np.rad2deg(theta_out_1_arith[:, cols_to_plot[1]]), label=fr'$\theta$ ({cols_to_plot_prop[1]}D)', linestyle='-', color='orange')

    plt.plot(y, theta_calibrated[:, circle_center[1]], label=fr'Calibrated $\theta$ (0.5D)', linestyle='--', color='blue')
    plt.plot(y, theta_calibrated[:, cols_to_plot[0]], label=fr'Calibrated $\theta$ ({cols_to_plot_prop[0]}D)', linestyle='--', color='green')
    plt.plot(y, theta_calibrated[:, cols_to_plot[1]], label=fr'Calibrated $\theta$ ({cols_to_plot_prop[1]}D)', linestyle='--', color='orange')
    
    # plt.plot([circle_center[0]-radius_of_circle, circle_center[0]-radius_of_circle], [theta_min_plot, theta_max_plot], linestyle='-', color='black', label='Edge of circle')
    # plt.plot([circle_center[0]+radius_of_circle, circle_center[0]+radius_of_circle], [theta_min_plot, theta_max_plot], linestyle='-', color='black')

    # plt.plot([circle_center[0]+distance_to_edge_circle_from_center[0], circle_center[0]+distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray')
    # plt.plot([circle_center[0]-distance_to_edge_circle_from_center[0], circle_center[0]-distance_to_edge_circle_from_center[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='gray', label='Edge of circle (for 0.25/0.75 D)')

    plt.plot([rows_lim[0], rows_lim[0]], [theta_min_plot, theta_max_plot], linestyle='-', color='black', label='Interpolation limit')
    plt.plot([rows_lim[1], rows_lim[1]], [theta_min_plot, theta_max_plot], linestyle='-', color='black')

    plt.xlabel('y pixel')
    plt.ylabel('Angle (degrees)')
    plt.title('Column lineouts (uncalibrated and calibrated)')
    plt.legend()
    plt.ylim([angle_name-154.5-10, angle_name-154.5+10])
    # plt.xlim([circle_center[0]-R-100, circle_center[0]+R+100])


def plot_calibrated_angle(theta, circle_center, R, crop_image):

    plt.imshow(theta, origin='lower') # extent=(x_min, x_max, y_min, y_max)

    if crop_image == 'No':
        x_min = int(circle_center[1]-R/np.sqrt(2))
        x_max = int(circle_center[1]+R/np.sqrt(2))

        y_min = int(circle_center[0]-R/np.sqrt(2))
        y_max = int(circle_center[0]+R/np.sqrt(2))

        # Add a rectangle
        ax = plt.gca()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)


    
    plt.title('Calibrated angle with interpolation limits (red)')
    # Add colorbar with label
    cbar = plt.colorbar()
    cbar.set_label('Angle (degrees)')

    


def cosine_1d(x_coord, y_constant, A, B, C, kx, ky, phi, D):
        return (A*x_coord + B*y_constant + C) * np.cos(kx * x_coord + ky * y_constant + phi) + D


def find_center(raw_image, width, height):

    # Columns

    # Fit the Gaussian to the data
    initial_guess = [50, 800, 300, 10, 1]  # Initial guesses for a, x0, sigma, b, p

    lower_bounds = [1, 0.1, 0.1, 0.1, 0.1] 
    upper_bounds = [200, 2000, 1000, 200, 10]

    mean_all_col_list = []
    for i in np.arange(0, width, 1):
        mean_all_col = np.mean(raw_image[:, i])
        mean_all_col_list.append(mean_all_col)

    x_data_col = np.arange(0, width, 1)
    y_data_col = mean_all_col_list

    params_col, covariance = curve_fit(gaussian, x_data_col, y_data_col, p0=initial_guess, bounds=(lower_bounds, upper_bounds))

    # Extract fitted parameters
    a_fit_col, x0_fit_col, sigma_fit_col, b_fit_col, p_fit_col = params_col

    center_col = int(x0_fit_col)

    # Rows
    mean_all_rows_list = []
    for i in np.arange(0, height, 1):
        mean_all_rows = np.mean(raw_image[i, :])
        mean_all_rows_list.append(mean_all_rows)

    x_data_row = np.arange(0, height, 1)
    y_data_row = mean_all_rows_list

    params_row, covariance = curve_fit(gaussian, x_data_row, y_data_row, p0=initial_guess, maxfev=10000)

    # Extract fitted parameters
    a_fit_row, x0_fit_row, sigma_fit_row, b_fit_row, p_fit_row = params_row

    center_row = int(x0_fit_row)

    center_point = [center_row, center_col]
    print(f'Centre point (r, c): ({center_point[0]}, {center_point[1]})')

    x_data = [x_data_row, x_data_col]
    y_data = [y_data_row, y_data_col]
    a_fit = [a_fit_row, a_fit_col]
    x0_fit = [x0_fit_row, x0_fit_col]
    sigma_fit = [sigma_fit_row, sigma_fit_col]
    b_fit = [b_fit_row, b_fit_col]
    p_fit = [p_fit_row, p_fit_col]

    params = [params_row, params_col]

    return center_point, y_data, x_data, params

def gaussian(x, a, x0, sigma, b, p):
    """
    Gaussian function.

    Parameters:
        x: Independent variable.
        a: Amplitude.
        x0: Mean (center of the peak).
        sigma: Standard deviation (width of the peak).
        b: Baseline offset.
        p: Power factor.

    Returns:
        Gaussian function value at x.
    """
    sigma = max(sigma, 1e-6)  # Prevent division by zero
    base = np.abs(x - x0) ** p  # Ensure valid exponentiation
    return b + (a * np.exp(-base / (sigma ** p)))

def plot_gaussian(x_data, y_data, params, label, title):

    a_fit, x0_fit, sigma_fit, b_fit, p_fit = params
    
    y_fit = gaussian(x_data, a_fit, x0_fit, sigma_fit, b_fit, p_fit)

    # Plot the data and the fitted curve
    plt.scatter(x_data, y_data, label=label, color="blue", s=10)
    plt.plot(x_data, y_fit, label=f"Fitted Gaussian\n(a={a_fit:.2f}, x0={x0_fit:.2f}, sigma={sigma_fit:.2f}, b={b_fit:.2f}, p={p_fit:.2f})", color="red")
    plt.title(title)
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.legend()
    plt.grid()


def find_noise_of_image(mean_columns, raw_image, threshold_for_cum_diff=0.001):

    mean_columns = np.array(mean_columns)

    # Compute the cumulative moving average (CMA)
    cumulative_avg = np.cumsum(mean_columns) / np.arange(1, len(mean_columns) + 1)

    print('1')
    
    # Compute differences in CMA
    diff = np.diff(cumulative_avg)  # Difference between consecutive elements in CMA
    print('2')
    index_at_increase = np.where(diff > threshold_for_cum_diff)[0][0]
    print('3')
    noise_image = np.mean(raw_image[:, 0:index_at_increase])
    noise_image_std = np.std(raw_image[:, 0:index_at_increase])

    return noise_image, noise_image_std


def find_edge_of_circle_spline(mean_col_arr, mean_row_arr, width, height, circle_center, gradient_threshold, s):

    mean_col_arr = np.array(mean_col_arr)  # Convert to NumPy array
    mean_row_arr = np.array(mean_row_arr)  # Convert to NumPy array

    # For cols left side
    width_l = np.arange(0, circle_center[1], 1)
    mean_cols_l = mean_col_arr[0:circle_center[1]]

    spline_l = UnivariateSpline(width_l, mean_cols_l, s=s)

    edge_left = np.where(np.gradient(spline_l(width_l)) > gradient_threshold)[0][0]

    # For cols right side
    width_r = np.arange(circle_center[1], width, 1)
    mean_cols_r = mean_col_arr[circle_center[1]:width]

    spline_r = UnivariateSpline(width_r, mean_cols_r, s=s)

    edge_right = np.where(np.gradient(spline_r(width_r)) < -gradient_threshold)[0][-1] + circle_center[1]

    # For rows bottom
    height_b = np.arange(0, circle_center[0], 1)
    mean_rows_b = mean_row_arr[0:circle_center[0]]

    spline_b = UnivariateSpline(height_b, mean_rows_b, s=s)

    edge_bottom = np.where(np.gradient(spline_b(height_b)) > gradient_threshold)[0][0]
    
    # For rows top
    height_t = np.arange(circle_center[0], height, 1)
    mean_rows_t = mean_row_arr[circle_center[0]: height]

    spline_t = UnivariateSpline(height_t, mean_rows_t, s=s)

    edge_top = np.where(np.gradient(spline_t(height_t)) < -gradient_threshold)[0][-1] + circle_center[0]

    print('Bottom edge: ', edge_bottom)
    print('Top edge: ', edge_top)
    print('Left edge: ', edge_left)
    print('Right edge: ', edge_right)

    indices_row = [int(edge_bottom), int(edge_top)]
    indices_col = [int(edge_left), int(edge_right)]

    # plt.subplot(2,2,1)
    # plt.plot(spline_l(width_l))
    # plt.subplot(2,2,2)
    # plt.plot(spline_r(width_r))
    # plt.subplot(2,2,3)
    # plt.plot(spline_t(height_t))
    # plt.subplot(2,2,3)
    # plt.plot(spline_b(height_b))



    #plt.show()

    

    return indices_row, indices_col
    
def plane_of_best_fit(x_data, y_data, z_data):

    # Validate inputs
    if z_data.shape != (len(y_data), len(x_data)):
        raise ValueError("Shape of z_data must match len(y_data) x len(x_data).")

    # Create grid of x, y coordinates
    X, Y = np.meshgrid(x_data, y_data)

    # Flatten the grids and z_data
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = z_data.ravel()

    # Create the design matrix
    A = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))

    # Solve for [a, b, c] using least squares
    coefficients, residuals, _, _ = np.linalg.lstsq(A, Z_flat, rcond=None)

    # Compute predicted Z values
    Z_pred = A @ coefficients

    # Compute R^2
    Z_mean = np.mean(Z_flat)
    TSS = np.sum((Z_flat - Z_mean) ** 2)
    RSS = np.sum((Z_flat - Z_pred) ** 2)
    R2 = 1 - (RSS / TSS)

    return coefficients, R2

def apply_top_hat_image(image, circle_center, width, height):

    if circle_center[0] > height//2 and circle_center[1] > width//2:
        edge_rows = [2*circle_center[0] - height, height]
        edge_cols = [2*circle_center[1] - width, width]
    elif circle_center[0] > height//2 and circle_center[1] <= width//2:
        edge_rows = [2*circle_center[0] - height, height]
        edge_cols = [0, 2*circle_center[1]]
    else:
        raise(ValueError('write more code'))

    masked_image = np.zeros_like(image)
    masked_image[edge_rows[0]:edge_rows[1], edge_cols[0]:edge_cols[1]] = image[edge_rows[0]:edge_rows[1], edge_cols[0]:edge_cols[1]]

    return masked_image, edge_rows, edge_cols

# Function to extract diagonal elements in a given direction
def extract_diagonal(arr, start_row, start_col, step_r, step_c):
    """Extract elements along a custom diagonal direction."""
    rows, cols = arr.shape
    diagonal_elements = []
    
    r, c = start_row, start_col
    while r < rows and c < cols:
        diagonal_elements.append(arr[r, c])
        r += step_r  # Move 2 rows down
        c += step_c  # Move 1 column right
    
    return diagonal_elements

## RUN CODE ##

if __name__ == '__main__':

    folderpath = '/home/sfv514/Documents/Project/Camera/Saved Images/21-02-25/Full calibration - coffin/'

    filename = 'Acquisition_135529' # 131157 131232 131251 131315 131335 131356 131416 131447 131511 

    filepath = f'{folderpath}{filename}.raw'

    df_meta_data = pd.read_csv(f'{folderpath}image_metadata.csv')

    # Extract angle_name based on the filename
    angle_name = df_meta_data.loc[df_meta_data['Image Name'] == filename, 'lp1'].values[0]
    
    width_original = 1920
    height_original = 1200

    a = 1920/1200

    pixel_size = 5.86e-6

    ## User settings

    rotated_system = 'Yes'

    if rotated_system == 'Yes':
        angle_name = angle_name - 45
        print(f'Input angle: {angle_name-154.5}°')

    elif rotated_system == 'No':
        print(f'Input angle: {angle_name-154.5}°')
    

    ## Pre-processing
    # Choose whether to crop the image
    crop_raw_image = 'No'
    # If cropping, choose shape. Rectangle or square. Rectangle retains original aspect ratio
    crop_shape = 'square'
    # Choose whether to apply a gaussian window and sigma to (cropped) image

    image_top_hat = 'Yes'

    image_gaussian = 'No'
    sigma = 2
    # Choose whether to apply a hanning window to (cropped) image
    image_hanning = 'Yes'
    # Choose whether to apply zero padding to the outside of the image
    add_padding_to_image = 'No'
    # Choose padding percentage
    percentage_of_pad = 10

    # Mid-processing
    # Select method of amplitude finding
    method = 'hilbert'
    # Choose whether to apply a hanning window to the masked FFTs
    fft_window = 'hanning'
    alpha = 1 # for tukey window. 1 returns Hann window, 0 returns top hat.

    # Old method
    n_stds = 110
    n_stds_pz = 5

    # Choose whether to add a Gaussian window to the IFFTs before applying Hilbert transform
    gaussian_pre_smoothing = 'No'
    median_post_smoothing = 'No'

    global_calibration = 'No'

    plot_label = ''


    ## START OF CODE ##

    raw_image = read_image(filepath=filepath, width=width_original, height=height_original)
    
    original_image = raw_image

    circle_center, y_data, x_data, params = find_center(raw_image=raw_image, width=width_original, height=height_original)

    # noise_image, noise_image_std = find_noise_of_image(y_data[1], raw_image=original_image)
    
    # print('Image noise: ', noise_image)
    # print('Image noise std: ', noise_image_std)
    
    #edge_of_circle_cols, edge_of_circle_rows = find_edge_of_circle(y_data[1], y_data[0], threshold=8, width=width_original, height=height_original, circle_center=circle_center)
    #edge_of_circle_cols, edge_of_circle_rows = find_edge_of_circle_cubic(y_data[1], y_data[0], width=width_original, height=height_original, circle_center=circle_center)
    edge_of_circle_rows, edge_of_circle_cols = find_edge_of_circle_spline(y_data[1], y_data[0], width=width_original, height=height_original, circle_center=circle_center, gradient_threshold=0.01, s=4)#noise_image + 3*noise_image_std) # 0.28

    R = round(np.min([circle_center[1]-edge_of_circle_cols[0], edge_of_circle_cols[1] - circle_center[1], circle_center[0] - edge_of_circle_rows[0], edge_of_circle_rows[1] - circle_center[0]]))
    print(f'Radius of circle: {R} pix.')


    if crop_raw_image == 'Yes':
        if crop_shape == 'rectangle':
            rows_rectangle, cols_rectangle, raw_image_cropped = crop_circle(R=R, circle_center=circle_center, a=a, raw_image=raw_image)
        elif crop_shape == 'square':
            rows_rectangle, cols_rectangle, raw_image_cropped = crop_into_square(R=R, circle_center=circle_center, raw_image=raw_image)

        raw_image = raw_image_cropped
        width = np.shape(raw_image)[1]
        height = np.shape(raw_image)[0]

    elif crop_raw_image == 'No':
        
        width = width_original
        height = height_original

        cols_rectangle = [0, width]
        rows_rectangle = [0, height]
    
    if image_top_hat == 'Yes':
        # Hanning window should extend all the way to the closest edges (row/column) from the center to preserve as much data as possible. 
        # The rest of the image should be zero ('top hat').
        raw_image, edge_rows_hann, edge_cols_hann = apply_top_hat_image(raw_image, circle_center=circle_center, width=width, height=height)

        print('Hanning window limits (columns): ', edge_cols_hann)
        print('Hanning window limits (rows): ', edge_rows_hann)

        raw_image = apply_hanning_window_to_image(raw_image, rows_for_window=edge_rows_hann, cols_for_window=edge_cols_hann)
        print('Top hat and Hanning applied to image')
    
    if image_gaussian == 'Yes':
        raw_image = apply_gaussian_window_to_image(raw_image, sigma)

    if add_padding_to_image == 'Yes':
        raw_image, height, width, row_pad, col_pad = pad_image(image=raw_image, percentage_of_pad=percentage_of_pad)
        edge_of_circle_cols = np.array(edge_of_circle_cols) + col_pad
        edge_of_circle_rows = np.array(edge_of_circle_rows) + row_pad

    fft_image, abs_fft, noise_floor, noise_std, *_ = fft_2d(raw_image, width, height)

    print('FFT noise:', noise_floor)
    print('FFT noise std: ', noise_std)

    if rotated_system == 'Yes':

        print('Rotated system analysis selected')

        all_bright_spot_coords = find_peaks_fft_45(abs_fft, width, height)

        #edge_coords = calculate_edges_45(abs_fft, threshold=threshold, threshold_pz=threshold_pz, all_bright_spot_coords=all_bright_spot_coords, height=height)
        edge_coords = calculate_edges_45_grad(image=median_filter(abs_fft, size=3), all_bright_spot_coords=all_bright_spot_coords, height=height)

        mask_pz, mask_pp, mask_pm = define_masks_45(fft_window=fft_window, edge_coords=edge_coords, height=height, width=width, alpha=alpha)

        print(f"Coords_PZ1: {all_bright_spot_coords[0]}")
        print(f"Coords_PZ2: {all_bright_spot_coords[1]}")
        print(f"Coords_PP1: {all_bright_spot_coords[2]}")
        print(f"Coords_PP2: {all_bright_spot_coords[3]}")
        print(f"Coords_PM1: {all_bright_spot_coords[4]}")
        print(f"Coords_PM2: {all_bright_spot_coords[5]}")


    elif rotated_system == 'No':

        print('Unrotated system analysis selected')

        all_bright_spot_coords = find_peaks_fft(abs_fft, width, height)

        edge_coords = calculate_edges(abs_fft, threshold=threshold, threshold_pz=threshold_pz, all_bright_spot_coords=all_bright_spot_coords, height=height)

        mask_pz, mask_pp, mask_pm = define_masks(fft_window=fft_window, edge_coords=edge_coords, height=height, width=width, alpha=alpha)

        print(f"Coords_PZ1: {all_bright_spot_coords[0]}")
        print(f"Coords_PZ2: {all_bright_spot_coords[1]}")
        print(f"Coords_PZ3: {all_bright_spot_coords[2]}")
        print(f"Coords_PZ4: {all_bright_spot_coords[3]}")
        print(f"Coords_PP1: {all_bright_spot_coords[4]}")
        print(f"Coords_PP2: {all_bright_spot_coords[5]}")
        print(f"Coords_PM1: {all_bright_spot_coords[6]}")
        print(f"Coords_PM2: {all_bright_spot_coords[7]}")

    
    masks = [mask_pz, mask_pp, mask_pm]

    masked_fft_pz, masked_fft_pp, masked_fft_pm = apply_masks(mask_pz, mask_pp, mask_pm, fft_image)
    masked_ffts = [masked_fft_pz, masked_fft_pp, masked_fft_pm]

    # Perform inverse Fourier transforms on the masked FFTs
    ifft_pz = np.real(ifft2(masked_fft_pz)) # (+0)
    ifft_pp = np.real(ifft2(masked_fft_pp)) # (++)
    ifft_pm = np.real(ifft2(masked_fft_pm)) # (+-)

    iffts = [ifft_pz, ifft_pp, ifft_pm]

    A_pz, A_pp, A_pm = find_amplitudes(ifft_pz, ifft_pp, ifft_pm, method, gaussian_pre_smoothing=gaussian_pre_smoothing, median_post_smoothing=median_post_smoothing)

    amplitudes = [A_pz, A_pp, A_pm]

    theta_out_1_pp, theta_out_1_pm, theta_out_arith, theta_out_2 = calculate_theta(A_pz, A_pp, A_pm, height=height, width=width)

    thetas = [theta_out_1_pp, theta_out_1_pm, theta_out_arith, theta_out_2]

    cols_for_lineout = [int(0.2*width), int(0.5*width), int(0.8*width)]
    rows_for_lineout = [int(0.2*height), int(0.5*height), int(0.8*height)]
    
    ifft_pz = np.real(iffts[0])
    ifft_pp = np.real(iffts[1])
    ifft_pm = np.real(iffts[2])

    A_pz = amplitudes[0]
    A_pp = amplitudes[1]
    A_pm = amplitudes[2]

    masked_fft_pz = masked_ffts[0]
    masked_fft_pp = masked_ffts[1]
    masked_fft_pm = masked_ffts[2]

    mask_pz = masks[0]
    mask_pp = masks[1]
    mask_pm = masks[2]

    theta_out_1_pp = thetas[0]
    theta_out_1_pm = thetas[1]
    theta_out_arith = thetas[2]
    theta_out_2 = thetas[3]
    
    # Divide by sqrt(2) to define maximum square inside circle
    cols_interp_lim = [int(circle_center[1]-R/np.sqrt(2)), int(circle_center[1]+R/np.sqrt(2))]
    rows_interp_lim = [int(circle_center[0]-R/np.sqrt(2)), int(circle_center[0]+R/np.sqrt(2))]

    print('Columns for interpolation limits: ', cols_interp_lim)
    print('Rows for interpolation limits: ', rows_interp_lim)
    
    
    col_min = cols_interp_lim[0]
    col_max = cols_interp_lim[1]

    row_min = rows_interp_lim[0]
    row_max = rows_interp_lim[1]

    xmin = (col_min - width//2)*pixel_size
    xmax = (col_max - width//2)*pixel_size

    ymin = (row_min - height//2)*pixel_size
    ymax = (row_max - height//2)*pixel_size

    
    x_values = pixel_size*np.arange(-width//2, width//2, 1)
    y_values = pixel_size*np.arange(-height//2, height//2, 1)

    X, Y = np.meshgrid(x_values, y_values)

    if global_calibration == 'Yes':
        print('### Using global calibration ###')
        calibration_function = -0.00280*X + 0.00283*Y + 2.930
        calibration_function = -0.00274*X + 0.00350*Y + 2.64
        calibration_function = 0.000366*X + 0.00419*Y - 1.291 # 11-2-25
        calibration_function = 6.16678529e+01*X + 7.17400656e+02*Y + 1.33225570 # 13-2-25, theta coeffs includeds afterwards for dimension compatability

        coefficients = [-1.04898391e-03,  4.21341916e-02,  7.17400656e+02,  6.16678529e+01, 1.33225570e+00]
        coefficients = [-1.75495604e-03,  7.80860530e-02,  7.77445127e+02,  1.16440498e+02, 1.77009372e+00] # 21/2 coffin calibration

        if angle_name - 154.5 < 45:
            offset_theta = coefficients[0]*np.rad2deg(theta_out_arith)**2 + coefficients[1]*np.rad2deg(theta_out_arith) + coefficients[2]*Y + coefficients[3]*X + coefficients[4]
        else:
            offset_theta = coefficients[0]*(np.rad2deg(theta_out_arith))**2 + coefficients[1]*(np.rad2deg(theta_out_arith)) + coefficients[2]*Y + coefficients[3]*X + coefficients[4]
        # Print the plane equation
        print(fr'Calibration function = {coefficients[3]:.2f}x + {coefficients[2]:.2f}y + {coefficients[0]:.5f}*theta^2 + {coefficients[1]:.5f}*theta + {coefficients[4]:.2f}')

        if angle_name - 154.5 < 45:
            theta_calib = np.rad2deg(theta_out_arith) - offset_theta
        else:
            print('1')
            theta_calib = np.rad2deg(theta_out_arith) + offset_theta

        print(f'Calibrated central theta (global) = {theta_calib[height//2, width//2] :.3f}°')
    elif global_calibration == 'No':
        # Fitting 
        coefficients, r_squared = plane_of_best_fit(x_data=np.arange(xmin, xmax, pixel_size), y_data=np.arange(ymin, ymax, pixel_size), 
                                                z_data=np.rad2deg(theta_out_arith)[row_min:row_max,col_min:col_max] - (angle_name-154.5))
        calibration_function = coefficients[0]*X + coefficients[1]*Y + coefficients[2] 
        theta_calib = np.rad2deg(theta_out_arith) - calibration_function
        print(f'Calibrated central theta (local) = {theta_calib[height//2, width//2] :.3f}°')
        print(fr'Calibration function = {coefficients[0]:.2f}x + {coefficients[1]:.2f}y + {coefficients[2]:.2f}')
        print(coefficients[0])
        print(coefficients[1])
        print(coefficients[2])
        print(r_squared)


    
    
    # np.save(f'/home/sfv514/Documents/Project/Camera/Python_JF/FFT of raw image/Calibrated theta arrays/Coffin calibration/{int(10*(angle_name-154.5))}.npy', theta_out_arith[row_min:row_max, col_min:col_max])  # Save as a .npy file
    # np.savez(f'/home/sfv514/Documents/Project/Camera/Python_JF/FFT of raw image/Calibrated theta arrays/Coffin calibration/xy{int(10*(angle_name-154.5))}.npz', x=np.arange(0, width, 1)[col_min:col_max], y=np.arange(0, height, 1)[row_min:row_max])  # Save as a .npy file



    # labels=['Mean value of row', 'Mean value of col.']
    # titles = ['Fit for mean row values', 'Fit for mean col. values']
    # # Create subplots
    # plt.figure(figsize=(10, 5))
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1)
    #     plot_gaussian(x_data[i], y_data[i], params[i], labels[i], titles[i])

    # plt.tight_layout()  # Adjust layout for better spacing
    # plt.show()
    
    if crop_raw_image == 'Yes':
        if image_hanning == 'Yes':
            plt.subplot(1,2,1)
            plot_image_with_crop(original_image, rows_rectangle=edge_of_circle_rows, cols_rectangle=edge_of_circle_cols)
            plt.title('Original Image')
            plt.subplot(1,2,2)
            plt.title('Cropped image with Hanning window applied')
            plt.imshow(raw_image, origin='lower')
        elif image_hanning == 'No':
            plt.subplot(1,2,1)
            plot_image_with_crop(original_image, rows_rectangle=edge_of_circle_rows, cols_rectangle=edge_of_circle_cols)
            plt.title('Image with cropping (no Hanning)')
            plt.subplot(1,2,2)
            plt.imshow(raw_image, origin='lower')
            plt.title('Cropped image with padding applied (if any)')
    
    elif crop_raw_image == 'No':
        plt.subplot(1,2,1)
        plot_image_with_circle_boundary(original_image, circle_center=circle_center, radius=R)
        plt.subplot(1,2,2)
        plot_image(raw_image)
        plt.title('Calculated boundaries of fringes shown')
    
    plt.show()
    
    if rotated_system == 'Yes':
        plot_fft_45(fft_image=fft_image, edge_coords=edge_coords)
        plt.show()

        plot_shifted_fft_with_windows_45(abs_fft_shifted=fftshift(abs_fft), width=width, height=height, edge_coords=edge_coords)
        plt.show()


    elif rotated_system == 'No':

        plot_fft(fft_image=fft_image, edge_coords=edge_coords)
        plt.show()
        plot_shifted_fft(to_keep=0.3, fft_image=fft_image, width=width, height=height)
        plt.show()
        
        plot_shifted_fft_with_windows(abs_fft_shifted=fftshift(abs_fft), width=width, height=height, edge_coords=edge_coords)
        plt.show()

    plt.show()

    # plotting_hilbert(original_function=ifft_pz, hilbert_amplitude=A_pz, function_name='+0', cols_for_lineout=cols_for_lineout, rows_for_lineout=rows_for_lineout)
    # plt.show()
    # plotting_hilbert(original_function=ifft_pp, hilbert_amplitude=A_pp, function_name='++', cols_for_lineout=cols_for_lineout, rows_for_lineout=rows_for_lineout)
    # plt.show()
    # plotting_hilbert(original_function=ifft_pm, hilbert_amplitude=A_pm, function_name='+-', cols_for_lineout=cols_for_lineout, rows_for_lineout=rows_for_lineout)
    # plt.show()
    
    # plt.subplot(2,2,1)
    # plt.title(f'IFFT: +0')
    # plt.imshow(ifft_pz, origin='lower')
    # plt.xlabel('x pixel')
    # plt.ylabel('y pixel')
    # plt.colorbar(label='Intensity')

    # plt.subplot(2,2,2)
    # plt.title(f'IFFT: ++')
    # plt.imshow(ifft_pp, origin='lower')
    # plt.xlabel('x pixel')
    # plt.ylabel('y pixel')
    # plt.colorbar(label='Intensity')

    # plt.subplot(2,2,3)
    # plt.title(f'IFFT: +-')
    # plt.imshow(ifft_pm, origin='lower')
    # plt.xlabel('x pixel')
    # plt.ylabel('y pixel')
    # plt.colorbar(label='Intensity')


    # plt.subplot(2,2,4)
    # plt.imshow(np.rad2deg(theta_out_arith), origin='lower')
    # plt.xlabel('x pixel')
    # plt.ylabel('y pixel')
    # plt.title(r'$\theta$ using $(|A(+,+)|+|A(+,-)|)/|A(+,0)|$')
    # plt.colorbar()
    # plt.show()
    
    contour_plot(theta_out_arith, circle_center=circle_center, R=R, crop_image=crop_raw_image, input_angle=angle_name-154.5)
    plt.show()
    # plot_calibrated_angle(theta=theta_calib, circle_center=circle_center, R=R, crop_image=crop_raw_image)
    # plt.show()
    
    
    plot_theta_central_row(width=width, 
                           angle_name=angle_name, thetas=thetas, theta_calibrated=theta_calib, radius_of_circle=R, circle_center=circle_center, cols_lim=cols_interp_lim)
    plt.show()
    plot_theta_cols(height=height, angle_name=angle_name, thetas=thetas, theta_calibrated=theta_calib, radius_of_circle=R, circle_center=circle_center, rows_lim=rows_interp_lim)
    #plt.xlim([400, 1600])
    # plt.ylim([18, 27])
    plt.show()
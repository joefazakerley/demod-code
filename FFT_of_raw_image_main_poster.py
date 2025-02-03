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
from scipy.ndimage import gaussian_filter, median_filter
from scipy import interpolate
import math

# Define file path
filepath = '/home/sfv514/Documents/Project/Camera/Python_JF/FFT of raw image/images/1645.raw'

# Define image dimensions
width = 1920
height = 1200

a = 1920/1200

# Read raw image data
with open(filepath, "rb") as image_file:
    raw_data = np.fromfile(image_file, dtype=np.uint8)
    
    # Ensure dimensions are correct
    if raw_data.size != width * height:
        raise ValueError(f"Size mismatch: Cannot reshape array of size {raw_data.size} into shape ({height}, {width})")
    
    # Reshape raw image data
    raw_image = raw_data.reshape((height, width))

raw_image = np.array(raw_image)

R = 575

h = 2*R/np.sqrt(a**2+1)
l = a*h

circ_c = [661, 986]

rows_rectangle = [int(circ_c[0] - h//2), int(circ_c[0] + h//2)]
cols_rectangle = [int(circ_c[1] - l//2), int(circ_c[1] + l//2)]

raw_image_cropped = raw_image[rows_rectangle[0]:rows_rectangle[1], cols_rectangle[0]:cols_rectangle[1]]

rectangle_1 = patches.Rectangle(
    (cols_rectangle[0], rows_rectangle[0]),  # Bottom-left corner
    cols_rectangle[1] - cols_rectangle[0],  # Width
    rows_rectangle[1] - rows_rectangle[0],  # Height
    edgecolor='red', facecolor='none', linewidth=1
)

plt.figure(figsize=(16,9))
plt.imshow(raw_image, origin='lower')
plt.gca().add_patch(rectangle_1)
plt.savefig('/home/sfv514/Documents/Project/Poster/Images/real_image_cropped.png')
plt.show()

crop_raw_image = 'Yes'

if crop_raw_image == 'Yes':
    raw_image = raw_image_cropped
    width = np.shape(raw_image)[1]
    height = np.shape(raw_image)[0]
    print(width)
    print(height)

# Define constants
n0 = 1.666
ne = 1.549
delta_n = n0 - ne
N_squared = (n0**2 - ne**2)/(n0**2 + ne**2)

L_s = 2e-3
L_d = 5.4e-3
L_0 = 4e-3

c = 299792458

# Lens focal length (m)
f = 50e-3

pixel_size = 5.86e-6

# Width and height of sensor in pixels
width_m = width*pixel_size
height_m = height*pixel_size

reduction_factor = 1

# Array of pixel numbers
x_pixel = np.arange(1, width+1, reduction_factor)
y_pixel = np.arange(1, height+1, reduction_factor)

# Pixels meshgrid
X_pixel, Y_pixel = np.meshgrid(x_pixel, y_pixel)
X_pixel = X_pixel.astype(np.float32)
Y_pixel = Y_pixel.astype(np.float32)

# Actual dimensions in m
x = np.linspace(-width_m/2, width_m/2 + pixel_size, width//reduction_factor)
y = np.linspace(-height_m/2, height_m/2 + pixel_size, height//reduction_factor)

# Create a 2D grid for x and y
X, Y = np.meshgrid(x, y)
X = X.astype(np.float32)
Y = Y.astype(np.float32)

lambda_sigma = 660e-9
omega_sigma = 2*np.pi*c/lambda_sigma

# Define constants as in Ford's paper
alpha = (L_s*N_squared)/(c*f*np.sqrt(2))
print('alpha:', alpha)
beta = (L_d*N_squared)/(c*f)
print('beta:', beta)
gamma = (delta_n/c)*(L_0 + 0.5*L_d)
print('gamma:', gamma)

# 2D fft of output intensity 
fft_image = fft2(raw_image)

# Absolute value of fft
abs_fft = np.abs(fft_image)

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

# Check the ifft of fft to return original image
ifft_image = ifft2(fft_image)

peaks = peak_local_max(abs_fft, min_distance=5, threshold_abs=.2e6, exclude_border=False)

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

print(f"Coords_PZ1: {coords_pz1}")
print(f"Coords_PZ2: {coords_pz2}")
print(f"Coords_PZ3: {coords_pz3}")
print(f"Coords_PZ4: {coords_pz4}")
print(f"Coords_PP1: {coords_pp1}")
print(f"Coords_PP2: {coords_pp2}")
print(f"Coords_PM1: {coords_pm1}")
print(f"Coords_PM2: {coords_pm2}")


def find_edge(edge_type, threshold, bright_spot_coords):
    """
    Find the edge coordinate in the FFT domain.

    Parameters:
        edge_type (str): The direction of the edge ('right', 'left', 'top', 'bottom').
        threshold (float): The threshold for determining the edge.
        bright_spot_coords (tuple): Coordinates of the bright spot (row, column).
        abs_fft (numpy.ndarray): 2D array representing the FFT magnitude.

    Returns:
        int: The coordinate of the detected edge.
    """
    if edge_type not in ['right', 'left', 'top', 'bottom']:
        raise ValueError("Invalid edge_type. Must be 'right', 'left', 'top', or 'bottom'.")

    edge_coord = bright_spot_coords[1] if edge_type in ['right', 'left'] else bright_spot_coords[0]
    
    while True:
        image = np.log(abs_fft + 1)
        current_value = image[bright_spot_coords[0], edge_coord] if edge_type in ['right', 'left'] else image[edge_coord, bright_spot_coords[1]]
        initial_value = image[bright_spot_coords[0], bright_spot_coords[1]]
        
        if current_value / initial_value < threshold:
            break

        if edge_type == 'right':
            edge_coord += 1
        elif edge_type == 'left':
            edge_coord -= 1
        elif edge_type == 'top':
            edge_coord += 1
        elif edge_type == 'bottom':
            edge_coord -= 1
    
    return edge_coord

threshold = 0.78
threshold_pz = 0.8

### Plus zero

## PZ1
# right edge
r_pz1 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pz1)

# left edge
l_pz1 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pz1)

# top edge
t_pz1_mean = (coords_pp1[0] + coords_pz1[0]) // 2
t_pz1_threshold = find_edge(edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz1)

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
r_pz2 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pz2)

# left edge
l_pz2 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pz2)

# top edge
t_pz2_mean = (coords_pm2[0] + coords_pz2[0]) // 2
t_pz2_threshold = find_edge(edge_type='top', threshold=threshold_pz, bright_spot_coords=coords_pz2)

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
r_pz3 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pz3)

# left edge
l_pz3 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pz3)

# top edge
t_pz3 = height

# bottom edge
b_pz3_mean = (coords_pm1[0] + coords_pz3[0]) // 2
b_pz3_threshold = find_edge(edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz3)

if b_pz3_threshold > b_pz3_mean:
    b_pz3 = b_pz3_threshold
    print('Bottom PZ3: Threshold')
else:
    b_pz3 = b_pz3_mean
    print('Bottom PZ3: Middle')

## PZ4
# right edge
r_pz4 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pz4)

# left edge
l_pz4 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pz4)

# top edge
t_pz4 = height

# bottom edge
b_pz4_mean = (coords_pp2[0] + coords_pz4[0]) // 2
b_pz4_threshold = find_edge(edge_type='bottom', threshold=threshold_pz, bright_spot_coords=coords_pz4)

if b_pz4_threshold > b_pz4_mean:
    b_pz4 = b_pz4_threshold
    print('Bottom PZ4: Threshold')
else:
    b_pz4 = b_pz4_mean
    print('Bottom PZ4: Middle')


### Plus plus

## PP1
# right edge
r_pp1 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pp1)

# left edge
l_pp1 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pp1)

# top edge
t_pp1 = find_edge(edge_type='top', threshold=threshold, bright_spot_coords=coords_pp1)

# bottom edge
b_pp1 = t_pz1

## PP2 
# right edge
r_pp2 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pp2)

# left edge
l_pp2 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pp2)

# top edge
t_pp2 = b_pz4

# bottom edge
b_pp2 = find_edge(edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pp2)


### Plus minus

## PM1
# right edge
r_pm1 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pm1)

# left edge
l_pm1 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pm1)

# top edge
t_pm1 = b_pz3

# bottom edge
b_pm1 = find_edge(edge_type='bottom', threshold=threshold, bright_spot_coords=coords_pm1)

## PM2 
# right edge
r_pm2 = find_edge(edge_type='right', threshold=threshold, bright_spot_coords=coords_pm2)

# left edge
l_pm2 = find_edge(edge_type='left', threshold=threshold, bright_spot_coords=coords_pm2)

# top edge
t_pm2 = find_edge(edge_type='top', threshold=threshold, bright_spot_coords=coords_pm2)

# bottom edge
b_pm2 = t_pz2

# Initialize empty masks with the same shape as the image
mask_pz1 = np.zeros((height, width))
mask_pz2 = np.zeros((height, width))
mask_pz3 = np.zeros((height, width))
mask_pz4 = np.zeros((height, width))
mask_pp1 = np.zeros((height, width))
mask_pp2 = np.zeros((height, width))
mask_pm1 = np.zeros((height, width))
mask_pm2 = np.zeros((height, width))

hanning = 'Yes'


if hanning == 'Yes':

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

if hanning =='No':

    # Applying Hanning window to each mask instead of using ones
    mask_pz1[b_pz1:t_pz1, l_pz1:r_pz1] = 1
    mask_pz2[b_pz2:t_pz2, l_pz2:r_pz2] = 1

    mask_pz3[b_pz3:t_pz3, l_pz3:r_pz3] = 1
    mask_pz4[b_pz4:t_pz4, l_pz4:r_pz4] = 1

    mask_pp1[b_pp1:t_pp1, l_pp1:r_pp1] = 1
    mask_pp2[b_pp2:t_pp2, l_pp2:r_pp2] = 1

    mask_pm1[b_pm1:t_pm1, l_pm1:r_pm1] = 1
    mask_pm2[b_pm2:t_pm2, l_pm2:r_pm2] = 1

# Combine masks for different regions
mask_pz = mask_pz1 + mask_pz2 + mask_pz3 + mask_pz4 # (+0)
mask_pp = mask_pp1 + mask_pp2 # (++)
mask_pm = mask_pm1 + mask_pm2 # (+-)

# Apply the Hanning-weighted masks to the FFT image
masked_fft_pz = mask_pz * fft_image # (+0)
masked_fft_pp = mask_pp * fft_image # (++)
masked_fft_pm = mask_pm * fft_image # (+-)

# Perform inverse Fourier transforms on the masked FFTs
ifft_pz = ifft2(masked_fft_pz) # (+0)
ifft_pp = ifft2(masked_fft_pp) # (++)
ifft_pm = ifft2(masked_fft_pm) # (+-)


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


def cosine_2d_linear(coords, A, B, C, kx, ky, phi, D):
    x_coord, y_coord = coords
    return (A*x_coord + B*y_coord + C) * np.cos(kx * x_coord + ky * y_coord + phi) + D

def compute_amplitude_2d_hilbert(function, function_name):
    """
    Compute the amplitude envelope of a 2D wave function using the Hilbert transform.

    Args:
        function (ndarray): Real-valued 2D array representing the wave function.

    Returns:
        ndarray: Amplitude envelope of the input wave function.
    """
    
    # Pre-smoothing to reduce high-frequency noise
    # function = gaussian_filter(function, sigma=1)

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
    # amplitude_tot = median_filter(amplitude_tot, size=5)

    return amplitude_tot


def cosine_1d(x_coord, y_constant, A, B, C, kx, ky, phi, D):
        return (A*x_coord + B*y_constant + C) * np.cos(kx * x_coord + ky * y_constant + phi) + D

def cosine_fitting_2d(function):

    coords = (X, Y)

    xdata = np.vstack((X.ravel(), Y.ravel()))

    zdata = np.real(function)
    zdata_rav = zdata.ravel()

    if np.array_equal(function, ifft_pz):
        initial_guess = [100, 100, 100, beta * omega_sigma, 0, gamma * omega_sigma, 0]
        function_name = '(+0)'

    elif np.array_equal(function, ifft_pp):
        initial_guess = [-100, 100, -100, beta * omega_sigma, alpha * omega_sigma, gamma * omega_sigma, 0]
        function_name = '(++)'

    elif np.array_equal(function, ifft_pm):
        initial_guess = [100, 100, 100, beta * omega_sigma, -alpha * omega_sigma, gamma * omega_sigma, 0]
        function_name = '(+-)'

    else:
        raise ValueError('Specify a valid function for function_to_fit')
    
    

    # Fit the model to the data
    params, covariance = curve_fit(cosine_2d_linear, xdata, zdata_rav, p0=initial_guess)

    # Output parameters
    A_fit, B_fit, C_fit, kx_fit, ky_fit, phi_fit, D_fit = params

    print(f"Fitted parameters: A={A_fit}, B={B_fit}, C={C_fit}, kx={kx_fit}, ky={ky_fit}, phi={phi_fit}, D={D_fit}")

    amplitude = A_fit*X + B_fit*Y + C_fit

    #params = np.delete(params, [1, 4])

    Z_fit = cosine_2d_linear(coords, *params)

    return amplitude, Z_fit, zdata, function_name, params, coords


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

def plotting_hilbert(original_function, hilbert_amplitude, function_name,  col_for_lineout, row_for_lineout):

    plt.close()
    plt.subplot(2,2,1)
    plt.title(f'Original function: {function_name}')
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
    plt.title(f'Lineout (rows: {row_for_lineout})')

    # Loop over rows for lineout
    for i, r in enumerate(row_for_lineout):  # enumerate gives index and value
        plt.plot(original_function[r, :], label=f'Original (row {r})')
        plt.plot(hilbert_amplitude[r, :], label=f'Hilbert (row {r})')
    plt.xlabel('x pixel')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title(f'Lineout (columns: {col_for_lineout})')

    # Loop over columns for lineout
    for i, c in enumerate(col_for_lineout):
        plt.plot(original_function[:, c], label=f'Original (col {c})')
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


# Define how much of the image you want to keep (will keep central part)
crop_fraction = 1

cols_for_lineout = [int(0.2*width), int(0.5*width), int(0.8*width)]
rows_for_lineout = [int(0.2*height), int(0.5*height), int(0.8*height)]

modulation = 'hilbert'

if modulation == 'curve_fit':

    A_pp, Z_fitpp, zdatapp, fn_pp, params_1dpp, coords = cosine_fitting_2d(ifft_pp, cosine_modulation='linear')
    A_pz, Z_fitpz, zdatA_pz, fn_pz, params_1dpz, coords = cosine_fitting_2d(ifft_pz, cosine_modulation='linear')
    A_pm, Z_fitpm, zdatapm, fn_pm, params_1dpm, coords = cosine_fitting_2d(ifft_pm, cosine_modulation='linear')
    
elif modulation == 'hilbert':

    A_pz = compute_amplitude_2d_hilbert(np.real(ifft_pz), '+0')
    A_pp = compute_amplitude_2d_hilbert(np.real(ifft_pp), '++')
    A_pm = compute_amplitude_2d_hilbert(np.real(ifft_pm), '+-')

elif modulation == 'find_peaks':
    A_pz = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pz)))
    A_pp = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pp)))
    A_pm = interpolate_zeros_2d(find_peaks_2d(np.real(ifft_pm)))
    
else:
    raise(ValueError('Define modulation!'))


def safe_divide(a, b):

    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


theta_out = np.pi/4 - 0.5*np.arctan(2*safe_divide(np.abs(A_pp), np.abs(A_pz)))

theta_out_00 = theta_out[height//2, width//2]

print(f'Theta out (0, 0): {np.rad2deg(theta_out_00):.3f}° ({np.rad2deg(np.min(theta_out)):.3f}° - {np.rad2deg(np.max(theta_out)):.3f}°)')
print(f'Std theta_1 central lineout: {np.std(theta_out[height//2, :])}')

theta_out_2 = np.pi/4 - 0.5*np.arctan(np.sqrt(4*safe_divide(np.abs(A_pp)*np.abs(A_pm), np.abs(A_pz)**2)))

theta_out_2_00 = theta_out_2[height//2, width//2]

print(f'Theta out_2 (0, 0): {np.rad2deg(theta_out_2_00):.3f}° ({np.rad2deg(np.min(theta_out_2)):.3f}° - {np.rad2deg(np.max(theta_out_2)):.3f}°)')
print(f'Std theta_2 central lineout: {np.std(theta_out_2[height//2, :])}')


def all_plots():
    rectangle_1 = patches.Rectangle(
    (cols_rectangle[0], rows_rectangle[0]),  # Bottom-left corner
    cols_rectangle[1] - cols_rectangle[0],  # Width
    rows_rectangle[1] - rows_rectangle[0],  # Height
    edgecolor='red', facecolor='none', linewidth=1
)

    #plt.subplot(2,1,1)
    plt.imshow(raw_image, origin='lower')
    plt.gca().add_patch(rectangle_1)

    plt.show()
    # Shifted FFT
    plt.close()
    to_keep = 0.3
    x_to_keep = [int((0.5-to_keep/2)*width), int((0.5+to_keep/2)*width)]
    y_to_keep = [int((0.5-to_keep/2)*height), int((0.5+to_keep/2)*height)]
    plt.figure(figsize=(12,12))
    plt.imshow(np.log(abs_fft_shifted + 1), extent=(np.min(freq_x), np.max(freq_x), np.min(freq_y), np.max(freq_y))) # [y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]] freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]
    # Add axis labels and title
    plt.xlabel('x freq. (cycles/pix.)', fontsize=16)
    plt.ylabel('y freq. (cycles/pix.)', fontsize=16)

    # Add colorbar with vertical label
    cbar = plt.colorbar(label=r'$log(\hat{I}+1)$', pad=0.02)
    cbar.ax.set_ylabel(r'$log(\hat{I}+1)$', rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=14) 

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # FFT with masking rectangles shown
    bottom_left_pz3 = [freq_y_grid_shifted[b_pz3 - height//2][0], freq_x_grid_shifted[0][l_pz3  + width//2]]
    bottom_left_pz4 = [freq_y_grid_shifted[b_pz4 - height//2][0], freq_x_grid_shifted[0][l_pz4- width//2]]
    bottom_left_pp1 = [freq_y_grid_shifted[b_pp1 + height//2][0], freq_x_grid_shifted[0][l_pp1 + width//2]]
    bottom_left_pp2 =  [freq_y_grid_shifted[b_pp2 - height//2][0], freq_x_grid_shifted[0][l_pp2 - width//2]]
    bottom_left_pm1 = [freq_y_grid_shifted[b_pm1 - height//2][0], freq_x_grid_shifted[0][l_pm1 + width//2]]
    bottom_left_pm2 = [freq_y_grid_shifted[b_pm2 + height//2][0], freq_x_grid_shifted[0][l_pm2 - width//2]]


    bottom_lefts = np.array([bottom_left_pz3, bottom_left_pz4, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])

    widths = ((np.max(freq_x) - np.min(freq_x))/width) * np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = ((np.max(freq_y) - np.min(freq_y))/height) * np.array([2*(t_pz1 - b_pz1), 2*(t_pz2 - b_pz2), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

    rectangles = []
    for i in range(len(widths)):
        rectangle = patches.Rectangle((bottom_lefts[i]),  widths[i],  heights[i],  edgecolor='red', facecolor='none', linewidth=1)
        rectangles.append(rectangle)
    

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

    plt.show()


    # FFT with masking rectangles shown
    bottom_left_pz1 = [b_pz1 - 0.5, l_pz1 - 0.5]
    bottom_left_pz2 = [b_pz2 - 0.5, l_pz2 - 0.5]
    bottom_left_pz3 = [b_pz3 - 0.5, l_pz3 - 0.5]
    bottom_left_pz4 = [b_pz4 - 0.5, l_pz4 - 0.5]
    bottom_left_pp1 = [b_pp1 - 0.5, l_pp1 - 0.5]
    bottom_left_pp2 =  [b_pp2 - 0.5, l_pp2 - 0.5]
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
    
    plt.title('FFT')
    plt.imshow(np.log(abs_fft+1), origin='lower', label='log(FFT+1)')
    plt.colorbar()

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

    #plt.show()
    
    # Masked FFTs

    plt.close()
    plt.subplot(2,2,1)
    plt.title('FFTs')
    plt.imshow(np.log(np.abs(masked_fft_pz)+1), origin='lower')
    plt.title('+0')

    plt.subplot(2,2,2)
    plt.imshow(np.log(np.abs(masked_fft_pp)+1), origin='lower')
    plt.title('++')

    plt.subplot(2,2,3)
    plt.imshow(np.log(np.abs(masked_fft_pm)+1), origin='lower')
    plt.title('+-')

    #plt.show()

    if modulation == 'curve_fit':

        plotting(coords, zdatA_pz, Z_fitpz, fn_pz, params_1dpz, x, 0)
        plt.show()
        plotting(coords, zdatapp, Z_fitpp, fn_pp, params_1dpp, x, 0)
        plt.show()
        plotting(coords, zdatapm, Z_fitpm, fn_pm, params_1dpm, x, 0)
        #plt.show()
        
    elif modulation == 'hilbert':

        plotting_hilbert(original_function=np.real(ifft_pz), hilbert_amplitude=A_pz, function_name='(+0)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show()

        plotting_hilbert(original_function=np.real(ifft_pp), hilbert_amplitude=A_pp, function_name='(++)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show()
        
        plotting_hilbert(original_function=np.real(ifft_pm), hilbert_amplitude=A_pm, function_name='(+-)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show()
    
    elif modulation == 'find_peaks':    
        
        plotting_interp(original_function=np.real(ifft_pz), peaks_interp=A_pz, function_name='(+0)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show()

        plotting_interp(original_function=np.real(ifft_pp), peaks_interp=A_pp, function_name='(++)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show()

        plotting_interp(original_function=np.real(ifft_pm), peaks_interp=A_pm, function_name='(+-)', col_for_lineout=cols_for_lineout, row_for_lineout=rows_for_lineout)
        #plt.show() 
    
    else:
        print('Define modulation')

    # Theta plots - total
    plt.close()
    plt.subplot(2,1,1)
    plt.imshow(np.rad2deg(theta_out), origin='lower')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $2|A(+,+)|/|A(+,0)|$: input 10 degs')
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(np.rad2deg(theta_out_2), origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $4|A(+,+)||A(+,-)|/|A(+,0)|^2$')
    plt.colorbar()

    #plt.show()
   
    # Theta plots (central bit)
    plt.close()
    plt.subplot(2,1,1)
    plt.imshow(np.rad2deg(theta_out[rows_for_lineout[0]:rows_for_lineout[-1], cols_for_lineout[0]:cols_for_lineout[-1]]), origin='lower')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $2|A(+,+)|/|A(+,0)|$ (central 60%)')
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(np.rad2deg(theta_out_2[rows_for_lineout[0]:rows_for_lineout[-1], cols_for_lineout[0]:cols_for_lineout[-1]]), origin='lower')
    plt.xlabel('x pixel')
    plt.ylabel('y pixel')
    plt.title(r'$\theta$ using $4|A(+,+)||A(+,-)|/|A(+,0)|^2$')
    plt.colorbar()

    #plt.show()

    # Theta plots - central row
    plt.close()
    plt.plot(np.rad2deg(theta_out[height//2, :]), label='theta_out_1')
    plt.plot(np.rad2deg(theta_out_2[height//2, :]), label='theta_out_2')
    plt.plot([0, width], [10, 10], label='Input')
    plt.xlabel('x pixel')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    #plt.show()


def poster_plots():
    
    # Read raw image data
    with open(filepath, "rb") as image_file:
        raw_data = np.fromfile(image_file, dtype=np.uint8)
        
        # Ensure dimensions are correct
        if raw_data.size != 1920 * 1200:
            raise ValueError(f"Size mismatch: Cannot reshape array of size {raw_data.size} into shape ({1200}, {1920})")
        
        # Reshape raw image data
        raw_image = raw_data.reshape((1200, 1920))

    raw_image = np.array(raw_image)

    R = 575

    h = 2*R/np.sqrt(a**2+1)
    l = a*h

    circ_c = [661, 986]

    rows_rectangle = [int(circ_c[0] - h//2), int(circ_c[0] + h//2)]
    cols_rectangle = [int(circ_c[1] - l//2), int(circ_c[1] + l//2)]

    rectangle_1 = patches.Rectangle(
        (cols_rectangle[0], rows_rectangle[0]),  # Bottom-left corner
        cols_rectangle[1] - cols_rectangle[0],  # Width
        rows_rectangle[1] - rows_rectangle[0],  # Height
        edgecolor='red', facecolor='none', linewidth=1
    )

    plt.figure(figsize=(16,9))

    plt.subplot(2,2,1)
    plt.imshow(raw_image, origin='lower')
    plt.gca().add_patch(rectangle_1)
    plt.xlabel('x pixel', fontsize=16)
    plt.ylabel(' y pixel', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add colorbar with vertical label
    cbar = plt.colorbar(label='Intensity as pixel value (8-bit)', pad=0.02)
    cbar.ax.set_ylabel(ylabel='Intensity as pixel value (8-bit)', rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=14) 

    plt.subplot(2,2,2)

    plt.imshow(np.log(abs_fft_shifted + 1), extent=(np.min(freq_x), np.max(freq_x), np.min(freq_y), np.max(freq_y))) # [y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]] freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]
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
    bottom_left_pz3 = [freq_y_grid_shifted[b_pz3 - height//2][0], freq_x_grid_shifted[0][l_pz3  + width//2]]
    bottom_left_pz4 = [freq_y_grid_shifted[b_pz4 - height//2][0], freq_x_grid_shifted[0][l_pz4- width//2]]
    bottom_left_pp1 = [freq_y_grid_shifted[b_pp1 + height//2][0], freq_x_grid_shifted[0][l_pp1 + width//2]]
    bottom_left_pp2 =  [freq_y_grid_shifted[b_pp2 - height//2][0], freq_x_grid_shifted[0][l_pp2 - width//2]]
    bottom_left_pm1 = [freq_y_grid_shifted[b_pm1 - height//2][0], freq_x_grid_shifted[0][l_pm1 + width//2]]
    bottom_left_pm2 = [freq_y_grid_shifted[b_pm2 + height//2][0], freq_x_grid_shifted[0][l_pm2 - width//2]]


    bottom_lefts = np.array([bottom_left_pz3, bottom_left_pz4, bottom_left_pp1, bottom_left_pp2, bottom_left_pm1, bottom_left_pm2])
    bottom_lefts = np.array([bl[::-1] for bl in bottom_lefts])

    widths = ((np.max(freq_x) - np.min(freq_x))/width) * np.array([(r_pz1 - l_pz1), (r_pz2 - l_pz2), (r_pp1 - l_pp1), (r_pp2 - l_pp2), (r_pm1 - l_pm1), (r_pm2 - l_pm2)])
    heights = ((np.max(freq_y) - np.min(freq_y))/height) * np.array([2*(t_pz1 - b_pz1), 2*(t_pz2 - b_pz2), (t_pp1 - b_pp1), (t_pp2 - b_pp2), (t_pm1 - b_pm1), (t_pm2 - b_pm2)])

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

    plt.subplot(2,2,3)

    plt.imshow(np.rad2deg(theta_out_2), extent=(cols_rectangle[0], cols_rectangle[1], rows_rectangle[0], rows_rectangle[1]), origin='lower')

    # Add axis labels and title
    plt.xlabel('x pixel', fontsize=16)
    plt.ylabel('y pixel', fontsize=16)

    # Add colorbar with vertical label
    cbar = plt.colorbar(label=r'$\theta$°', pad=0.02)
    cbar.ax.set_ylabel(r'$\theta$°', rotation=90, labelpad=15, fontsize=16)
    cbar.ax.tick_params(labelsize=14) 

    # Customize ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()  # Adjust layout to prevent overlap

    plt.subplot(2,2,4)
    plt.plot(np.arange(cols_rectangle[0], cols_rectangle[1], 1), np.rad2deg(theta_out_2[height//2, :]), label=r'Calculated $\theta$ (central row)')
    plt.plot([cols_rectangle[0], cols_rectangle[1]], [10, 10], label=r'Input $\theta$')
    plt.xlabel('x pixel', fontsize=16)
    plt.ylabel(r'$\theta$°', fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=14)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig('/home/sfv514/Documents/Project/Poster/Images/real_all.png')
    plt.show()

poster_plots()
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

from FFT_of_raw_image_main import *

## RUN CODE ##



def run_fft(angle_name, method, hanning, crop_raw_image, threshold, threshold_pz, R, circle_center, gaussian_pre_smoothing, median_post_smoothing):

    # Define image dimensions
    width = 1920
    height = 1200

    a = 1920/1200

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

    if angle_name > 999 and angle_name < 10000:
        angle_name = angle_name/10

    # Define file path
    filepath = f'/home/sfv514/Documents/Project/Camera/Python_JF/FFT of raw image/images/{angle_name}.raw'

    raw_image = read_image(filepath=filepath, width=width, height=height)

    if crop_raw_image == 'Yes':

        rows_rectangle, cols_rectangle, raw_image_cropped = crop_circle(R=R, circle_center=circle_center, a=a, raw_image=raw_image)

        rectangle = patches.Rectangle(
        (cols_rectangle[0], rows_rectangle[0]),  # Bottom-left corner
        cols_rectangle[1] - cols_rectangle[0],  # Width
        rows_rectangle[1] - rows_rectangle[0],  # Height
        edgecolor='red', facecolor='none', linewidth=1
        )
        raw_image = raw_image_cropped
        width = np.shape(raw_image)[1]
        height = np.shape(raw_image)[0]

    fft_image, abs_fft, *_ = fft_2d(raw_image, width, height)


    all_bright_spot_coords = find_peaks_fft(abs_fft, width, height)

    print(f"Coords_PZ1: {all_bright_spot_coords[0]}")
    print(f"Coords_PZ2: {all_bright_spot_coords[1]}")
    print(f"Coords_PZ3: {all_bright_spot_coords[2]}")
    print(f"Coords_PZ4: {all_bright_spot_coords[3]}")
    print(f"Coords_PP1: {all_bright_spot_coords[4]}")
    print(f"Coords_PP2: {all_bright_spot_coords[5]}")
    print(f"Coords_PM1: {all_bright_spot_coords[6]}")
    print(f"Coords_PM2: {all_bright_spot_coords[7]}")


    edge_coords = calculate_edges(abs_fft, threshold=threshold, threshold_pz=threshold_pz, all_bright_spot_coords=all_bright_spot_coords, height=height)

    mask_pz, mask_pp, mask_pm = define_masks(hanning, edge_coords, height=height, width=width)

    masked_fft_pz, masked_fft_pp, masked_fft_pm = apply_masks(mask_pz, mask_pp, mask_pm, fft_image)

    # Perform inverse Fourier transforms on the masked FFTs
    ifft_pz = np.real(ifft2(masked_fft_pz)) # (+0)
    ifft_pp = np.real(ifft2(masked_fft_pp)) # (++)
    ifft_pm = np.real(ifft2(masked_fft_pm)) # (+-)

    iffts = [ifft_pz, ifft_pp, ifft_pm]

    A_pz, A_pp, A_pm = find_amplitudes(ifft_pz, ifft_pp, ifft_pm, method, gaussian_pre_smoothing=gaussian_pre_smoothing, median_post_smoothing=median_post_smoothing)

    amplitudes = [A_pz, A_pp, A_pm]

    theta_out_1_pp, theta_out_1_pm, theta_out_2 = calculate_theta(A_pz, A_pp, A_pm, height=height, width=width)

    thetas = [theta_out_1_pp, theta_out_1_pm, theta_out_2]

    cols_for_lineout = [int(0.2*width), int(0.5*width), int(0.8*width)]
    rows_for_lineout = [int(0.2*height), int(0.5*height), int(0.8*height)]

    return thetas, cols_for_lineout, rows_for_lineout, width, height, cols_rectangle, raw_image, fft_image, edge_coords, iffts, amplitudes

def plot_single(crop_raw_image, height, cols_rectangle, edge_circ_cols, angle_name, plot_label, thetas):

    theta_out_1_pp = thetas[0]
    theta_out_1_pm = thetas[1]
    theta_out_2 = thetas[2]
    
    if crop_raw_image == 'Yes':

        x_crop = np.arange(cols_rectangle[0], cols_rectangle[1], 1)

        plt.plot(x_crop, np.rad2deg(theta_out_1_pp[height//2, :]), label=fr'$\theta_1pp$ cropped ({plot_label})')
        plt.plot(x_crop, np.rad2deg(theta_out_1_pm[height//2, :]), label=fr'$\theta_1pm$ cropped ({plot_label})')
        plt.plot(x_crop, np.rad2deg(theta_out_2[height//2, :]), label=fr'$\theta_2$ cropped ({plot_label})')
        plt.plot([x_crop[0], x_crop[-1]], [angle_name-154.5, angle_name-154.5], label='Input')
        plt.xlabel('x pixel')
        plt.ylabel('Angle (degrees)')
        plt.legend()

    elif crop_raw_image == 'No':

        x_crop = np.arange(edge_circ_cols[0], edge_circ_cols[1])

        plt.plot(x_crop, np.rad2deg(theta_out_1_pp[height//2, edge_circ_cols[0]:edge_circ_cols[1]]), label=fr'$\theta_1pp$ ({plot_label})')
        plt.plot(x_crop, np.rad2deg(theta_out_1_pm[height//2, edge_circ_cols[0]:edge_circ_cols[1]]), label=fr'$\theta_1pm$ ({plot_label})')
        plt.plot(x_crop, np.rad2deg(theta_out_2[height//2, edge_circ_cols[0]:edge_circ_cols[1]]), label=fr'$\theta_2$ ({plot_label})')
        plt.plot([edge_circ_cols[0], edge_circ_cols[1]], [angle_name-154.5, angle_name-154.5], label='Input')
        plt.xlabel('x pixel')
        plt.ylabel('Angle (degrees)')
        plt.legend()

def fft_plot(width, height, fft_image):

    abs_fft_shifted = np.abs(fftshift(fft_image))

    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    # Shifted FFT
    plt.close()
    to_keep = 0.3
    x_to_keep = [int((0.5-to_keep/2)*width), int((0.5+to_keep/2)*width)]
    y_to_keep = [int((0.5-to_keep/2)*height), int((0.5+to_keep/2)*height)]
    plt.figure(figsize=(12,12))
    plt.imshow(np.log(abs_fft_shifted + 1)[y_to_keep[0]:y_to_keep[1], x_to_keep[0]:x_to_keep[1]], extent=(freq_x[x_to_keep[1]], freq_x[x_to_keep[0]], freq_y[y_to_keep[1]], freq_y[y_to_keep[0]]))

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

    plt.show()

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
    
    plt.close()
    plt.title('FFT')
    plt.imshow(np.log(abs_fft+1), origin='lower', label='log(FFT+1)')
    plt.colorbar()

    for i in range(len(rectangles)):
        plt.gca().add_patch(rectangles[i])

    plt.show()


R = 575
circ_c = [661, 986]

edge_circ_cols = [circ_c[1] - R, circ_c[1] + R]
edge_circ_rows = [circ_c[0] - R, circ_c[0] + R]

angle_name = 160
crop_raw_image = 'Yes'
method = 'hilbert'
hanning = 'Yes'
gaussian_pre_smoothing = 'No'
median_post_smoothing = 'No'
threshold = 0.05
threshold_pz = 0.06

plot_label = ''

thetas, cols_for_lineout, rows_for_lineout, width, height, cols_rectangle, raw_image, fft_image, edge_coords, iffts, amplitudes = run_fft(angle_name=angle_name, method=method, hanning=hanning, crop_raw_image=crop_raw_image, threshold=threshold, 
                                                                                          threshold_pz=threshold_pz, R=R, circle_center=circ_c, gaussian_pre_smoothing=gaussian_pre_smoothing, median_post_smoothing=median_post_smoothing)

theta_1_pp = thetas[0]
theta_1_pm = thetas[1]
theta_2 = thetas[2]

plot_single(crop_raw_image=crop_raw_image, height=height, cols_rectangle=cols_rectangle, edge_circ_cols=edge_circ_cols, angle_name=angle_name, plot_label=plot_label, thetas=thetas)


angle_name = 160
crop_raw_image = 'Yes'
method = 'hilbert'
hanning = 'Yes'
gaussian_pre_smoothing = 'No'
median_post_smoothing = 'No'
threshold = 0.05
threshold_pz = 0.02

plot_label = 'big pz window'

thetas, cols_for_lineout, rows_for_lineout, width, height, cols_rectangle, raw_image, fft_image, edge_coords, iffts, amplitudes = run_fft(angle_name=angle_name, method=method, hanning=hanning, crop_raw_image=crop_raw_image, threshold=threshold, 
                                                                                          threshold_pz=threshold_pz, R=R, circle_center=circ_c, gaussian_pre_smoothing=gaussian_pre_smoothing, median_post_smoothing=median_post_smoothing)

theta_1_pp = thetas[0]
theta_1_pm = thetas[1]
theta_2 = thetas[2]

plot_single(crop_raw_image=crop_raw_image, height=height, cols_rectangle=cols_rectangle, edge_circ_cols=edge_circ_cols, angle_name=angle_name, plot_label=plot_label, thetas=thetas)


plt.show()
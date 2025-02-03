import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft2, fftshift, ifft2
from scipy.signal import windows

def fft_2d_raw(raw_image, fringe_orientation, mask):

    # Perform the 2D Fourier Transform
    fft_image = fft2(raw_image)

    # Absolute value of fft
    abs_fft_image = np.abs(fft_image)

    plt.imshow(np.log(abs_fft_image + 1), origin='lower')
    # plt.show()

    # Shift the zero frequency component to the center of the spectrum
    fft_image_shifted = fftshift(fft_image)

    # Calculate the magnitude spectrum (already computed with abs_fft_image)
    abs_fft_image_shifted = np.abs(fft_image_shifted)

    # Define the frequency axes
    freq_x = fftfreq(width, 1)   # Frequency range in the x-axis
    freq_y = fftfreq(height, 1)  # Frequency range in the y-axis

    # Create the frequency grids
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Shift the grids to match the FFT shift
    freq_x_grid_shifted = fftshift(freq_x_grid)
    freq_y_grid_shifted = fftshift(freq_y_grid)

    # Define masks based on fringe orientation
    if fringe_orientation == 'horizontal':
        mask = (mask_range[0] < freq_y_grid_shifted) & (freq_y_grid_shifted < mask_range[1])
    elif fringe_orientation == 'vertical':
        mask = (mask_range[0] < freq_x_grid_shifted) & (freq_x_grid_shifted < mask_range[1])
    else:
        raise ValueError('Invalid fringe orientation. Use "horizontal" or "vertical".')
    
    # Apply the mask to the magnitude spectrum
    masked_fft = np.where(mask, abs_fft_image_shifted, 0)

    # plt.imshow(np.log(masked_fft+1))
    # plt.show()

    # Find the index of the maximum value in the masked magnitude spectrum
    max_location = np.unravel_index(np.argmax(masked_fft), masked_fft.shape)

    # Retrieve the corresponding frequency values
    freq_max_x = fftshift(freq_x)[max_location[1]]
    freq_max_y = fftshift(freq_y)[max_location[0]]

    print('Max location (index):', np.array(max_location))
    print(f'log(max value): {np.log(np.max(masked_fft)):.2f}')
    print(f'Frequency (cycles/pixel) at max fft (x, y): ({freq_max_x:.5f}, {freq_max_y:.5f})')

    plt.figure(figsize=(12, 6))
    # Plot the original image

    plt.subplot(1, 2, 1)
    img_plot = plt.imshow(raw_image, origin='lower')
    cbar = plt.colorbar(img_plot, fraction=0.03, pad=0.04)
    plt.title('Original Image')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.gca().set_aspect(1)  # Maintain aspect ratio based on pixel dimensions

    # Plot the magnitude spectrum with frequency labels
    plt.subplot(1, 2, 2)
    fft_plot = plt.imshow(np.log(abs_fft_image_shifted + 1), extent=(freq_x_grid_shifted.min(), freq_x_grid_shifted.max(), freq_y_grid_shifted.min(), freq_y_grid_shifted.max()))
    cbar = plt.colorbar(fft_plot, fraction=0.03, pad=0.04)
    cbar.set_label('log(Magnitude of FFT)')
    plt.title('2D FFT Magnitude Spectrum')
    plt.xlabel('Frequency X (cycles/pixel)')
    plt.ylabel('Frequency Y (cycles/pixel)')
    #plt.xlim([-0.03, 0.03])
    #plt.ylim([-0.05, 0.05])
    plt.gca().set_aspect(height/width)  # Maintain aspect ratio based on frequency dimensions

    plt.show()

    plt.close()

    return freq_max_x, freq_max_y

def masked_fft(fft_image):
    # Define where regions of interst are (bright spots of FFT) in pixels

    coords_pluszero = np.array([47, 4])
    coords_plusplus = np.array([47, 46])
    coords_plusminus = np.array([1153, 36])

    coords_nq_pluszero = np.array([height, width]) - coords_pluszero 
    coords_nq_plusplus = np.array([height, width]) - coords_plusplus
    coords_nq_plusminus = np.array([height, width]) - coords_plusminus

    # Initialize masks
    masks = [np.zeros((height, width)) for _ in range(6)]

    # Applying Hanning window to each mask instead of using ones
    # Use slicing to create a window around each (x, y) position and apply the Hanning window

    # For +0 frequency
    row_delta = int(np.mean([0, coords_pluszero[0]]))  # Mid point between DC component and +0 component
    col_delta = (coords_plusplus[1] - coords_pluszero[1]) // 2  # Half the distance between frequencies

    hann_window_1d_row = np.hanning(2*row_delta)  # 1D Hanning window for rows
    hann_window_1d_col = np.hanning(2*col_delta)   # 1D Hanning window for columns
    hann_window_2d = np.outer(hann_window_1d_row, hann_window_1d_col)  # Create a 2D Hanning window


    for i, coords in enumerate([coords_pluszero, coords_plusplus, coords_plusminus]):

        row_start = coords[0] - row_delta
        row_start_crop = max(0, row_start)
        row_end = coords[0] + row_delta
        row_end_crop = min(height, row_end)

        col_start = coords[1] - col_delta
        col_start_crop = max(0, col_start)
        col_end = coords[1] + col_delta
        col_end_crop = min(width, col_end)

        if i == 0:
            masks[2*i][row_start_crop:row_end_crop, col_start_crop:col_end_crop] = hann_window_2d[:, col_delta - coords_pluszero[1]:]
            
        else:
            masks[2*i][row_start_crop:row_end_crop, col_start_crop:col_end_crop] = hann_window_2d
            


    for j, coords in enumerate([coords_nq_pluszero, coords_nq_plusplus, coords_nq_plusminus]):

        # Define row and column indices for slicing
        row_start_nq = coords[0] - row_delta
        row_start_nq_crop = max(0, row_start_nq)
        row_end_nq = coords[0] + row_delta
        row_end_nq_crop = min(height, row_end_nq)

        col_start_nq = coords[1] - col_delta
        col_start_nq_crop = max(0, col_start_nq)
        col_end_nq = coords[1] + col_delta
        col_end_nq_crop = min(width, col_end_nq)

        # Assign Hanning window to the appropriate mask
        if j == 0:
            masks[2*j + 1][row_start_nq_crop:row_end_nq_crop, col_start_nq_crop:col_end_nq_crop] = hann_window_2d[:row_end_nq_crop - row_start_nq_crop, :col_end_nq_crop - col_start_nq_crop]
        
        else:
            masks[2*j + 1][row_start_nq_crop:row_end_nq_crop, col_start_nq_crop:col_end_nq_crop] = hann_window_2d




    mask_12 = masks[0] + masks[1]
    mask_34 = masks[2] + masks[3]
    mask_56 = masks[4] + masks[5]



    # # For nq +0 frequency (on the opposite side of the image)
    # row_start_nq = coords[0] - row_delta
    # print(row_start_nq)
    # row_start_nq_crop = max(0, row_start_nq)
    # row_end_nq = coords_nq_pluszero[0] + row_delta
    # print(row_end_nq)
    # row_end_nq_crop = min(height, row_end_nq)

    # col_start_nq = coords_nq_pluszero[1] - col_delta
    # print(col_start_nq)
    # col_start_nq_crop = max(0, col_start_nq)
    # col_end_nq = coords_nq_pluszero[1] + col_delta
    # print(col_end_nq)
    # col_end_nq_crop = min(width, col_end_nq)

    # Ensure proper slicing for mask_2
    # mask_2[row_start_nq_crop:row_end_nq_crop, col_start_nq_crop:col_end_nq_crop] = hann_window_2d[:, :width-col_start_nq]


    # Apply the Hanning-weighted masks to the FFT image
    masked_fft_12 = mask_12 * fft_image # (+0)
    masked_fft_34 = mask_34 * fft_image # (++)
    masked_fft_56 = mask_56 * fft_image # (+-)



    # Perform inverse Fourier transforms on the masked FFTs
    ifft_12 = ifft2(masked_fft_12) # (+0)
    ifft_34 = ifft2(masked_fft_34) # (++)
    ifft_56 = ifft2(masked_fft_56) # (+-)

    plt.close()
    plt.subplot(2,2,1)
    plt.imshow(np.log(abs(fft_image)+1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(np.log(abs(masked_fft_12) + 1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(np.real(ifft_12), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(np.imag(ifft_12), origin='lower')
    plt.colorbar()

    plt.show()

    plt.subplot(2,2,1)
    plt.imshow(np.log(abs(fft_image)+1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(np.log(abs(masked_fft_34) + 1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(np.real(ifft_34), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(np.imag(ifft_34), origin='lower')
    plt.colorbar()

    plt.show()


    plt.subplot(2,2,1)
    plt.imshow(np.log(abs(fft_image)+1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(np.log(abs(masked_fft_56) + 1), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(np.real(ifft_56), origin='lower')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(np.imag(ifft_56), origin='lower')
    plt.colorbar()

    plt.show()

    return ifft_12, ifft_34, ifft_56






if __name__ == '__main__':

    #
    # >>> RAW IMAGE ANALYSIS <<<
    #

    # Define file path
    filepath = '/home/sfv514/Documents/Project/Camera/Full optical system/Horizontal fringes/fringes 1.raw'

    # Define image dimensions
    width = 1920
    height = 1200

    # Define desired window size
    window_width = width//2
    window_height = height//2

    # Read raw image data
    with open(filepath, "rb") as image_file:
        raw_data = np.fromfile(image_file, dtype=np.uint8)
        
        # Ensure dimensions are correct
        if raw_data.size != width * height:
            raise ValueError(f"Size mismatch: Cannot reshape array of size {raw_data.size} into shape ({height}, {width})")
        
        # Reshape raw image data
        raw_image = raw_data.reshape((height, width))

    # Normalise
    raw_image_norm = raw_image/np.max(raw_image)

    # Some pixel stats
    mean_pixel_value = np.mean(raw_data)
    print("Mean pixel value:", mean_pixel_value)

    min_pixel_value = np.min(raw_image)
    print("Min pixel value:", min_pixel_value)

    max_pixel_value = np.max(raw_image)
    print("Max pixel value:", max_pixel_value)

    # Bright column
    column_bright = raw_image[:, 560*2]

    # Normalise
    column_bright_normalised = column_bright/np.max(column_bright)

    # 'vertical' or 'horizontal'
    fringe_orientation = 'horizontal'

    input_wavelength = 653
    input_polarisation = 25

    n_o = 1.666
    n_e = 1.549

    sav_width, sav_angle = 7.6, 45
    disp_width, disp_cut_angle, disp_angle = 5.4, 45, 0 + 90

    # Delay plate width in mm
    waveplate_width = 1.2

    # calculate no. waves delay
    waves_delay = (waveplate_width/1000)*(n_o - n_e)/(input_wavelength*1e-9)
    print('waves delay: ', waves_delay)
    
    waveplate_angle = 0 + 90
    lp_angle = 45

    # pycrisp_image = imse_amplitude(sav_width, disp_width, disp_cut_angle, disp_angle, waves_delay, waveplate_angle, lp_angle, input_wavelength, input_polarisation)  

    # fft_2d_pycrisp(pycrisp_image)

    
    mask_range = [0.03, 0.1]

    image = raw_image_norm
    freq_max_x, freq_max_y = fft_2d_raw(raw_image, fringe_orientation, mask_range)

    # angle from vertical
    theta = np.arctan(freq_max_x/freq_max_y)
    print(np.rad2deg(theta))

    ifft_12, ifft_34, ifft_56 = masked_fft(fft2(raw_image))

    ifft_12 = np.real(ifft_12)
    ifft_34 = np.real(ifft_34)
    ifft_56 = np.real(ifft_56)

    ratio = np.max(ifft_34)/np.max(ifft_12)

    ratio_all = ifft_34/ifft_12

    theta_all = 0.5*np.arctan(2*ratio_all)
    plt.imshow(theta_all, origin='lower')
    plt.show()

    theta_out = 0.5*np.arctan(2*ratio)

    print('theta_out:', np.rad2deg(theta_out))
# image_capture -> raspberry_ndvi -> latlong_pixel -> calculate_pixel -> assign_latlong_to_pixel -> latlong_to_country

from orbit import ISS # Only works on raspberry pi
from picamera import PiCamera
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import cv2
import numpy as np
import datetime
from numpy import uint8
from exif import Image
np.set_printoptions(threshold=np.inf, linewidth=200)

colour_map = np.array([[[255, 255, 255]],

       [[250, 250, 250]],

       [[245, 245, 245]],

       [[241, 241, 241]],

       [[237, 237, 237]],

       [[232, 232, 232]],

       [[228, 228, 228]],

       [[225, 225, 225]],

       [[221, 221, 221]],

       [[216, 216, 216]],

       [[212, 212, 212]],

       [[208, 208, 208]],

       [[204, 204, 204]],

       [[199, 199, 199]],

       [[194, 194, 194]],

       [[190, 190, 190]],

       [[186, 186, 186]],

       [[182, 182, 182]],

       [[177, 177, 177]],

       [[174, 174, 174]],

       [[170, 170, 170]],

       [[166, 166, 166]],

       [[161, 161, 161]],

       [[157, 157, 157]],

       [[153, 153, 153]],

       [[148, 148, 148]],

       [[144, 144, 144]],

       [[139, 139, 139]],

       [[135, 135, 135]],

       [[131, 131, 131]],

       [[128, 128, 128]],

       [[123, 123, 123]],

       [[119, 119, 119]],

       [[115, 115, 115]],

       [[111, 111, 111]],

       [[106, 106, 106]],

       [[102, 102, 102]],

       [[ 97,  97,  97]],

       [[ 93,  93,  93]],

       [[ 89,  89,  89]],

       [[ 84,  84,  84]],

       [[ 80,  80,  80]],

       [[ 77,  77,  77]],

       [[ 73,  73,  73]],

       [[ 68,  68,  68]],

       [[ 64,  64,  64]],

       [[ 60,  60,  60]],

       [[ 55,  55,  55]],

       [[ 51,  51,  51]],

       [[ 55,  55,  55]],

       [[ 60,  60,  60]],

       [[ 64,  64,  64]],

       [[ 68,  68,  68]],

       [[ 73,  73,  73]],

       [[ 77,  77,  77]],

       [[ 80,  80,  80]],

       [[ 84,  84,  84]],

       [[ 89,  89,  89]],

       [[ 93,  93,  93]],

       [[ 97,  97,  97]],

       [[102, 102, 102]],

       [[106, 106, 106]],

       [[111, 111, 111]],

       [[115, 115, 115]],

       [[119, 119, 119]],

       [[123, 123, 123]],

       [[128, 128, 128]],

       [[131, 131, 131]],

       [[135, 135, 135]],

       [[139, 139, 139]],

       [[144, 144, 144]],

       [[148, 148, 148]],

       [[153, 153, 153]],

       [[157, 157, 157]],

       [[161, 161, 161]],

       [[166, 166, 166]],

       [[170, 170, 170]],

       [[174, 174, 174]],

       [[177, 177, 177]],

       [[182, 182, 182]],

       [[186, 186, 186]],

       [[190, 190, 190]],

       [[194, 194, 194]],

       [[199, 199, 199]],

       [[204, 204, 204]],

       [[208, 208, 208]],

       [[212, 212, 212]],

       [[216, 216, 216]],

       [[221, 221, 221]],

       [[225, 225, 225]],

       [[228, 228, 228]],

       [[232, 232, 232]],

       [[237, 237, 237]],

       [[241, 241, 241]],

       [[245, 245, 245]],

       [[250, 250, 250]],

       [[255, 255, 255]],

       [[250, 250, 250]],

       [[245, 245, 245]],

       [[240, 240, 240]],

       [[235, 235, 235]],

       [[230, 230, 230]],

       [[225, 225, 225]],

       [[219, 219, 219]],

       [[214, 214, 214]],

       [[209, 209, 209]],

       [[204, 204, 204]],

       [[199, 199, 199]],

       [[194, 194, 194]],

       [[190, 190, 190]],

       [[185, 185, 185]],

       [[180, 180, 180]],

       [[175, 175, 175]],

       [[170, 170, 170]],

       [[165, 165, 165]],

       [[160, 160, 160]],

       [[154, 154, 154]],

       [[151, 151, 151]],

       [[145, 145, 145]],

       [[140, 140, 140]],

       [[135, 135, 135]],

       [[130, 130, 130]],

       [[125, 125, 125]],

       [[120, 120, 120]],

       [[115, 115, 115]],

       [[111, 111, 111]],

       [[106, 106, 106]],

       [[101, 101, 101]],

       [[ 96,  96,  96]],

       [[ 91,  91,  91]],

       [[ 86,  86,  86]],

       [[ 80,  80,  80]],

       [[ 75,  75,  75]],

       [[ 70,  70,  70]],

       [[ 65,  65,  65]],

       [[ 60,  60,  60]],

       [[ 55,  55,  55]],

       [[ 79,  65,  65]],

       [[105,  77,  77]],

       [[129,  87,  87]],

       [[154,  97,  97]],

       [[180, 107, 107]],

       [[204, 119, 119]],

       [[230, 129, 129]],

       [[255, 139, 139]],

       [[239, 147, 130]],

       [[222, 153, 121]],

       [[207, 161, 112]],

       [[190, 167, 105]],

       [[175, 175,  96]],

       [[158, 182,  87]],

       [[143, 190,  78]],

       [[126, 196,  69]],

       [[111, 204,  60]],

       [[ 94, 211,  51]],

       [[ 78, 218,  42]],

       [[ 63, 226,  35]],

       [[ 46, 232,  26]],

       [[ 31, 240,  17]],

       [[ 14, 246,   8]],

       [[  0, 255,   0]],

       [[  0, 255,   7]],

       [[  0, 255,  14]],

       [[  0, 255,  23]],

       [[  0, 255,  31]],

       [[  0, 255,  38]],

       [[  0, 255,  46]],

       [[  0, 255,  55]],

       [[  0, 255,  63]],

       [[  0, 255,  70]],

       [[  0, 255,  78]],

       [[  0, 255,  87]],

       [[  0, 255,  94]],

       [[  0, 255, 102]],

       [[  0, 255, 111]],

       [[  0, 255, 119]],

       [[  0, 255, 126]],

       [[  0, 255, 134]],

       [[  0, 255, 143]],

       [[  0, 255, 151]],

       [[  0, 255, 158]],

       [[  0, 255, 166]],

       [[  0, 255, 175]],

       [[  0, 255, 182]],

       [[  0, 255, 190]],

       [[  0, 255, 199]],

       [[  0, 255, 207]],

       [[  0, 255, 214]],

       [[  0, 255, 222]],

       [[  0, 255, 231]],

       [[  0, 255, 239]],

       [[  0, 255, 246]],

       [[  0, 255, 255]],

       [[  0, 249, 255]],

       [[  0, 244, 255]],

       [[  0, 239, 255]],

       [[  0, 232, 255]],

       [[  0, 227, 255]],

       [[  0, 222, 255]],

       [[  0, 217, 255]],

       [[  0, 212, 255]],

       [[  0, 207, 255]],

       [[  0, 200, 255]],

       [[  0, 195, 255]],

       [[  0, 190, 255]],

       [[  0, 185, 255]],

       [[  0, 180, 255]],

       [[  0, 175, 255]],

       [[  0, 170, 255]],

       [[  0, 163, 255]],

       [[  0, 158, 255]],

       [[  0, 153, 255]],

       [[  0, 148, 255]],

       [[  0, 143, 255]],

       [[  0, 138, 255]],

       [[  0, 131, 255]],

       [[  0, 126, 255]],

       [[  0, 121, 255]],

       [[  0, 115, 255]],

       [[  0, 111, 255]],

       [[  0, 106, 255]],

       [[  0, 100, 255]],

       [[  0,  94, 255]],

       [[  0,  89, 255]],

       [[  0,  84, 255]],

       [[  0,  78, 255]],

       [[  0,  74, 255]],

       [[  0,  69, 255]],

       [[  0,  63, 255]],

       [[  0,  58, 255]],

       [[  0,  52, 255]],

       [[  0,  46, 255]],

       [[  0,  41, 255]],

       [[  0,  37, 255]],

       [[  0,  31, 255]],

       [[  0,  26, 255]],

       [[  0,  21, 255]],

       [[  0,  14, 255]],

       [[  0,   9, 255]],

       [[  0,   4, 255]],

       [[  0,   0, 255]],

       [[ 14,   0, 255]],

       [[ 31,   0, 255]],

       [[ 46,   0, 255]],

       [[ 63,   0, 255]],

       [[ 78,   0, 255]],

       [[ 94,   0, 255]],

       [[111,   0, 255]],

       [[126,   0, 255]],

       [[143,   0, 255]],

       [[158,   0, 255]],

       [[175,   0, 255]],

       [[190,   0, 255]],

       [[207,   0, 255]],

       [[222,   0, 255]],

       [[239,   0, 255]]], dtype=uint8)


def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    point = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    latitude = point.latitude
    longitude = point.longitude

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = latitude
    camera.exif_tags['GPS.GPSLongitude'] = longitude

    # Capture the image
    camera.capture(image)

def save_image(base_folder, count):
    """Saves the image in the base folder and uses the capture function to take an image"""
    try:
        cam = PiCamera() # opens camera
        cam.resolution = (1296,972) # sets resolution
        count_string = str(count)
        capture(cam, f"{base_folder}/gps" + count_string + ".jpg") # saves the taken image in the base folder
        cam.close()
    except KeyboardInterrupt:
        print("Camera stopped by user") # Error logging
    except Exception as e:
        print("Error occurred:", e) # Error logging
    finally:
        filename = f"{base_folder}/gps" + count_string + ".jpg"
        return filename
# ndvi 
def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

def convert_image_to_ndvi(base_folder, image, now_time, count):
    timestamp = str(now_time)
    contrasted = contrast_stretch(image)
    ndvi = calc_ndvi(contrasted)
    ndvi_contrasted = contrast_stretch(ndvi)
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, colour_map)
    cv2.imwrite('color_mapped_image.png', color_mapped_image)
    file_name = f"{base_folder}/" + count + timestamp + "_color_mapped_image.png"
    cv2.imwrite(file_name, color_mapped_image)
    return file_name

# end ndvi
def get_spatial_resolution(img1, img2):
    # Load the two images
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    
    # Read the latitude and longitude values from the EXIF metadata
    lat1, lon1 = get_lat_lon_from_exif(img1)
    lat2, lon2 = get_lat_lon_from_exif(img2)
    
    # Calculate the difference in latitude and longitude
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Calculate the number of pixels shifted in the x and y directions
    shift_x, shift_y = get_pixel_shift(image1, image2)
    
    # Divide the difference in latitude and longitude by the number of pixels shifted
    resolution_x = delta_lon / shift_x # size of each pixel in lat and long
    resolution_y = delta_lat / shift_y
    
    return resolution_x, resolution_y

def get_lat_lon_from_exif(image_path):
    with open(image_path, 'rb') as f:
        tags = Image(f)
        lat = tags.get('GPS.GPSLatitude', None)
        lon = tags.get('GPS.GPSLongitude', None)
        return lat, lon

def get_pixel_shift(img1, img2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # gray scale allows us to see the way it changes 
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the shift between the two images using a phase correlation method
    shift, error, phase_diff = cv2.phaseCorrelate(gray1, gray2) # higher the error value the less accurate the results are: the goal is to take this value for n images taken
    # hence this should be run n/2 times, and the least error value should be taken for most accurate results 
    # Round the shift to the nearest integer
    shift_x, shift_y = np.round(shift).astype(int)
    
    return shift_x, shift_y

def calculate_ndvi(img):
    # Convert the image to a floating-point array
    img = img.astype(np.float32)
    
    # Split the image into red, green, and blue channels
    red = img[:, :, 2]
    blue = img[:, :, 0]
    green = img[:, :, 1]
    
    # Calculate the NDVI of each pixel
    ndvi = (red - blue) / (red + blue + 1e-9)
    
    return ndvi

def create_pixel_dict(pixel_array, central_lon, central_lat, pixel_size_lat, pixel_size_lon ,  num_rows, num_cols, base_folder, count):
    # Calculate the starting latitude and longitude of the top-left pixel
    start_lon = central_lon - ((num_cols // 2) * pixel_size_lon)
    start_lat = central_lat + ((num_rows // 2) * pixel_size_lat)
    count_string = str(count)
    filename_create = f"{base_folder}/" + count_string + ".txt"
    file = open(filename_create, "w+")
    file.close()
    # row -> colum 
    for col in range(num_rows-1):
        for row in range(num_cols-1):
            lon_pixel_row = str(start_lon + row*pixel_size_lon)
            lat_pixel_col = str(start_lat + col*pixel_size_lat)
            ndvi_pixel = str(pixel_array[row][col])
            ndvi_string_pixel = "ndvi: " + ndvi_pixel
            file = open(filename_create, "a")
            content = lat_pixel_col + "," + lon_pixel_row + "=" + ndvi_string_pixel
            file.write(content)
            file.close()

def cv_size(img):
    return tuple(img.shape[1::-1])


# Create a `datetime` variable to store the start time
start_time = datetime.now()
# Create a `datetime` variable to store the current time
# (these will be almost the same at the start)
now_time = datetime.now()
count = 1
# Run a loop for 3 hours
while (now_time < start_time + timedelta(minutes=180)):
    base_folder = Path(__file__).parent.resolve() # resolves base folder where the image where the image will be saved
    for i in range(1,3):
        if i == 1:
            filename1 = save_image(base_folder=base_folder, count=count)
            image_original1 = cv2.imread(filename1)
            filename_colorImage1 = convert_image_to_ndvi(base_folder=base_folder, image=image_original1, now_time=now_time, count=count)
        else:
            count1 = count + 1
            filename2 = save_image(base_folder=base_folder, count=count)
            image_original2 = cv2.imread(filename2)
            filename_colorImage2 = convert_image_to_ndvi(base_folder=base_folder, image=image_original2, now_time=now_time, count=count1)
    
    pixel_x, pixel_y = get_spatial_resolution(filename_colorImage1, filename_colorImage2)
    
    img1 = cv2.imread(filename_colorImage1)
    ndvi1 = calculate_ndvi(img1)
    lat1, lon1 = get_lat_lon_from_exif(filename_colorImage1)
    size1 = cv_size(img1)

    create_pixel_dict(ndvi1, lon1, lat1, pixel_x, pixel_y ,  size1[0], size1[1], base_folder, count)
    
    img2 = cv2.imread(filename_colorImage2)
    ndvi2 = calculate_ndvi(img2)
    lat2, lon2 = get_lat_lon_from_exif(filename_colorImage2)
    size2 = cv_size(img2)

    create_pixel_dict(ndvi2, lon2, lat2, pixel_x, pixel_y ,  size2[0], size2[1], base_folder, count1)

    
    
    
    sleep(5)
    # Update the current time
    now_time = datetime.now()
    count = count + 1

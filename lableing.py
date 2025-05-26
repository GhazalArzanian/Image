import cv2
import os
import random
import numpy as np

# class ids for each symbol
dict_name_class_id = {
    'shut-off_valve_on': 0,
    'shut-off_valve_off': 1,
    'pump_on_up': 2,
    'pump_on_right': 3,
    'pump_on_left': 4,
    'pump_on_down': 5,
    'pump_off_up': 6,
    'pump_off_right': 7,
    'pump_off_left': 8,
    'pump_off_down': 9,
    'metering_device': 10,
    'frequency_inverter_on': 11,
    'frequency_inverter_off': 12,
    'digital_volume_sensor_on': 13,
    'digital_volume_sensor_off': 14,
    'digital_temperature_sensor': 15,
    'digital_relative_humidity_sensor': 16,
    'digital_differential_pressure_sensor': 17,
    'digital_absolute_humidity_sensor': 18,
    'differential_pressure_sensor': 19,
    'analog_pressure_sensor': 20,
    'analogue_relative_humidity_sensor': 21,
    '3-way-control_valve': 22,
    '2-way-control_valve':23,

    # following are the ids defined for relatively larged sized symbols/components compared to above enlisted components
    'chiller':24,
    'chiller_off':25,
    'chiller_on':26,
    'combined_coarse_fine_filter_right':27,
    'constant_volume_flow_controller_left':28,
    'constant_volume_flow_controller_right':29,
    'consumer':30,
    'cooling_coil':31,
    'damper_off':32,
    'discharge_well':33,
    'electical_heater':34,
    'electrical_air_damper_on':35,
    'electrical_hot_water_storage_tank':36,
    'exhaust_air_combined_fine_coarse_filter_left':37,
    'fan_on_left':38,
    'fan_on_left_1':39,
    'fan_on_right':40,
    'fan_on_right_1':41,
    'fan_on_right_2':42,
    'fan_right': 43,
    'filter_left':44,
    'filter_right':45,
    'fine_filter':46,
    'heat_exchanger':47,
    'heat_exchanger_ventilation_system':48,
    'heating_coil':49,
    'inlet_vane_controlled_fan_off_left':50,
    'inlet_vane_controlled_fan_off_right':51,
    'inlet_vane_controlled_fan_on_left_1':52,
    'inlet_vane_controlled_fan_on_right_1':53,
    'pressurization_system':54,
    'rooftop_chiller_unit':55,
    'room_switch_off':56,
    'room_switch_on':57,
    'rotary_heat_exchanger':58,
    'steam_humidifer':59,
    'steam_humidifier_off':60,
    'steam_humidifier_on':61,
    'variable_volume_flow_controller_left':62,
    'variable_volume_flow_controller_right':63,
    'vertical_heat_exchanger_heating_cooling':64,
    'water_storage_tank':65,
    'water_storage_tank_1':66

}


#ground_truth_folder = 'ground_truth_mutated/'
ground_truth_folder = 'ground_truth_dilt_labelled/'
context_images_folder = 'context_images/'
output_image_folder = 'synthetic_dataset/'
output_label_folder = 'synthetic_labels/'

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

num_images = 100  #nr. of synthetic images to generate
image_width = 1700
image_height = 800
background_color = (230, 178, 172)  #background color -> purple

#load ground truth images
ground_truth_images = []
for symbol_file in os.listdir(ground_truth_folder):
    symbol_path = os.path.join(ground_truth_folder, symbol_file)
    symbol_image = cv2.imread(symbol_path, cv2.IMREAD_UNCHANGED)
    ground_truth_images.append((symbol_file, symbol_image))

#context images -> considered noise
context_images = []
for context_file in os.listdir(context_images_folder):
    context_path = os.path.join(context_images_folder, context_file)
    context_image = cv2.imread(context_path, cv2.IMREAD_UNCHANGED)
    context_images.append(context_image)

#to check if two bounding boxes overlap
def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)

def get_groundtruth_class_id(symbol_file):
    symbol_name = symbol_file.split('.')[0]
    for key in dict_name_class_id.keys():
        if key in symbol_name:
            return dict_name_class_id[key]
        

#to place symbols randomly on a blank canvas without overlapping
def generate_synthetic_image():
    canvas = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)  #background with specified color
    annotations = []
    occupied_regions = []  #keep track of regions occupied by both objects and context images

    #place ground truth symbols
    for symbol_file, symbol_image in ground_truth_images:
        symbol_height, symbol_width = symbol_image.shape[:2]
        max_x = image_width - symbol_width
        max_y = image_height - symbol_height
        placed = False
        attempts = 0

        while not placed and attempts < 100:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            overlap = False
            
            for ox, oy, ow, oh in occupied_regions:
                if is_overlapping(x, y, symbol_width, symbol_height, ox, oy, ow, oh):
                    overlap = True
                    break
            symbol_id = get_groundtruth_class_id(symbol_file)
            if not overlap:
                canvas[y:y+symbol_height, x:x+symbol_width] = symbol_image[:, :, :3]
                center_x = (x + symbol_width / 2) / image_width
                center_y = (y + symbol_height / 2) / image_height
                width = symbol_width / image_width
                height = symbol_height / image_height
                annotations.append((symbol_id, center_x, center_y, width, height))
                occupied_regions.append((x, y, symbol_width, symbol_height))
                placed = True
            attempts += 1

    #place context images
    for context_image in context_images:
        context_height, context_width = context_image.shape[:2]
        max_x = image_width - context_width
        max_y = image_height - context_height
        placed = False
        attempts = 0

        while not placed and attempts < 100:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            overlap = False
            
            for ox, oy, ow, oh in occupied_regions:
                if is_overlapping(x, y, context_width, context_height, ox, oy, ow, oh):
                    overlap = True
                    break
            
            if not overlap:
                canvas[y:y+context_height, x:x+context_width] = context_image[:, :, :3]
                occupied_regions.append((x, y, context_width, context_height))
                placed = True
            attempts += 1

    return canvas, annotations

#generate and save synthetic images and annotations
for i in range(num_images):
    canvas, annotations = generate_synthetic_image()
    image_path = os.path.join(output_image_folder, f'synthetic_{i}.jpg')
    label_path = os.path.join(output_label_folder, f'synthetic_{i}.txt')
    
    cv2.imwrite(image_path, canvas)
    
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(' '.join(map(str, ann)) + '\n')

print("Synthetic dataset generation completed.")



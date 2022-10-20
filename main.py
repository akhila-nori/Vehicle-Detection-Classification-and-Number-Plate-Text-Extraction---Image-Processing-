# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:38:04 2022

@author: Dell
"""

from detection import vehicle_detection
from recognition import vehicle_recognition
from number_plate_recognition import number_plate


'''
rv = vehicle_recognition()
print('segregate cars')
rv.recognise('extracted_cars\cars')
'''
np = number_plate()
np.text_extraction()
# vh = vehicle_detection()
# vh.detect('video5.mp4', 2)

# print("scooter video")
# vh.detect("input_videos\scooters.mp4", 1)

# print("car video")
# vh.detect("input_videos\scooters.mp4", 2)

# print("thailand video")
# vh.detect("input_videos\cars0.mp4", 2)

# print("single car driving towards camera", 2)
# vh.detect("input_videos\cars1.mp4", 2)


# print('segregate scooters')
# rv.recognise('extracted_scooters\scooters')


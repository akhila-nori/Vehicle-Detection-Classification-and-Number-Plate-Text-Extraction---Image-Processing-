# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:38:06 2022

@author: Dell
"""

import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt


class vehicle_detection:
    def detect(self, video_name, output_images):
        print("Reading video")
        video1 = cv2.VideoCapture('video5.mp4')
        print(video1.isOpened())
        i = 0
        frame_array_for_video = []
        col_images = []
        flag = 0

        while(video1.isOpened()):
            ret, frame = video1.read()  # reads frame by frame
            print(ret)

            if ret == False:
                break

            #frame = cv2.resize(frame, (240, 480))
            #cv2.imshow(" ", frame)
            #cv2.waitKey()



            #cv2.imwrite('frames/Frame'+str(i)+'.jpg', frame)
            col_images.append(frame)
            i += 1

        print(len(col_images))

        if(flag == 0):
            car_no = 0
            scooter_no = 0
            for j in range(50, len(col_images)-1):
                framei = j
                # for frame in [framei, framei+1]:
                # plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
                # plt.title("frame: "+str(frame))
                # plt.show()

                # convert the frames to grayscale
                grayA = cv2.cvtColor(col_images[framei], cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(col_images[framei+1], cv2.COLOR_BGR2GRAY)

                diff_image = cv2.absdiff(grayB, grayA)

                # plot the image after frame differencing
                #plt.imshow(diff_image, cmap='gray')
                #plt.title("subtraction of grayscale images")
                # plt.show()

                # image threshold -> 30
                ret, thresh = cv2.threshold(
                    diff_image, 30, 255, cv2.THRESH_BINARY)

                # plot image after thresholding
                #plt.imshow(thresh, cmap='gray')
                # plt.title('thresholding')
                # plt.show()

                # smoothening of images
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(thresh, kernel, iterations=1)

                # plot dilated image
                # plt.imshow(dilated, cmap='gray')
                # plt.title("smoothened")
                # plt.show()

                # vehicles are detected after they cross this point
                # dialated = cv2.line(dilated, (0, 400), (1250, 400), (100, 0, 0), 9)
                # dialated = cv2.line(dilated, (0, 500), (1250, 500), (100, 0, 0), 9)
                # cv2.imshow("",dialated)
                # cv2.waitKey()

                # find initial contours: gives multiple contours for one car
                contours, hierarchy = cv2.findContours(
                    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                valid_cntrs = []

                for i, cntr in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cntr)
                    if (y >= 600) & (cv2.contourArea(cntr) >= 1000):
                        valid_cntrs.append(cntr)

                # count of discovered contours
                # print('len of valid contours',len(valid_cntrs))

                dmy = col_images[framei+1].copy()
                dmy = cv2.line(dmy, (0, 500), (1280, 500), (100, 0, 0), 2)

                # filled rectangles (merged)
                img_merged_rect = np.zeros(dmy.shape)

                sorted_contours = sorted(
                    valid_cntrs, key=cv2.contourArea, reverse=True)

                # merge all prev contours
                for cnt in sorted_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_merged_rect, (x, y),
                                  (x+w, y+h), (0, 255, 0), -1)

                # cv2.imshow("colored",img_merged_rect)
                # cv2.waitKey()

                # conv to grayscale
                img_float32 = np.float32(img_merged_rect)
                grey = img_float32.astype(np.uint8)
                grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)

                #plt.title("grey scale image")
                # plt.imshow(grey)
                # cv2.waitKey()

                # find contours
                cntrs, hierarchy = cv2.findContours(
                    grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # mark contours
                merged = col_images[framei+1].copy()
                cv2.drawContours(merged, cntrs, -1, (0, 255, 0), 3)
                # cv2.imshow("merged",merged)
                # cv2.waitKey()

                for cnt in cntrs:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(merged, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    car = merged[y:y+h, x:x+w]
                    # cv2.imshow("car",car)
                    # cv2.waitKey()
                    if(output_images == 1):
                        cv2.imwrite('extracted_scooters\scooters' +
                                    str(car_no)+'.jpg', car)
                        car_no += 1

                    elif(output_images == 2):
                        cv2.imwrite('extracted_cars\cars' +
                                    str(scooter_no)+'.jpg', car)
                        scooter_no += 1

                height, width, layers = dmy.shape
                size = (width, height)
                frame_array_for_video.append(merged)

                if(output_images == 1):
                    # video name
                    pathOut = 'vehicle_detection_v1.mp4'

                elif(output_images == 2):
                    pathOut = 'vehicle_detection_v2.mp4'

                # frames per second
                fps = 14.0

                out = cv2.VideoWriter(
                    pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

                for i in range(len(frame_array_for_video)):
                    # writing to a image array
                    out.write(frame_array_for_video[i])

                out.release()



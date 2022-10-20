import cv2
import matplotlib.pyplot as plt
import numpy as np


class vehicle_recognition:
    def recognise(self, img_path):
        config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model='frozen_inference_graph.pb'
        model=cv2.dnn_DetectionModel(frozen_model,config_file)
        classLabels=[]
        file_name='labels.txt'
        with open(file_name,'rt') as fpt:
            classLabels=fpt.read().rstrip('\n').split('\n')
        print(classLabels)
        model.setInputSize(320,320)
        model.setInputScale(1.0/127.5)
        model.setInputMean((127.5,127.5,127.5))
        model.setInputSwapRB(True)
        for i in range(0,100):
          #read an image
          img=cv2.imread(img_path+str(i)+'.jpg')
          #plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
          cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
          ClassIndex,confidece,bbox=model.detect(img,confThreshold=0.35)
          #index=ClassIndex
          #print(len(ClassIndex))
          #if len(ClassIndex)>1:
          # index=ClassIndex[0]
          #classLabels[index-1]
          print("For car"+str(i))
          print(ClassIndex)
          count_arr = np.bincount(ClassIndex)
          if(count_arr[3]>0):
            print('Object is a car')
            cv2.imwrite('output\CARS\Car_Image'+str(i)+'.jpg',img)
          elif(count_arr[2]>0):
            print("Object is a bicycle")
            cv2.imwrite('output\BICYCLE\Bicycle_Image'+str(i)+'.jpg',img)
          elif(count_arr[4]>0):
            print("It's a motorbike")
            cv2.imwrite('output\MOTORCYCLE\Motorcycle_Image'+str(i)+'.jpg',img)
          elif(count_arr[6]>0):
            print("It's a bus")
            cv2.imwrite('output\BUS\Bus_Image'+str(i)+'.jpg',img)
          elif(count_arr[8]>0):
            print("It's a truck")
            cv2.imwrite('output\TRUCK\Truck_Image'+str(i)+'.jpg',img)
          else:
            print('Object not identified')
            cv2.imwrite('output\OTHERS\Image'+str(i)+'.jpg',img)

# rv = vehicle_recognition()
# print('segregate cars')
# rv.recognise('extracted_cars\cars')
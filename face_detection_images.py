import mediapipe as mp
import cv2

#Photo by Nicholas Green on Unsplash

#Importing the image from the same directory as that of the code file
img = cv2.imread("people.jpg")

#the image was 6000*4000 pixels which was too large and so I had to resize it
img = cv2.resize(img , (600 , 600) , interpolation = cv2.INTER_NEAREST)

#This function is the most important one as it detects faces in the image
mp_face_detection = mp.solutions.face_detection

#This function is used for drawing on the image, can be easily substituted
#using custom functions
mp_drawing = mp.solutions.drawing_utils

#This is how you use the Face Detection function in mediapipe

#The model_selection parameter should be kept 1 if we are looking for faces
#that are more then 2 meters away from the camera and 0 if we are interested
#in faces less than 2 meters away from the camera.
#min_detection_confidence = 0.5 means that the bounding boxes will only be
#considered if the model is more than 50% sure that there exists a face in it.
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:

  #The face_detection.process function is the one that actually takes the
  #image as an input and gives us the output which are landmarks and bounding
  #boxes, also opencv uses the BGR format for image colours but Mediapipe
  #uses the RGB format, so convert BGR to RGB.
  results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  #Here we make a copy of the original image so that we have it as refernce
  annotated_image = img.copy()

  #Here, the results is what we got from the process function above as an
  #output and results.detections has all the data relating to the bounding
  #boxes.
  print("Number of people detected " , len(results.detections) , "\n\n")
  print("Description of people\n")

  #We then run a loop for each face detected.
  for i , detection in enumerate(results.detections):

      #We print the id of the person that we are dealing with
      print("Person" , i , "\n")

      #The data that we are interested in is in a very nested form and so
      # we have shortened the form to the variable box
      box = detection.location_data.relative_bounding_box

      #Below is the way we find the real start coordinates of the left top corner
      #of the bounding box for the image 
      x_start , y_start = int(box.xmin * img.shape[1]) , int(box.ymin * img.shape[0])

      #Below is the way we find the real end coordinates of the right bottom
      #corner of the bounding box for the image
      x_end , y_end = int((box.xmin + box.width) * img.shape[1]) , int((box.ymin + box.height) * img.shape[0])

      #We now display the bounding box in the annotated imag using the
      #cv2.rectangle function of opencv
      annotated_image = cv2.rectangle(annotated_image , (x_start , y_start) , (x_end , y_end) , (0 , 255 , 0) , 5)

      #These are the landmarks that we get from mediapipe face detection
      #There are 6 keypoints in the image.
      #uncomment these lines below to get the coordinates printed in the console
      '''print("Nose co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      print("Mouth co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER))
      print("Right eye co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE))
      print("Left eye co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
      print("Right ear co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION))
      print("Left ear co-ordinates\n" , mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION))'''

      #Uncomment the line below if you want mediapipe to display its own results
      #which are bounding boxes and the landmarks
      '''mp_drawing.draw_detection(annotated_image, detection)'''

#Finally display the image
cv2.imshow("test" , annotated_image)


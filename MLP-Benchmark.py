import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from math import sqrt, pow, acos
import statistics
import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#------------variables:--------------------------------------------------------------
classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'space', 'null')

numofclasses = len(classes)

#configuration variables
fromwebcam = True
savevideo = False
savebenchmark = False

is_3D = False

# Model name - change this when loading new model weights
model_name='large_NA_SN_old' #this dir will be made inside benchmark dir
#model_name = sys.argv[1]

# Filenames and paths
benchmarktype_name = 'benchmark_v3' #must be existing dir
weightsfile = '../mlpmodels/'+model_name+'/weights.pth'
benchmarklabelfile = 'media/benchmark/'+benchmarktype_name+'/inputlabels.txt'
#if not fromwebcam:
inputvideo = 'media/benchmark/'+benchmarktype_name+'/'+'inputvideo.mp4'
#if savevideo ans savebenchmark:
outputvideo = 'media/benchmark/'+benchmarktype_name+'/'+model_name+'/output.mp4'
#if savebenchmark:
outputbenchmark = 'media/benchmark/'+benchmarktype_name+'/'+model_name+'/detections.txt'
outputconfmtx = 'media/benchmark/'+benchmarktype_name+'/'+model_name+'/conf_mtx.png'
outputthresholdplot = 'media/benchmark/'+benchmarktype_name+'/'+model_name+'/threshold-acc.png'

# Model parameters extracted from the model name
parameters = model_name.split('_')
modelsize = parameters[0] #large OR small
withnullclass = False
if len(parameters) >= 2:
    if parameters[1] == 'NULL':
        withnullclass = True

#variables for statistics
truth_array = []
pred_array = []
conf_array = []

#------------network:--------------------------------------------------------------

if not withnullclass: numofclasses = numofclasses - 1

class SmallNet(nn.Module):
    def __init__(self, numofclasses):
        super().__init__()
        self.fc1 = nn.Linear(42, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, numofclasses)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LargeNet(nn.Module):
    def __init__(self, numofclasses):
        super().__init__()
        self.fc1 = nn.Linear(42, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.fc6 = nn.Linear(500, numofclasses)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

if   modelsize == 'small': net = SmallNet(numofclasses)
elif modelsize == 'large': net = LargeNet(numofclasses)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

model = net.to(device)

# Load trained weights from file
model.load_state_dict(torch.load(weightsfile, map_location=torch.device('cpu')))
model.eval()

#------------functions:--------------------------------------------------------------

# Displays detection result on the video frames
def drawText(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(100, 100),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(255, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y - text_h - 2), (x + text_w, y), text_color_bg, -1)
    cv2.putText(img, text, (x, y - 2), font, font_scale, text_color, font_thickness)

# Plots and saves threshold - accuracy graph
def plotThresHoldGraph(thresholds, accuracies, best_result, filepath):
    text = 'Thr.: '+str(round(best_result[0], 3))+' Acc.: '+str(round(best_result[1], 3))
    plt.figure(dpi=500)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.axis([0, 1, 0, 1])
    plt.plot(thresholds, accuracies, 'g')
    plt.annotate(text, xy=best_result, xytext=(0.4, 0.9),
             arrowprops=dict(facecolor='black', width=1, shrink=0.05),
             )
    plt.savefig(filepath)

# Calculates data for threshold - accuracy graph
def saveThresholdGraph(truths, preds, confs, resolution, classes, filepath):
    step = 1.0 / float(resolution)
    thresholds = []
    accuracies = []
    best_result = (0.0, 0.0) #(threshold, accuracy)
    for i in range(0, resolution + 1):
        threshold = i * step
        thresholds.append(threshold)
        correct = 0
        for i in range(0, len(truths)):
            if ((truths[i] == preds[i] and confs[i]>=threshold) or
                (truths[i] == classes.index('null') and confs[i]<threshold)):
                correct = correct + 1
        accuracy = correct / len(truths)
        if accuracy > best_result[1]:
            best_result = (threshold, accuracy)
        accuracies.append(accuracy)
    plotThresHoldGraph(thresholds, accuracies, best_result, filepath)
    return best_result[0]

# Normalize hand coordinates
def transformPoints(handcoords):
    x_coords = []
    y_coords = []
    for idx, coord in enumerate(handcoords):
        if idx % 2: y_coords.append(coord)
        else: x_coords.append(coord)
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    transformed = []
    for idx, coord in enumerate(handcoords):
        if idx % 2:
            tf_coord = (coord - y_min) / (y_max - y_min)
        else:
            tf_coord = (coord - x_min) / (x_max - x_min)
        transformed.append(tf_coord)
    return transformed

# Overwrites predictions with low confidences to null
def applyThreshold(pred_array, conf_array, best_threshold):
    adjusted_pred_array = []
    for idx, pred in enumerate(pred_array):
        if conf_array[idx] < best_threshold:
            adjusted_pred_array.append(classes.index('null'))
        else:
            adjusted_pred_array.append(pred)
    return adjusted_pred_array

def feedForward(handcoords):
    testdata = []
    handcoords = transformPoints(handcoords)
    coords_tensor = torch.Tensor(handcoords)
    element = (coords_tensor, 0)
    testdata.append(element)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1)
    with torch.no_grad():
      for data in testloader:
          images, labels = data[0].to(device), data[1].to(device)
          # Calculate outputs by running images through the network
          outputs = net(images)
          # Apply softmax to get 0-1 range probability outputs
          soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
          # The class with the highest score is what we choose as prediction
          prob, predicted = torch.max(soft_outputs.data, 1)
          return prob, predicted

#------------process:--------------------------------------------------------------

frames = 1
benchmarktext = ""
imgtext = ""

if fromwebcam:
  cap = cv2.VideoCapture(0)
else:
  cap = cv2.VideoCapture(inputvideo)

if (cap.isOpened() == False):
  print("Unable to open video file")

if savebenchmark:
  os.mkdir('media/benchmark/'+benchmarktype_name+'/'+model_name)
  benchmarkout = open(outputbenchmark, 'w')
  with open(benchmarklabelfile, 'r') as benchmarkonlylabels:
    for line in benchmarkonlylabels:
      truth_array.append(classes.index(line.rstrip()))

if savevideo:
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  out = cv2.VideoWriter(outputvideo, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  # Loop until the video has frames
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Run MediaPipe Hands then the MLP
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        # Check if right hand is detected, select if it is, abort if not
        righthand_idx = 0
        success = False
        for i, handedness in enumerate(results.multi_handedness):
            if handedness.classification[0].label == 'Left': # Since it is mirrored by default
                righthand_idx = i
                success = True
                break
        if success: # If right hand found
            hand_landmarks = results.multi_hand_landmarks[righthand_idx] # Only process one hand
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Assemble data for MLP
            handcoords = []
            for point in hand_landmarks.landmark:
                handcoords.append(point.x)
                handcoords.append(point.y)

            # Run MLP on keypoints
            prob, predicted = feedForward(handcoords)
            prob_f = float(prob[0])
            pred = predicted[0]
        else: # If no right hand found
            prob_f = 1.0
            pred = classes.index('null')

    else: # If no hand found
      prob_f = 1.0
      pred = classes.index('null')

    # Generate text for benchmark file and video frame
    benchmarktext = str(frames) + '\t' + classes[pred] + '\t' + str(prob_f) + '\n'
    imgtext = classes[pred] + ' | Prob.: ' + str(prob_f)
    
    # Save testing data
    pred_array.append(pred)
    conf_array.append(prob_f)

    # Export testing data to txt file
    if savebenchmark:
      benchmarkout.write(benchmarktext)
      
    drawText(image, imgtext)
    # Save output video
    if savevideo: out.write(image)

    # Show output video in real-time, while running
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == ord('5'):
      break
    frames = frames + 1

# End of video
cap.release()

if savevideo:
    out.release()

# Save the testing data
if savebenchmark:
    best_threshold = saveThresholdGraph(truth_array, pred_array, conf_array, 100, classes, outputthresholdplot)
    adjusted_pred_array = applyThreshold(pred_array, conf_array, best_threshold)

    cf_matrix = confusion_matrix(truth_array, adjusted_pred_array)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt="1.0f")
    plt.savefig(outputconfmtx)
    benchmarkout.close()
import numpy as np
from collections import OrderedDict


class BoxTracker():
  def __init__(self, maxDisappeared=10):
    self.nextObjectID = 0
    self.objects = OrderedDict()
    self.disappeared = OrderedDict()
    self.maxDisappeared = maxDisappeared

  def register(self, box):
    box.id = self.nextObjectID
    self.objects[self.nextObjectID] = box
    self.disappeared[self.nextObjectID] = 0
    self.nextObjectID += 1
  
  def deregister(self, objectID):
    del self.objects[objectID]
    del self.disappeared[objectID]

  def update(self, inputBoxes):
    if len(inputBoxes) == 0:
      for objectID in list(self.disappeared.keys()):
        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)
      return self.objects

    if len(self.objects) == 0:
      for i in range(0, len(inputBoxes)):
        self.register(inputBoxes[i])
    
    else:
      objectIDs = list(self.objects.keys())
      boxes = list(self.objects.values())
      D = distances_boxes(boxes, inputBoxes)
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]
      usedRows = set()
      usedCols = set()
      for (row, col) in zip(rows, cols):
        if row in usedRows or col in usedCols:
          continue
        objectID = objectIDs[row]
        self.objects[objectID] = inputBoxes[col]
        self.objects[objectID].id = objectID
        self.disappeared[objectID] = 0
        usedRows.add(row)
        usedCols.add(col)
      unusedRows = set(range(0, D.shape[0])).difference(usedRows)
      unusedCols = set(range(0, D.shape[1])).difference(usedCols)
      
      if D.shape[0] >= D.shape[1]:
        for row in unusedRows:
          objectID = objectIDs[row]
          self.disappeared[objectID] += 1
          if self.disappeared[objectID] > self.maxDisappeared:
            self.deregister(objectID)
      else:
        for col in unusedCols:
          self.register(inputBoxes[col])
    return self.objects


def distance_box(box1, box2):
  cx1, cy1 = (box1.xmin + box1.xmax) / 2, (box1.ymin + box1.ymax) / 2
  cx2, cy2 = (box2.xmin + box2.xmax) / 2, (box2.ymin + box2.ymax) / 2
  return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def distances_boxes(boxes1, boxes2, dist=distance_box):
  res = np.zeros((len(boxes1), len(boxes2)))
  for i, box1 in enumerate(boxes1):
    for j, box2 in enumerate(boxes2):
      res[i,j] = dist(box1, box2)
  return res

def NMS(inputBoxes, overlapThresh=0.4):
  # Return an empty list, if no boxes given
  if len(inputBoxes) == 0:
    return []

  # Convert boxes to (:,4) array
  boxes = np.zeros((len(inputBoxes), 4))
  for i, box in enumerate(inputBoxes):
    boxes[i] = np.array([box.xmin, box.ymin, box.xmax, box.ymax])
  
  x1 = boxes[:, 0]  # x coordinate of the top-left corner
  y1 = boxes[:, 1]  # y coordinate of the top-left corner
  x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
  y2 = boxes[:, 3]  # y coordinate of the bottom-right corner

  # Compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)

  # The indices of all boxes at start. We will redundant indices one by one.
  indices = np.arange(len(x1))
  for i,box in enumerate(boxes):
    # Create temporary indices  
    temp_indices = indices[indices != i]

    # Find out the coordinates of the intersection box
    xx1 = np.maximum(box[0], boxes[temp_indices, 0])
    yy1 = np.maximum(box[1], boxes[temp_indices, 1])
    xx2 = np.minimum(box[2], boxes[temp_indices, 2])
    yy2 = np.minimum(box[3], boxes[temp_indices, 3])

    # Find out the width and the height of the intersection box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # Compute the ratio of overlap
    overlap = (w * h) / areas[temp_indices]

    # If the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
    if np.any(overlap) > overlapThresh:
      indices = indices[indices != i]

  # Return only the boxes at the remaining indices
  outputBoxes = []
  for i in indices:
    outputBoxes.append(inputBoxes[i])
  return outputBoxes
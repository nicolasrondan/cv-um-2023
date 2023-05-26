import numpy as np
import cv2
import matplotlib.pyplot as plt


def sliding_window(image, window_size, scale, stride):
    [image_rows, image_cols] = image.shape;
    window_rows = window_size[0];
    window_cols = window_size[1];

    patches = np.zeros((window_rows, window_cols,5));
    bbox_locations = np.zeros((5,4))
    r = np.random.randint(0,image_rows-window_rows,5); # Sample top left position
    c = np.random.randint(0,image_cols-window_cols,5);
    for i in range(0,5):
        patches[:,:,i] = image[r[i]:r[i]+window_rows, c[i]:c[i]+window_cols];
        bbox_locations[i,:] = [r[i],c[i],window_rows,window_cols]; # top-left y,x, height, width


    return patches, bbox_locations

def detections_gt(bboxes, iou_threshold=0.5):

    ground_thruth = []
    ratio = []

    for bbox in bboxes:
        ratio.append(bb_intersection_over_union(bbox, [82,91,84,84]));         
    ratio = np.asarray(ratio)
    ground_thruth = ratio>=iou_threshold
    return ground_thruth


def show_image_with_bbox(image,bboxes,draw_GT=True):
    GT = [82,91,166,175]
    if draw_GT:
        cv2.rectangle(image, (GT[0],GT[1]), (GT[2],GT[3]), (0, 0, 255), 2)

    for bbox in bboxes:
        if len(bbox) == 4:   
            top_left = (int(bbox[0]),int(bbox[1]))
            bottom_right = (int(bbox[0])+ int(bbox[2]),int(bbox[1])+int(bbox[3]))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

    cv2.imshow('image',image)
    cv2.waitKey(0) #wait for any key
    cv2.destroyAllWindows()

 
# Malisiewicz et al. From pyimagesearch modifid with boxes_probabilities to return only one bbox
def non_max_suppression_fast(boxes, overlapThresh, boxes_probabilities):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [[],0]
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    #x2 = boxes[:,2]
    #y2 = boxes[:,3]
    # Modify this for different bbox format

    x2 = boxes[:,0]+boxes[:,2]
    y2 = boxes[:,0]+boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type

    #filtered_probabilites = boxes_probabilities[pick]
    #max_prob_index = np.argmax(filtered_probabilites)

    #selected_bbox = boxes[max_prob_index].astype("int")
    #selected_score = boxes_probabilities[max_prob_index]
    selected_bbox = boxes[pick].astype("int")
    selected_score = boxes_probabilities[pick]

    return [selected_bbox, selected_score]
    #return boxes[pick].astype("int")

#PyImageSearch Intersection over Union
def bb_intersection_over_union(bboxA, bboxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = [bboxA[0],bboxA[1],bboxA[0]+bboxA[2], bboxA[1]+bboxA[3]]
    boxB = [bboxB[0],bboxB[1],bboxB[0]+bboxB[2], bboxB[1]+bboxB[3]]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # print('interArea {}'.format(interArea))
    # compute the area of both the prediction and ground-truth
    # rectangles

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # print('bboxAArea {}'.format(boxAArea))
    # print('bboxBArea {}'.format(boxBArea))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def non_max_suppression(boxes, probs, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [],[]

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    #x1 = boxes[:, 0]
    x1 = boxes[:, 1]
    #y1 = boxes[:, 1]
    y1 = boxes[:, 0]
    #x2 = boxes[:, 2]
    x2 = boxes[:, 1] + boxes[:,3]
    #y2 = boxes[:, 3]
    y2 = boxes[:,0] + boxes[:,2]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), probs[pick]


def draw_lines_polar(image,lines_params,theta_grad=False):
  fig, ax = plt.subplots()
  ax.imshow(image,  extent=[0, image.shape[1], 0, image.shape[0]])
  #phis = np.arange(-np.pi,np.pi,0.01)
  for line_p in lines_params:
    rho = line_p[0]
    theta = line_p[1]
    print(f'drawing rho {rho} theta {theta}')
    draw_line_polar(image,rho,theta,ax,theta_grad)

def draw_line_polar(image,rho,theta,ax,theta_grad=False):
  if theta_grad:
    theta = np.deg2rad(theta)
  phis = np.arange(-np.pi,np.pi,0.01)
  xs = np.array([ (rho/np.cos(phi-theta))*np.cos(phi) for phi in phis])
  ys = np.array([ (rho/np.cos(phi-theta))*np.sin(phi) for phi in phis])
  xs_in_image = (xs>0) & (xs < image.shape[1])
  ys_in_image = (ys>0) & (ys < image.shape[0])
  filter = xs_in_image & ys_in_image
  xs = xs[filter]
  ys = ys[filter]
  ax.plot(xs,ys,'r-')

def draw_lines(img,lines_params):
    fig, ax = plt.subplots()
    ax.imshow(img,  extent=[0, img.shape[1], 0, img.shape[0]])
    lin = np.linspace(0,img.shape[1])
    for params in lines_params:
        ax.plot(lin,lin*params[0]+params[1],'r-')
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])
    plt.show()     



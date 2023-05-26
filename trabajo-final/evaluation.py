import numpy as np
from image_utils import *

def evaluate_detector(bboxes, positive_probabilites, gt_box=[82,91,84,84]):
    
    true_positives_number = np.zeros((100))
    actual_positives = np.zeros((100))
    predicted_positives = np.zeros((100))
    overlap_threshold = 0.3
    mIoU = 0.3

    for i in np.arange(0,1,0.01):
        probability_threshold = i
        idx = int(np.round(i*100))

        true_positives_number[idx] = 0;
        actual_positives[idx] = 1
        predicted_positives[idx] = 0
        
        if len(bboxes) > 0:

            positive_bboxes = bboxes[positive_probabilites>=probability_threshold]
            positive_bboxes_prob = positive_probabilites[positive_probabilites>=probability_threshold]
           
            if len(positive_bboxes) > 0:
                [selected_bboxes, selected_scores] = non_max_suppression(positive_bboxes, positive_bboxes_prob,0.3)

                ratio = []
                for selected_bbox in selected_bboxes:
                    ratio.append(bb_intersection_over_union(selected_bbox, gt_box));                                   
                
                ratio = np.asarray(ratio)
                positive_number = sum(ratio>=mIoU); 
                
                true_positives_number[idx] = positive_number>=1;
                actual_positives[idx] = 1
                predicted_positives[idx] = len(ratio)
            

    return [true_positives_number, actual_positives, predicted_positives]

def precision_and_recall(true_positives_number,actual_positives,predicted_positives):

    summed_true_positives = np.sum(true_positives_number,axis=0)
    total_positives = np.sum(actual_positives,axis=0)
    summed_predicted_positives = np.sum(predicted_positives,axis=0)
    
    precision = np.divide(summed_true_positives,summed_predicted_positives)
    recall = np.divide(summed_true_positives,total_positives)
    return np.nan_to_num(precision), np.nan_to_num(recall)

def interpolated_average_precision(recall,precision):

    mprecision = np.concatenate(([0],precision,[0]))
    mrecall = np.concatenate(([0],recall,[1]))
    
    for i in np.arange(len(mprecision)-2,-1,-1):
        mprecision[i]=max(mprecision[i],mprecision[i+1]);

    #for i=numel(mpre)-1:-1:1
    #    mpre(i)=max(mpre(i),mpre(i+1));
    non_zero = np.nonzero(mrecall[1:] != mrecall[0:-1]) 
    i=non_zero + np.ones(len(non_zero))
    i=i.astype('uint8')
    
    ap=np.sum(np.multiply((mrecall[i]-mrecall[i-1]),mprecision[i]),axis=1);

    return ap

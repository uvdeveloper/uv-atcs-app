# -----------------------------------------------
# Pascal VOC Evaluation Code for object detection
# Author: Deval Shah
# Date  : 07-08-2018
# Email : devalshah190@gmail.com
# -----------------------------------------------

import os
import numpy as np
import argparse
import json
import pprint

np.seterr(divide='ignore', invalid='ignore')
tc = 0

#Modify the json fn as per your json input 
def parse_json(filename,cDict):
    objects = []
    cDict = {v:k for k,v in cDict.items()}
    with open(filename) as f:
        data = json.load(f)
    for i in data['bbox']:
        obj_struct = {}
        if i["attributes"]!={}:
            name = list(i["attributes"]["MT-Names"].keys())[0]
            obj_struct['name'] = cDict[name]
            xmin = i['coordinates']['xmin']
            ymin = i['coordinates']['ymin']
            xmax = i['coordinates']['xmax']
            ymax = i['coordinates']['ymax']
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
            objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def readDir(directory):
    fp = []
    for file in os.listdir(directory):
        if file.endswith(".txt") and file!="image_set_file.txt":
            fp.append(os.path.abspath(os.path.join(directory, file)))
    fl = []
    fp.sort()
    for i in range(len(fp)):
        fl.append(readFile(fp[i],i))
    return fl

def readFile(fname,image_id):
    with open(fname) as f:
        content = f.readlines()
        t1 = []
        #print(content)
        for x in content:
            t = list(map(float, x.strip().split(' ')))
            t.insert(0, float(image_id))
            t1.append(t)
        content = t1
    return content

def mergeClass(fl,classID):
    classArr = []
    classID = float(classID)
    for i in fl:
        for j in i:
            if j[1] == classID:
                classArr.append(j)
    return classArr

def display(BB,class_recs,im_dict):
        print("\nDetected Bounding Boxes \n")
        print(BB)
        print("\nGround Truth BBOX \n")
        for image_id,imagename in im_dict.items():
            print(class_recs[imagename]['bbox'])
        print("\n")

def classesDict(fname):
    classes = {}
    with open(fname) as f:
        content = f.readlines()
        for i in range(len(content)):
            classes[i] = content[i].split('\n')[0]
    return classes,len(classes)

def displayResults(cDict, rec, prec, ap,class_cnt):
    if cDict[i].upper() != "PERSON" and str(round(rec[-1], 3) * 100)!= "nan":
        print("\n\t\tClass " + cDict[i])
        print("\n\t\tTotal 		  : ", class_cnt)
        print("\t\tRecall            : ", str(round(rec[-1], 3) * 100)[:4], "%")
        print("\t\tPrecision         : ", str(round(prec[-1], 3) * 100)[:4], "%")
        print("\t\tAverage Precision : ", str(round(ap, 3) * 100)[:4], "%")
	

def displayForm():
    """	rec = tp / float(npos)
        prec = tp / (tp + fp)
    """
    print("\n\t\t******************************************************")
    print("\t\tFormulas for used for evaluating the object detections")
    print("\t\t******************************************************")
    print("\n\t\tPrecision 	  :  TRUE POSITIVE/(TRUE POSITIVE + FALSE POSITIVE)\n") 
    print("\t\tRecall 	      	  :  TRUE POSITIVE/(TRUE POSITIVE + FALSE NEGATIVE)\n")
    print("\n\t\tAP (average precision) is computed as the average of maximum precision at these 11 recall levels ranging from 0 to 1.1 at a 0.1 step.")
    print("\t\tThis approximates the the total area under the curve and divides it by 11. \n")
   

def voc_eval(detpath,
             annopath,
             imagesetfile,
             class_id,
             cDict,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the # Co - ordinates of bboxPASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    class id: ID of the class
    cDict = Dictionary contatining mapping of the classes from names text file
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    
    # first load gt
    """
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    """
    
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    global tc
    if tc == 0:
        print("\n\t\t*******************************************")
        print("\t\tTotal number of the images evaluated : ",len(imagenames))
        print("\t\t*******************************************") 
        tc = tc + 1

    im_dict = {}
    recs = {}

    for i, imagename in enumerate(imagenames):
        #recs[imagename] = parse_rec(annopath.format(imagename))
        tmp = imagename[:-4] + ".json"
        im_dict[i] = imagename
        recs[imagename] = parse_json(annopath+tmp,cDict)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    class_cnt = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == int(class_id)]
        class_cnt = class_cnt + len(R)
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        
   
    
    # read dets
    fl = readDir(detpath)
    splitlines = mergeClass(fl,class_id)
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[2]) for x in splitlines])
    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])
    if list(BB):
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        #display(BB,class_recs,im_dict)

        for d in range(nd):
            #print(im_dict[image_ids[d]])
            R = class_recs[im_dict[image_ids[d]]]
            #print(R)
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)


                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] =  1.
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / (tp + fp)
        ap = voc_ap(rec, prec, use_07_metric)
        
        return rec, prec, ap,class_cnt
    else:
        return [],[],-0.1,0
    
 
if __name__ == "__main__":    
    ap = argparse.ArgumentParser()
    ap.add_argument("-detpath", required=True,help="Give directory path where all the darknet predictions are stored")
    ap.add_argument("-annopath", required=True,help="Annotation files path")
    ap.add_argument("-imagesetfile", required=True,help="Image set file containing image names in each line of the file")
    ap.add_argument("-class_id", required=False,help="Give the class id for which you want to calulate AP")
    ap.add_argument("-names_file", required=True,help="Give the text file containing names of the classes")
    args = vars(ap.parse_args())
    cDict,total_classes = classesDict(args["names_file"])
    displayForm()

    for i in range(total_classes):
        if cDict[i].upper() != "PERSON":
            rec, prec, ap, class_cnt =  voc_eval(args["detpath"],args["annopath"],args["imagesetfile"],i,cDict,ovthresh=0.4,use_07_metric=False)
            if ap != -0.1 and str(round(rec[-1], 3) * 100)!= "nan":
                displayResults(cDict,rec,prec,ap,class_cnt)
                f = open("result.txt","a")
                f.write("\n"+cDict[i]+"\n")
                f.write("\t\tRecall : "+str(round(rec[-1],3)*100)[:4]+"%"+"  Precision : "+str(round(prec[-1],3)*100)[:4]+"%"+"   Average Precision : "+str(round(ap,3)*100)[:4]+"%" +"\n")
                f.close()
            else:
                print("\n\t\tClass "+cDict[i]+" not found")
                f = open("result.txt","a")
                f.write("\n"+cDict[i]+" not found\n")
                f.close()

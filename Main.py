import cv2
import numpy as np
import pathlib
import os, shutil
from shutil import copyfile
from preprocessing import Prep as p
from featext.texture import Haralick as har
from featext.texture import Tamura as tam
from featext.texture import King as k
from featext.physical import Gabor as g
from mlmodels import Classifiers as CLF
from mlmodels import DecisionSurfacePlotter as DSP
import time

imgcount = 0

''' 
A Generic Show Image function which shows the different type of pre-processed set of images for a particular test-image
'''
def __showImages(lstofimgs):
    print("Generating feature images")
    for tpls in lstofimgs:
        # cv2.namedWindow(tpls[1], cv2.WINDOW_NORMAL)
        # cv2.imshow(tpls[1], tpls[0])
        if (tpls[2] != None):
            cv2.imwrite(tpls[2], tpls[0])
        else:
            continue
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Feature images generated")

'''
Creating Training Sets from a collection of various skin-lesion images placed in their respective
class folders i.e., 'images/benign', 'images/malignant', 'images/negative'. These images are pre-processed
and a set of quantified-features are extracted from them, which comprises the 'training-set' data.
'''
def __createDataSet(restype, img_num):
    print("=====FOR %s SET===== \n" % restype.upper())
    if (((pathlib.Path('dataset.npz')).exists() == True) & ((pathlib.Path('dataset.npz')).is_file() == True)):
        dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    else:
        dset = np.empty(0, dtype=np.dtype([('featureset', float, (34,)), ('result', object)]), order='C')
        featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
    for i in range(0, img_num, 1):
         os.makedirs('results/dataset/' + restype + '/' + str(i))
         global imgcount
         print("Iterating for image - %d \n" % i)
         obj = p.Prep('images/' + restype + '/' + str(i) + '.jpg')
         feobj = har.HarFeat(obj.getSegGrayImg())
         feobj2 = tam.TamFeat(obj.getSegGrayImg())
         feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
         feobj4 = k.KingFeat(obj.getSegGrayImg())
         featarr = np.empty(0, dtype=float, order='C')
         featarr = np.insert(featarr, featarr.size, feobj.getAngularSecondMomentASM(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getEnergy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getEntropy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getHomogeneity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDm(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getCorrelation(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getHarCorrelation(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getClusterShade(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getClusterProminence(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment1(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment2(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment3(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getMoment4(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDasm(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDmean(), 0)
         featarr = np.insert(featarr, featarr.size, feobj.getDentropy(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getCoarseness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getKurtosis(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getLineLikeness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getDirectionality(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getRegularity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj2.getRoughness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getAsymmetryIndex(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getCompactIndex(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getFractalDimension(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getDiameter(), 0)
         featarr = np.insert(featarr, featarr.size, feobj3.getColorVariance(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsCoarseness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsContrast(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsBusyness(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsComplexity(), 0)
         featarr = np.insert(featarr, featarr.size, feobj4.getKingsStrength(), 0)
         dset = np.insert(dset, dset.size, (featarr, restype), 0)
         imgcount = imgcount + 1
    np.savez('dataset', dset=dset, featnames=featnames)

'''
The above generated training data, is then passed on to the various classifier/regressor objects for training/learning.
The trained models are persistently saved as python objects or pickle units, in individual '.pkl' files.
'''
def __createAndTrainMlModels():
    dset, featnames = (np.load('dataset.npz'))['dset'], (np.load('dataset.npz'))['featnames']
    CLF.Classifiers(featureset=dset['featureset'], target=__convertTargetTypeToInt(dset['result']), mode='train', path='mlmodels/')
    print("Training successfully completed!!! \n")

def __getTestImages():
    count = 0
    dset = np.empty(0, dtype=np.dtype([('featureset', float, (34,)), ('result', object)]), order='C')
    featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
    while(True):
        os.makedirs('results/testset/' + str(count))
        print("Iterating for image - %d \n" % count)
        imgnm = str(input('Enter image name : \n'))
        obj = p.Prep('temp/' + imgnm)
        feobj = har.HarFeat(obj.getSegGrayImg())
        feobj2 = tam.TamFeat(obj.getSegGrayImg())
        feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
        feobj4 = k.KingFeat(obj.getSegGrayImg())
        __showImages([(obj.getActImg(), 'imgcol' + str(count), 'results/testset/' + str(count) + '/' + 'imgcol' + str(count) + '.jpg'),
                      (obj.getGrayImg(), 'imggray' + str(count), 'results/testset/' + str(count) + '/' + 'imggray' + str(count) + '.jpg'),
                      (obj.getInvrtGrayImg(), 'imggrayinvrt' + str(count), 'results/testset/' + str(count) + '/' + 'imggrayinvrt' + str(count) + '.jpg'),
                      (obj.getBinaryImg(), 'imgbin' + str(count), 'results/testset/' + str(count) + '/' + 'imgbin' + str(count) + '.jpg'),
                      (obj.getSegColImg(), 'segimgcol' + str(count), 'results/testset/' + str(count) + '/' + 'segimgcol' + str(count) + '.jpg'),
                      (obj.getSegGrayImg(), 'segimggray' + str(count), 'results/testset/' + str(count) + '/' + 'segimggray' + str(count) + '.jpg'),
                      (feobj2.getPrewittHorizontalEdgeImg(), 'PrewittX' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittX' + str(count) + '.jpg'),
                      (feobj2.getPrewittVerticalEdgeImg(), 'PrewittY' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittY' + str(count) + '.jpg'),
                      (feobj2.getCombinedPrewittImg(), 'PrewittIMG' + str(count), 'results/testset/' + str(count) + '/' + 'PrewittIMG' + str(count) + '.jpg'),
                      (feobj3.getGaussianBlurredImage(), 'gblurimg' + str(count), 'results/testset/' + str(count) + '/' + 'gblurimg' + str(count) + '.jpg'),
                      (feobj3.getSelectedContourImg(), 'slccntimg' + str(count), 'results/testset/' + str(count) + '/' + 'slccntimg' + str(count) + '.jpg'),
                      (feobj3.getBoundingRectImg(), 'bndrectimg' + str(count), 'results/testset/' + str(count) + '/' + 'bndrectimg' + str(count) + '.jpg'),
                      (feobj3.getBoundedCircImg(), 'bndcircimg' + str(count), 'results/testset/' + str(count) + '/' + 'bndcircimg' + str(count) + '.jpg')])
        featarr = np.empty(0, dtype=float, order='C')
        featarr = np.insert(featarr, featarr.size, feobj.getAngularSecondMomentASM(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getEnergy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getEntropy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getHomogeneity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDm(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getCorrelation(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getHarCorrelation(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getClusterShade(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getClusterProminence(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment1(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment2(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment3(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment4(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDasm(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDmean(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDentropy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getCoarseness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getKurtosis(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getLineLikeness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getDirectionality(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getRegularity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getRoughness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getAsymmetryIndex(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getCompactIndex(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getFractalDimension(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getDiameter(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getColorVariance(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsCoarseness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsBusyness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsComplexity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsStrength(), 0)
        dset = np.insert(dset, dset.size, (featarr, str(input('Enter your result : \n'))), 0)
        count = count + 1
        if(str(input('Do you want to enter more images?? \n')) == 'y'):
            continue
        else:
            break
    np.savez('testcase', dset=dset, featnames=featnames)

def __generateFeatSetWImgs():
    dset = np.empty(0, dtype=np.dtype([('featureset', float, (34,)), ('result', object)]), order='C')
    featnames = np.array(['ASM', 'ENERGY', 'ENTROPY', 'CONTRAST', 'HOMOGENEITY', 'DM', 'CORRELATION', 'HAR-CORRELATION', 'CLUSTER-SHADE', 'CLUSTER-PROMINENCE', 'MOMENT-1', 'MOMENT-2', 'MOMENT-3', 'MOMENT-4', 'DASM', 'DMEAN', 'DENTROPY', 'TAM-COARSENESS', 'TAM-CONTRAST', 'TAM-KURTOSIS', 'TAM-LINELIKENESS', 'TAM-DIRECTIONALITY', 'TAM-REGULARITY', 'TAM-ROUGHNESS', 'ASYMMETRY-INDEX', 'COMPACT-INDEX', 'FRACTAL-DIMENSION', 'DIAMETER', 'COLOR-VARIANCE', 'KINGS-COARSENESS', 'KINGS-CONTRAST', 'KINGS-BUSYNESS', 'KINGS-COMPLEXITY', 'KINGS-STRENGTH'], dtype=object, order='C')
    try:
        shutil.rmtree('results/op')
        print("Found previous output folder.")
        print("Removed previous output folder")
    except OSError as e:
        print("No output folder found")
    os.makedirs('results/op')
    print("Created new output folder")
    print("Generating Feature Set for the uploaded input Image")
    while(True):
        obj = p.Prep('ip/' + "0.jpg")
        feobj = har.HarFeat(obj.getSegGrayImg())
        feobj2 = tam.TamFeat(obj.getSegGrayImg())
        feobj3 = g.Gabor(obj.getSegGrayImg(), obj.getSegColImg())
        feobj4 = k.KingFeat(obj.getSegGrayImg())
        __showImages([(obj.getActImg(), 'imgcol', 'results/op/'+ 'imgcol' + '.jpg'),
                        (obj.getGrayImg(), 'imggray', 'results/op/'+ 'imggray' + '.jpg'),
                        (obj.getInvrtGrayImg(), 'imggrayinvrt', 'results/op/'+ 'imggrayinvrt' + '.jpg'),
                        (obj.getBinaryImg(), 'imgbin', 'results/op/'+ 'imgbin' + '.jpg'),
                        (obj.getSegColImg(), 'segimgcol', 'results/op/'+ 'segimgcol' + '.jpg'),
                        (obj.getSegGrayImg(), 'segimggray', 'results/op/'+ 'segimggray' + '.jpg'),
                        (feobj2.getPrewittHorizontalEdgeImg(), 'PrewittX', 'results/op/'+ 'PrewittX' + '.jpg'),
                        (feobj2.getPrewittVerticalEdgeImg(), 'PrewittY', 'results/op/'+ 'PrewittY' + '.jpg'),
                        (feobj2.getCombinedPrewittImg(), 'PrewittIMG', 'results/op/'+ 'PrewittIMG' + '.jpg'),
                        (feobj3.getGaussianBlurredImage(), 'gblurimg', 'results/op/'+ 'gblurimg' + '.jpg'),
                        (feobj3.getSelectedContourImg(), 'slccntimg', 'results/op/'+ 'slccntimg' + '.jpg'),
                        (feobj3.getBoundingRectImg(), 'bndrectimg', 'results/op/'+ 'bndrectimg' + '.jpg'),
                        (feobj3.getBoundedCircImg(), 'bndcircimg', 'results/op/'+ 'bndcircimg' + '.jpg')])
        print("Generating features numpy array")
        featarr = np.empty(0, dtype=float, order='C')
        featarr = np.insert(featarr, featarr.size, feobj.getAngularSecondMomentASM(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getEnergy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getEntropy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getHomogeneity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDm(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getCorrelation(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getHarCorrelation(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getClusterShade(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getClusterProminence(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment1(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment2(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment3(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getMoment4(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDasm(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDmean(), 0)
        featarr = np.insert(featarr, featarr.size, feobj.getDentropy(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getCoarseness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getKurtosis(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getLineLikeness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getDirectionality(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getRegularity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj2.getRoughness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getAsymmetryIndex(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getCompactIndex(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getFractalDimension(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getDiameter(), 0)
        featarr = np.insert(featarr, featarr.size, feobj3.getColorVariance(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsCoarseness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsContrast(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsBusyness(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsComplexity(), 0)
        featarr = np.insert(featarr, featarr.size, feobj4.getKingsStrength(), 0)
        dset = np.insert(dset, dset.size, (featarr, str("negative")), 0)        
        break;
    np.savez('finalcase', dset=dset, featnames=featnames)
    print("Generating Feature Set Complete")

def __convertTargetTypeToInt(arr):
    cvt_arr = np.zeros((arr.size,), int, 'C')
    for i in range(0, arr.size, 1):
        if (arr[i] == 'malignant'):
            cvt_arr[i] = 1
        elif (arr[i] == 'negative'):
            cvt_arr[i] = -1
        else:
            continue
    return cvt_arr

def __convertTargetTypeToStr(arr):
    cvt_arr = np.empty((arr.size,), object, 'C')
    for i in range(0, arr.size, 1):
        if (int(np.round(arr[i])) >= 1):
           cvt_arr[i] = 'malignant'
        elif (int(np.round(arr[i])) <= -1):
           cvt_arr[i] = 'negative'
        elif (int(np.round(arr[i])) == 0):
           cvt_arr[i] = 'benign'
        else:
            pass
    return cvt_arr

def __formatResult(arr):
    result = ""
    for i in range(0, arr.size, 1):
        if (int(np.round(arr[i])) >= 1):
           result = "malignant"
        elif (int(np.round(arr[i])) <= -1):
           result = "negative"
        elif (int(np.round(arr[i])) == 0):
           result = "benign"
        else:
            pass
    return result

def __predictFromSavedTestCase():
    clasfobj = CLF.Classifiers(path='mlmodels/')
    dset = (np.load('testcase.npz'))['dset']
    print(dset['result'])
    print("\n")
    print("Now predicting results : \n \n")
    pred_res = clasfobj.predicto(dset['featureset'], __convertTargetTypeToInt(dset['result']))
    return pred_res

def __predictFromGenFeatSet():
    clasfobj = CLF.Classifiers(path='mlmodels/')
    dset = (np.load('finalcase.npz'))['dset']
    print("Now predicting results for the input image")
    pred_res = clasfobj.predictForSingleImage(dset['featureset'])
    print("Succesully predicted results")
    return pred_res

def __printPredResWithProperFormatting(predres, type=None):
    if (type == 'SVM'):
        print("FOR SVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVM'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['SVM'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['SVM'])['Accuracy'] * 100) + "\n")
    elif (type == 'RFC'):
        print("FOR RFC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFC'])['Prediction Results'])) + " \n Prediction results (raw) : " + str((predres['RFC'])['Prediction Results']) + " \n Prediction Accuracy : " + str((predres['RFC'])['Accuracy'] * 100) + "\n")

# def __showGLCM(feobj):
#     print(feobj.getGLCM())

# def __showHaralickFeatures(feobj):
#     print("->.->.->.->.->.->.->.->.->.HARALICK TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
#     __showGLCM(feobj)
#     print("\n")
#     print("Angular Second Moment-ASM of seg gray img %f \n" % feobj.getAngularSecondMomentASM())
#     print("Energy of seg gray img %f \n" % feobj.getEnergy())
#     print("Entropy of seg gray img %f \n" % feobj.getEntropy())
#     print("Contrast of seg gray img %f \n" % feobj.getContrast())
#     print("Homogeneity of seg gray img %f \n" % feobj.getHomogeneity())
#     print("Directional-Moment of seg gray img %f \n" % feobj.getDm())
#     print("Correlation of seg gray img %f \n" % feobj.getCorrelation())
#     print("Haralick-Correlation of seg gray img %f \n" % feobj.getHarCorrelation())
#     print("Cluster-Shade of seg gray img %f \n" % feobj.getClusterShade())
#     print("Cluster-Prominence of seg gray img %f \n" % feobj.getClusterProminence())
#     print("Moment1 of seg gray img %f \n" % feobj.getMoment1())
#     print("Moment2 of seg gray img %f \n" % feobj.getMoment2())
#     print("Moment3 of seg gray img %f \n" % feobj.getMoment3())
#     print("Moment4 of seg gray img %f \n" % feobj.getMoment4())
#     print("Differential-ASM of seg gray img %f \n" % feobj.getDasm())
#     print("Differential-Mean of seg gray img %f \n" % feobj.getDmean())
#     print("Differential-Entropy of seg gray img %f \n" % feobj.getDentropy())

# def __showTamuraFeatures(feobj2):
#     print("->.->.->.->.->.->.->.->.->.TAMURA TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
#     print("Tamura-Coarseness of seg gray img %f \n" % feobj2.getCoarseness())
#     print("Tamura-Contrast of seg gray img %f \n" % feobj2.getContrast())
#     print("Tamura-Kurtosis of seg gray img %f \n" % feobj2.getKurtosis())
#     print("Tamura-LineLikeness of seg gray img %f \n" % feobj2.getLineLikeness())
#     print("Tamura-Directionality of seg gray img %f \n" % feobj2.getDirectionality())
#     print("Tamura-Regularity of seg gray img %f \n" % feobj2.getRegularity())
#     print("Tamura-Roughness of seg gray img %f \n" % feobj2.getRoughness())

# def __showKingsFeatures(feobj4):
#     print("->.->.->.->.->.->.->.->.->.KING'S TEXTURE FEATURES.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
#     print("\n")
#     print(feobj4.getNGTDM())
#     print("\n")
#     print("King's-Coarseness of seg gray img %f \n" % feobj4.getKingsCoarseness())
#     print("King's-Contrast of seg gray img %f \n" % feobj4.getKingsContrast())
#     print("King's-Busyness of seg gray img %f \n" % feobj4.getKingsBusyness())
#     print("King's-Complexity of seg gray img %f \n" % feobj4.getKingsComplexity())
#     print("King's-Strength of seg gray img %f \n" % feobj4.getKingsStrength())

# def __showGaborPhysicalFeatures(feobj3):
#     print("->.->.->.->.->.->.->.->.->.GABOR PHYSICAL FEATURES OF LESION.<-.<-.<-.<-.<-.<-.<-.<-.<- \n")
#     print("List of Contour-Points ::: \n")
#     print(feobj3.getListOfContourPoints())
#     print("\n")
#     print("Hierarchy of extracted contours ::: \n")
#     print(feobj3.getHierarchyOfContours())
#     print("\n")
#     print("List of moments for corresponding-contours ::: \n")
#     print(feobj3.getListOfMomentsForCorrespondingContours())
#     print("\n")
#     print("List of centroids for corresponding-contours ::: \n")
#     print(feobj3.getListOfCentroidsForCorrespondingContours())
#     print("\n")
#     print("List of areas for corresponding-contours ::: \n")
#     print(feobj3.getListOfAreasForCorrespondingContours())
#     print("\n")
#     print("List of perimeters for corresponding-contours ::: \n")
#     print(feobj3.getListOfPerimetersForCorrespondingContours())
#     print("\n")
#     print("Mean_Edge of covering rectangle for lesion img %f \n" % feobj3.getMeanEdgeOfCoveringRect())
#     print("Bounded_Circle radius %f \n" % feobj3.getBoundedCircRadius())
#     print("Asymmetry-Index of lesion %f \n" % feobj3.getAsymmetryIndex())
#     print("Compact-Index of lesion %f \n" % feobj3.getCompactIndex())
#     print("Fractal-Dimension of lesion %f \n" % feobj3.getFractalDimension())
#     print("Diameter of lesion %f \n" % feobj3.getDiameter())
#     print("Color-Variance of lesion %f \n" % feobj3.getColorVariance())

def predictType():
    t = time.time()
    __generateFeatSetWImgs()
    predres = __predictFromGenFeatSet()
    print("Prediction time : %f secs" % (time.time() - t))
    __moveImagesToStatic()           
    return (str(__formatResult((predres['RFC'])['Prediction Results'])))

def __moveImagesToStatic():
    print("Moving images to static folder")
    try:
        shutil.rmtree('app/static')
        print("Found previous static folder.")
        print("Removed previous static folder")
    except OSError as e:
        print("No output folder found")
    os.makedirs('app/static')
    shutil.copy('results/op/segimgcol.jpg','app/static/1.jpg')
    shutil.copy('ip/0.jpg','app/static/0.jpg')
    print("Moving Complete")

def main_menu():
    print("_______WELCOME TO THE MELANOMA-PREDICTION PROGRAM_______ \n")
    print("\t CRH. \n")

    while (True):
       print("\t Available options are given below : \n")
       print("\t 1.Create \'training-dataset\' from the images of known ->MELANOMA<- types!! \n")
       print("\t 2.Train classifiers and regressors on the created \'training-dataset\'!! \n")
       print("\t 3.Create \'testing-dataset\' from the supervised images in temp folder!! \n")
       print("\t 4.Predict results and calculate Accuracy from the \'testcase.npz\' numpy file!! \n")
       print("\t 5.Predict an image from the \'ip folder\'!! \n")
       print("\t Enter \'e\' to exit!! \n")
       c = str(input("Enter your choice - \n"))
       if (c == '1'):
           print("If you see a \'results\' folder in the root directory of the project, delete the \'dataset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your corresponding images in the \'images\' folder under the \'malignant\', \'benign\' or \'negative\' directories. \n")
           print("The image file-names must be numeric starting from 0 in sequence under each category folder. \n")
           print("Eg. - 0.jpg, 1.jpg, 2.jpg, ..... etc \n")
           print("You must provide images under each category!!! \n")
           __createDataSet("malignant", 30)
           __createDataSet("benign", 30)
           __createDataSet("negative", 10)
           print("\'Training-Dataset\' successfully generated!! \n")
           print("This dataset consists of the features-array of the corresponding images and their classified types. \n")
           print("All results are stored in the numpy file \'dataset.npz\'. \n")
           print("Total training-images count : %d \n" % imgcount)
       elif (c == '2'):
           print("Now we'll train our various classifiers and regressors on the training-data stored in the \'dataset.npz\' numpy file. \n")
           print("All machine-learning models will be saved as individual \'.pkl\' files in the \'mlmodels\' python-package. \n")
           __createAndTrainMlModels()
           print("Training is now complete!! \n")
       elif (c == '3'):
           print("If you see a \'results\' folder in the root directory of the project, delete the \'testset\' folder in it. \n")
           print("Now, before you proceed, just make sure that you have your test-images in the \'temp\' folder. \n")
           print("If you haven't already made the directories, please make them and place the test-images in them. \n")
           input("Just press any key when your are ready : \n")
           __getTestImages()
           print("\'Testing-Dataset\' successfully generated!! \n")
           print("This dataset consists of the features-array of the test-images and their supervised-classified types. \n")
           print("All results are stored in the numpy file \'testset.npz\' \n")
       elif (c == '4'):
           print("This will predict results from \'testcase.npz\' and also calculate the prediction accuracy of the individual models. \n")
           pred_res = __predictFromSavedTestCase()
           print("Before we start, here is the reference legend __ \n")
           print("\'1\' : MALIGNANT. \n")
           print("\'0\' : BENIGN. \n")
           print("\'-1\' : NEGATIVE. \n")
           while (True):
              type = str(input('Select Classifier/Regressor acronym : \n'))
              if (type in pred_res):
                    __printPredResWithProperFormatting(pred_res, type)
              else:
                  __printPredResWithProperFormatting(pred_res)
                  break
       elif (c == '5'):
           __generateFeatSetWImgs()           
           predres = __predictFromGenFeatSet()
           print("Here is the reference legend __ \n")
           print("Using SVM Classifier") 
           print("FOR SVM - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['SVM'])['Prediction Results'])) + "\n")
           print("FOR RFC - \n Prediction results (String) : " + str(__convertTargetTypeToStr((predres['RFC'])['Prediction Results'])) + "\n")              
       else:
           print("Thank-You For Using This Program!!!")
           print("Now Exiting.")
           break

if __name__ == '__main__':
    main_menu()

#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    t = np.argmin(np.abs(1.-tpr-fpr))
    
    return (fpr[t] + 1 - tpr[t])/2


def pd_at_far(y_true, y_score, fpr_th):
    '''
    Returns the Pd at fixed FAR for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(fpr_th, fpr, tpr)


def macc(y_true, y_score):
    '''
    Returns the maximum accuracy for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.max(tpr+1-fpr)/2

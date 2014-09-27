package edu.neu.ccs.pyramid.multilabel_classification.hmlgb;

import edu.neu.ccs.pyramid.regression.Regressor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 9/27/14.
 */
public class HMLGradientBoosting implements Serializable{
    private static final long serialVersionUID = 1L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    private transient HMLGBTrainer trainer;

    public HMLGradientBoosting(int numClasses) {
        this.numClasses = numClasses;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
    }
}

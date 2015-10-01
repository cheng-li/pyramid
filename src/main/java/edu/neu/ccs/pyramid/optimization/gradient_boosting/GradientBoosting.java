package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/1/15.
 */
public class GradientBoosting {
    protected int numEnsembles;
    protected List<Ensemble> ensembles;

    public GradientBoosting(int numEnsembles) {
        this.numEnsembles = numEnsembles;
        this.ensembles = new ArrayList<>();
        for (int k=0;k<numEnsembles;k++){
            ensembles.add(new Ensemble());
        }
    }
    
}

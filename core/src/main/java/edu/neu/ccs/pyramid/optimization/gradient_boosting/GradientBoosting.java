package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/1/15.
 */
public class GradientBoosting implements Serializable{
    private static final long serialVersionUID = 1L;
    protected int numEnsembles;
    protected List<Ensemble> ensembles;
    protected FeatureList featureList;

    
    public GradientBoosting(int numEnsembles) {
        this.numEnsembles = numEnsembles;
        this.ensembles = new ArrayList<>();
        for (int k=0;k<numEnsembles;k++){
            ensembles.add(new Ensemble());
        }
    }

    public int getNumEnsembles() {
        return numEnsembles;
    }

    public Ensemble getEnsemble(int ensembleIndex){
        return ensembles.get(ensembleIndex);
    }

    public double score(Vector vector, int ensembleIndex){
        return ensembles.get(ensembleIndex).score(vector);
    }


    public double[] scores(Vector vector){
        double[] scores = new double[numEnsembles];
        for (int k=0;k<numEnsembles;k++){
            scores[k] = score(vector,k);
        }
        return scores;
    }

    public FeatureList getFeatureList() {
        return featureList;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("GradientBoosting{");
        sb.append("numEnsembles=").append(numEnsembles);
        sb.append(", ensembles=").append(ensembles);
        sb.append('}');
        return sb.toString();
    }
}

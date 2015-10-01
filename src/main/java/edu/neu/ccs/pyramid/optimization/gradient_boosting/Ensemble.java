package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * An ensemble is a scorning function
 * Created by chengli on 10/1/15.
 */
public class Ensemble implements Serializable{
    private static final long serialVersionUID = 1L;

    private List<Regressor> regressors;

    public Ensemble() {
        this.regressors = new ArrayList<>();
    }

    public void add(Regressor regressor){
        this.regressors.add(regressor);
    }

    public Regressor get(int index){
        return this.regressors.get(index);
    }


    public double score(Vector vector) {
        double res = 0;
        for (Regressor regressor: regressors){
            res += regressor.predict(vector);
        }
        return res;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Ensemble{");
        sb.append("regressors=").append(regressors);
        sb.append('}');
        return sb.toString();
    }
}

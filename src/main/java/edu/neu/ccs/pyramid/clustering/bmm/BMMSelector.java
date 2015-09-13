package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;


/**
 * select the best BMM from multiple random starts
 * Created by chengli on 9/12/15.
 */
public class BMMSelector {

    public static BMM select(DataSet dataSet,int numClusters, int numRuns) {
        BMM best = null;
        double bestObjective = Double.NEGATIVE_INFINITY;
        for (int i=0;i<numRuns;i++){
            BMMTrainer trainer = new BMMTrainer(dataSet,numClusters);
            BMM bmm = trainer.train();
            double objective = trainer.terminator.getLastValue();
            if (objective>bestObjective){
                bestObjective = objective;
                best = bmm;
            }
        }
        return best;
    }
}

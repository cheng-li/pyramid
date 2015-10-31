package edu.neu.ccs.pyramid.clustering.bmm;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;

import java.util.Arrays;


/**
 * select the best BMM from multiple random starts
 * Created by chengli on 9/12/15.
 */
public class BMMSelector {

    public static BMM select(DataSet dataSet,int numClusters, int numRuns) {
        BMM best = null;
        double bestObjective = Double.POSITIVE_INFINITY;
        for (int i=0;i<numRuns;i++){
            BMMTrainer trainer = new BMMTrainer(dataSet,numClusters);
            BMM bmm = trainer.train();
            double objective = trainer.terminator.getLastValue();
            if (objective < bestObjective){
                bestObjective = objective;
                best = bmm;
            }
        }
        return best;
    }

    public static BMMTrainer selectTrainer(DataSet dataSet,int numClusters, int numRuns) {
        BMMTrainer best = null;
        double bestObjective = Double.POSITIVE_INFINITY;
        for (int i=0;i<numRuns;i++){
            BMMTrainer trainer = new BMMTrainer(dataSet,numClusters);
            BMM bmm = trainer.train();
            double objective = trainer.terminator.getLastValue();
            if (objective < bestObjective){
                bestObjective = objective;
                best = trainer;
            }
        }
        return best;
    }


    public static double[][] selectGammas(int numClasses, MultiLabel[] multiLabels, int numClusters) {
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(multiLabels.length)
                .numFeatures(numClasses)
                .build();
        for (int i=0;i<multiLabels.length;i++){
            MultiLabel multiLabel = multiLabels[i];
            for (int label: multiLabel.getMatchedLabels()){
                dataSet.setFeatureValue(i,label,1);
            }
        }
        BMMTrainer trainer = BMMSelector.selectTrainer(dataSet, numClusters, 10);
//        System.out.println("bmm = "+trainer.bmm);
//        System.out.println("gamma = "+ Arrays.deepToString(trainer.gammas));
        return trainer.gammas;
    }


}

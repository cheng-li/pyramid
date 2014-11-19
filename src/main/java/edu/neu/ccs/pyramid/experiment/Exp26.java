package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;

import java.io.File;

/**
 * gradient boosting hyper parameter tuning, single label
 * Created by chengli on 11/18/14.
 */
public class Exp26 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        for (int k=0;k<config.getInt("numTrials");k++){
            oneTrial(config);
        }
    }



    static void oneTrial(Config config) throws Exception{
        Config hyperParams = genHyperParams(config);
        System.out.println("=================");
        System.out.println("hyper parameters for the trial:");
        System.out.println(hyperParams);

        int numIterations = hyperParams.getInt("numIterations");
        int numLeaves = hyperParams.getInt("numLeaves");
        double learningRate = hyperParams.getDouble("learningRate");
        int minDataPerLeaf = hyperParams.getInt("minDataPerLeaf");

        Pair<ClfDataSet, ClfDataSet> dataSets = loadDataSets(config);
        ClfDataSet trainingSet = dataSets.getFirst();
        ClfDataSet validationSet = dataSets.getSecond();
        int numClasses = trainingSet.getNumClasses();

        LKTBConfig trainConfig = new LKTBConfig.Builder(trainingSet)
                .learningRate(learningRate).minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves).dataSamplingRate(1)
                .featureSamplingRate(1).build();
        LKTreeBoost lkTreeBoost = new LKTreeBoost(numClasses);
        lkTreeBoost.setPriorProbs(trainingSet);
        lkTreeBoost.setTrainConfig(trainConfig);
        for (int i=0;i<numIterations;i++){
            lkTreeBoost.boostOneRound();
        }


        System.out.println("accuracy on training set = "+ Accuracy.accuracy(lkTreeBoost,
                trainingSet));
        System.out.println("accuracy on validation set = "+ Accuracy.accuracy(lkTreeBoost,
                validationSet));

    }

    private static Pair<ClfDataSet, ClfDataSet> loadDataSets(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        ClfDataSet clfDataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE, false);
        return DataSetUtil.splitToTrainValidation(clfDataSet, config.getDouble("trainPercentage"));

    }


    private static Config genHyperParams(Config config){
        Config hyperParams = new Config();
        hyperParams.setInt("numIterations", Sampling.intUniform(config.getIntegers("numIterations").get(0),
                config.getIntegers("numIterations").get(1)));
        hyperParams.setInt("numLeaves", Sampling.intUniform(config.getIntegers("numLeaves").get(0),
                config.getIntegers("numLeaves").get(1)));
        hyperParams.setDouble("learningRate", Sampling.doubleLogUniform(config.getDoubles("learningRate").get(0),
                config.getDoubles("learningRate").get(1)));
        hyperParams.setInt("minDataPerLeaf", Sampling.intUniform(config.getIntegers("minDataPerLeaf").get(0),
                config.getIntegers("minDataPerLeaf").get(1)));
        return hyperParams;
    }
}

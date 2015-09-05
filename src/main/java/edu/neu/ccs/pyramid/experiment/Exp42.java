package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticTrainer;
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
 * multi-label logistic regression hyper parameter tuning via cross-validation
 * todo not finished
 * Created by chengli on 12/23/14.
 */
public class Exp42 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        run(config);


    }

    private static Config genHyperParams(Config config){
        Config hyperParams = new Config();
        double gaussianPriorVariance = Sampling.doubleLogUniform(config.getDoubles("gaussianPriorVariance").get(0),
                config.getDoubles("gaussianPriorVariance").get(1));
        hyperParams.setDouble("gaussianPriorVariance", gaussianPriorVariance);

        return hyperParams;

    }

    private static void run(Config config) throws Exception{
        Config bestParams;
        double bestPerformance = Double.NEGATIVE_INFINITY;

        for (int trial=0;trial<config.getInt("numTrials");trial++){
            Config hyperParams = genHyperParams(config);
            System.out.println("==============================");
            System.out.println("hyper parameters for the trial:");
            System.out.println(hyperParams);
            double aveValidAcc= 0;
            for (int validation=0;validation<config.getInt("numValidations");validation++){
                System.out.println("validation "+validation);
                Pair<ClfDataSet,ClfDataSet> dataSets = loadDataSets(config);

                ClfDataSet trainSet =dataSets.getFirst();

                ClfDataSet validationSet = dataSets.getSecond();

                RidgeLogisticTrainer trainer = RidgeLogisticTrainer.getBuilder()
                        .setHistory(5)
                        .setGaussianPriorVariance(hyperParams.getDouble("gaussianPriorVariance"))
                        .build();

                Classifier classifier = trainer.train(trainSet);

                System.out.println("accuracy on the training set = "+ Accuracy.accuracy(classifier, trainSet));

                double validationAcc = Accuracy.accuracy(classifier, validationSet);
                System.out.println("accuracy on the validation set = "+validationAcc);
                aveValidAcc += validationAcc;
            }

            aveValidAcc /= config.getInt("numValidations");

            System.out.println("average accuracy on the validation set = "+ aveValidAcc);
            if (aveValidAcc>bestPerformance){
                bestPerformance = aveValidAcc;
                bestParams = hyperParams;
                System.out.println("**************************************");
                System.out.println("best performance got so far: "+bestPerformance);
                System.out.println("best hyper parameters got so far: ");
                System.out.println(bestParams);
                System.out.println("**************************************");
            }
        }
    }

    private static Pair<ClfDataSet, ClfDataSet> loadDataSets(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        ClfDataSet clfDataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE, false);
        return DataSetUtil.splitToTrainValidation(clfDataSet, config.getDouble("trainPercentage"));

    }
}

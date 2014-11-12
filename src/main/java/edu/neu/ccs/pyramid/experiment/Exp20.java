package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.glm.LassoLearner;

import java.io.File;

/**
 * Lasso logistic regression for
 * Created by chengli on 11/11/14.
 */
public class Exp20 {
    public static void main(String[] args) {
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

    }

    private static void train(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        Instances trainSet = InstancesReader.read(null,trainFile);
        LassoLearner learner = new LassoLearner();
        learner.setTask(Learner.Task.CLASSIFICATION);
        learner.setMaxNumIters(config.getInt("train.numIterations"));
        learner.setVerbose(false);
        learner.setLambda(config.getDouble("train.lambda"));

    }
}

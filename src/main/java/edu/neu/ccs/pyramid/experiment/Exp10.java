package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.ClassifierFactory;
import edu.neu.ccs.pyramid.classification.TrainConfig;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBFactory;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainConfig;
import edu.neu.ccs.pyramid.classification.ecoc.CodeMatrix;
import edu.neu.ccs.pyramid.classification.ecoc.ECOC;
import edu.neu.ccs.pyramid.classification.ecoc.ECOCConfig;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.File;

/**
 * ecoc
 * Created by chengli on 10/5/14.
 */
public class Exp10 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        train(config);
        test(config);
    }

    private static void train(Config config) throws Exception{
        String folder = config.getString("input.folder");
        File data = new File(folder,config.getString("input.trainData"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(data, DataSetType.CLF_DENSE,false);
        ClassifierFactory classifierFactory = new LKTBFactory();
        TrainConfig trainConfig = new LKTBTrainConfig()
                .setNumLeaves(config.getInt("numLeaves"))
                .setLearningRate(0.1)
                .setNumIterations(config.getInt("numIterations"));
        ECOCConfig ecocConfig = new ECOCConfig().setCodeType(CodeMatrix.CodeType.RANDOM)
                .setNumFunctions(config.getInt("numFunctions"));
        ECOC ecoc = new ECOC(ecocConfig,
                dataSet,
                new File(config.getString("archive"),"ecoc/models").getAbsolutePath(),
                classifierFactory,
                trainConfig);
        System.out.println(ecoc.getCodeMatrix().toString());
        ecoc.train();
        ecoc.serialize(new File(config.getString("archive"),"ecoc/ecoc.ser"));
    }

    private static void test(Config config) throws Exception{
        String folder = config.getString("input.folder");
        File data = new File(folder,config.getString("input.testData"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(data, DataSetType.CLF_DENSE,false);
        ECOC ecoc = ECOC.deserialize(new File(config.getString("archive"),"ecoc/ecoc.ser"));
        System.out.println(Accuracy.accuracy(ecoc,dataSet));
    }

}

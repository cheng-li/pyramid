package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CRFLoss;
import edu.neu.ccs.pyramid.multilabel_classification.crf.InstanceF1Predictor;
import edu.neu.ccs.pyramid.multilabel_classification.crf.SubsetAccPredictor;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.concurrent.TimeUnit;

import static edu.stanford.nlp.util.logging.RedwoodConfiguration.Handlers.output;

/**
 * pair-wise CRF for multi-label classification
 * Created by chengli on 8/19/16.
 */
public class App6 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }

    }

    private static void train(Config config) throws Exception{
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        CMLCRF cmlcrf = new CMLCRF(trainSet);
        double gaussianVariance = config.getDouble("train.gaussianVariance");
        cmlcrf.setConsiderPair(config.getBoolean("train.considerLabelPair"));
        CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);
        crfLoss.setParallelism(true);
        crfLoss.setRegularizeAll(config.getBoolean("train.regularizeAll"));
        LBFGS optimizer = new LBFGS(crfLoss);

        PluginPredictor<CMLCRF> predictor = null;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                predictor = new SubsetAccPredictor(cmlcrf);
                break;
            case "instanceFMeasure":
                predictor = new InstanceF1Predictor(cmlcrf);
                break;
            default:
                throw new IllegalArgumentException("predict.target must be subsetAccuracy or instanceFMeasure");
        }

        int progressInterval = config.getInt("train.showProgress.interval");
        int iteration = 0;
        while(true){
            optimizer.iterate();
            iteration += 1;

            if (iteration%progressInterval==0){
                System.out.println("iteration "+iteration);
                System.out.println("training objective = "+optimizer.getTerminator().getLastValue());
                System.out.println("training performance:");
                System.out.println(new MLMeasures(predictor,trainSet));
            }

            if (optimizer.getTerminator().shouldTerminate()){
                System.out.println("iteration "+iteration);
                System.out.println("training objective = "+optimizer.getTerminator().getLastValue());
                System.out.println("training performance:");
                System.out.println(new MLMeasures(predictor,trainSet));
                System.out.println("training done!");
                break;
            }
        }

        String modelName = "model_crf";
        String output = config.getString("output.folder");
        (new File(output)).mkdirs();
        File serializeModel = new File(output, modelName);
        cmlcrf.serialize(serializeModel);

    }


    private static void test(Config config) throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        String modelName = "model_crf";
        String output = config.getString("output.folder");
        CMLCRF cmlcrf = (CMLCRF)Serialization.deserialize(new File(output, modelName));
        PluginPredictor<CMLCRF> predictor = null;
        String predictTarget = config.getString("predict.target");
        switch (predictTarget){
            case "subsetAccuracy":
                predictor = new SubsetAccPredictor(cmlcrf);
                break;
            case "instanceFMeasure":
                predictor = new InstanceF1Predictor(cmlcrf);
                break;
            default:
                throw new IllegalArgumentException("predict.target must be subsetAccuracy or instanceFMeasure");
        }
        System.out.println("test performance:");
        System.out.println(new MLMeasures(predictor,testSet));

    }
}

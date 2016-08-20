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
import edu.neu.ccs.pyramid.multilabel_classification.crf.CMLCRF;
import edu.neu.ccs.pyramid.multilabel_classification.crf.CRFLoss;
import edu.neu.ccs.pyramid.multilabel_classification.crf.InstanceF1Predictor;
import edu.neu.ccs.pyramid.optimization.LBFGS;

import java.io.File;
import java.util.concurrent.TimeUnit;

import static edu.stanford.nlp.util.logging.RedwoodConfiguration.Handlers.output;

/**
 * pair-wise CRF
 * Created by chengli on 8/19/16.
 */
public class App6 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SEQ_SPARSE, true);


        String output = config.getString("output.folder");
        String modelName = "model_crf";



    }

    private static void train(MultiLabelClfDataSet trainSet, Config config) throws Exception{
        CMLCRF cmlcrf = new CMLCRF(trainSet);
        double gaussianVariance = config.getDouble("gaussianVariance");
        cmlcrf.setConsiderPair(config.getBoolean("considerLabelPair"));
        CRFLoss crfLoss = new CRFLoss(cmlcrf, trainSet, gaussianVariance);
        crfLoss.setParallelism(true);
        crfLoss.setRegularizeAll(config.getBoolean("regularizeAll"));
        LBFGS optimizer = new LBFGS(crfLoss);

        while(true){
            optimizer.iterate();
            System.out.println("training objective = "+optimizer.getTerminator().getLastValue());

            if (optimizer.getTerminator().shouldTerminate()){
                break;
            }
        }

        String modelName = "model_crf";
        String output = config.getString("output.folder");

        if (config.getBoolean("saveModel")) {
            (new File(output)).mkdirs();
            File serializeModel = new File(output, modelName);
            cmlcrf.serialize(serializeModel);
        }

    }
}

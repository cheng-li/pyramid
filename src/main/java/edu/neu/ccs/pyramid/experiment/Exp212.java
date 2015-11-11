package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;

import java.io.File;
import java.io.IOException;

/**
 * Created by Rainicy on 11/3/15.
 */
public class Exp212 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        // data to be reported.
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.data"),
                DataSetType.ML_CLF_SPARSE, true);

        // load model
        String output = config.getString("output");
        String modelName = config.getString("modelName");

        BMMClassifier bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));

        double softmaxVariance = config.getDouble("softmaxVariance");
        double logitVariance = config.getDouble("logitVariance");

        String reportsPath = config.getString("reportPath");
//        bmmClassifier.generateReports(dataSet, reportsPath, softmaxVariance, logitVariance, config);
    }
}

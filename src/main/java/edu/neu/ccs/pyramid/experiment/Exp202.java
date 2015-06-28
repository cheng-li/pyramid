package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.List;

/**
 * Created by Rainicy on 6/27/15.
 */
public class Exp202 {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the conifg file.");
        }

        Config config = new Config(args[0]);

        String path = config.getString("path");

        // deserialize
        IMLGradientBoosting imlGradientBoosting = (IMLGradientBoosting) Serialization.deserialize(new File(path,"model"));

        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "train"), DataSetType.ML_CLF_SPARSE ,true);
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(new File(new File(path, "data_sets"), "test"), DataSetType.ML_CLF_SPARSE ,true);

        List<MultiLabel> testPredict = imlGradientBoosting.predict(test);
        MultiLabel[] testTrue = test.getMultiLabels();

        LabelTranslator labelTranslator = test.getLabelTranslator();
        String careLabel = "CCAT";
        int labelIndex = labelTranslator.toIntLabel(careLabel);


        // get accuracy
        int count = 0;
        for (int i=0; i<testTrue.length; i++) {
            MultiLabel predicts = testPredict.get(i);
            MultiLabel labels = testTrue[i];

            if ( (predicts.matchClass(labelIndex) && labels.matchClass(labelIndex)) || (!predicts.matchClass(labelIndex) && !labels.matchClass(labelIndex)) ) {
                count++;
            }
        }

        double acc = (double) count / testTrue.length;

        System.out.println("Accuracy on CCAT: " + acc);


    }
}

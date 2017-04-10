package edu.neu.ccs.pyramid.multilabel_classification.cbm;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MLMeasures;

import java.io.File;

/**
 * Created by chengli on 4/9/17.
 */
public class LRRecoverCBMOptimizerTest  {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");


    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception {
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
                DataSetType.ML_CLF_DENSE, true);


        int numComponents = 50;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumComponents(numComponents)
                .setMultiClassClassifierType("lr")
                .setBinaryClassifierType("lr")
                .build();

        LRRecoverCBMOptimizer optimizer = new LRRecoverCBMOptimizer(cbm, dataSet);
        optimizer.setPriorVarianceBinary(0.1);
        optimizer.setPriorVarianceMultiClass(0.1);
        optimizer.setDropProb(0.1);
        optimizer.initialize();

        AccPredictor accPredictor = new AccPredictor(cbm);
        for (int iter=1;iter<=25;iter++){
            System.out.println("iter = "+iter);
            optimizer.iterate();
            optimizer.updateGroundTruth();
            System.out.println(new MLMeasures(accPredictor, testSet));
        }

    }
}
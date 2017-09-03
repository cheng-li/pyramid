package edu.neu.ccs.pyramid.multilabel_classification.cbm;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
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


        int numComponents = 10;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumComponents(numComponents)
                .setMultiClassClassifierType("lr")
                .setBinaryClassifierType("lr")
                .build();

        LRRecoverCBMOptimizer optimizer = new LRRecoverCBMOptimizer(cbm, dataSet);
        optimizer.setPriorVarianceBinary(1);
        optimizer.setPriorVarianceMultiClass(1);
//        optimizer.setDropProb(1000);
        optimizer.initialize();

        MultiLabel multiLabel1 = new MultiLabel();
        multiLabel1.addLabel(0);

        MultiLabel multiLabel2 = new MultiLabel();
        multiLabel2.addLabel(5);

        MultiLabel multiLabel3 = new MultiLabel();
        multiLabel3.addLabel(5);
        multiLabel3.addLabel(0);

        AccPredictor accPredictor = new AccPredictor(cbm);
        for (int iter=1;iter<=20;iter++){
            System.out.println("iter = "+iter);
            optimizer.iterate();
            System.out.println(accPredictor.predict(dataSet.getRow(14)));
            System.out.println("probability on 0 ="+cbm.predictLogAssignmentProb(dataSet.getRow(14),multiLabel1));
            System.out.println("probability on 5 ="+cbm.predictLogAssignmentProb(dataSet.getRow(14),multiLabel2));
            System.out.println("probability on 0 and 5 ="+cbm.predictLogAssignmentProb(dataSet.getRow(14),multiLabel3));
            if (iter>=1){
                optimizer.updateGroundTruth();
            }

            System.out.println(new MLMeasures(accPredictor, testSet));
        }

        for (int i=0;i<1000;i++){
//            System.out.println(accPredictor.predict(dataSet.getRow(i)));
            System.out.println("given="+dataSet.getMultiLabels()[i]+", corrected="+optimizer.groundTruth.getMultiLabels()[i]+", predicted="+accPredictor.predict(dataSet.getRow(i)));
        }

    }
}
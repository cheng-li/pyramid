package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import junit.framework.TestCase;

import java.io.File;

/**
 * Created by chengli on 3/5/17.
 */
public class SparseCBMOptimzerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/train"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "scene/test"),
                DataSetType.ML_CLF_DENSE, true);

        int numComponents = 20;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(dataSet.getNumClasses())
                .setNumFeatures(dataSet.getNumFeatures())
                .setNumComponents(numComponents)
                .setMultiClassClassifierType("lr")
                .setBinaryClassifierType("lr")
                .build();

        SparseCBMOptimzer optimzer = new SparseCBMOptimzer(cbm, dataSet);
        optimzer.initalizeGammaByBM();
        optimzer.updateMultiClassLR();
        optimzer.updateAllBinary();
//        System.out.println(new MLMeasures(cbm, dataSet));
        System.out.println("test");
        System.out.println(new MLMeasures(cbm, testSet));

        System.out.println("update gamma");
        optimzer.updateGamma();
        optimzer.updateMultiClassLR();
        optimzer.updateAllBinary();
//        System.out.println(new MLMeasures(cbm, dataSet));
        System.out.println("test");
        System.out.println(new MLMeasures(cbm, testSet));

        System.out.println("update gamma again");
        optimzer.updateGamma();
        optimzer.updateMultiClassLR();
        optimzer.updateAllBinary();
//        System.out.println(new MLMeasures(cbm, dataSet));
        System.out.println("test");
        System.out.println(new MLMeasures(cbm, testSet));
    }

}
package edu.neu.ccs.pyramid.core.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;

import java.io.File;

public class CBMInitializerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{

        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/flags/train"),
                DataSetType.ML_CLF_SPARSE, true);


        int numClusters = 2;
        double softmaxVariance = 1000;
        double logitVariance = 1000;
        CBM cbm = CBM.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumComponents(numClusters)
                .setBinaryClassifierType("lr")
                .setMultiClassClassifierType("lr")
                .build();

        CBMOptimizer optimizer = new CBMOptimizer(cbm,trainSet);
        CBMInitializer.initialize(cbm,trainSet,optimizer);
    }

}
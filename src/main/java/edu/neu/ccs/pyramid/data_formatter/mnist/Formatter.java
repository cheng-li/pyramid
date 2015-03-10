package edu.neu.ccs.pyramid.data_formatter.mnist;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 10/21/14.
 */
public class Formatter {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        dumpTrain();
        dumpTest();

    }

    private static void dumpTrain() throws Exception{
        File featureFile = new File(DATASETS,"mnist/training_image.txt");
        File labelFile = new File(DATASETS,"mnist/training_label.txt");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(10, featureFile, labelFile, ",", DataSetType.CLF_DENSE,false);
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<10;i++){
            extLabels.add(""+i);
        }

        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        dataSet.setLabelTranslator(labelTranslator);
        TRECFormat.save(dataSet, new File(TMP, "train.trec"));
    }

    private static void dumpTest() throws Exception{
        File featureFile = new File(DATASETS,"mnist/testing_image.txt");
        File labelFile = new File(DATASETS,"mnist/testing_label.txt");
        ClfDataSet dataSet = StandardFormat.loadClfDataSet(10, featureFile, labelFile, ",", DataSetType.CLF_DENSE,false);
        List<String> extLabels = new ArrayList<>();
        for (int i=0;i<10;i++){
            extLabels.add(""+i);
        }

        LabelTranslator labelTranslator = new LabelTranslator(extLabels);
        dataSet.setLabelTranslator(labelTranslator);
        TRECFormat.save(dataSet, new File(TMP, "test.trec"));
    }
}

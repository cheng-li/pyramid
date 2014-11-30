package edu.neu.ccs.pyramid.data_formatter.spam;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created by chengli on 9/4/14.
 */
public class Formatter {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        saveTrainData();
//        saveTestData();
        saveTrainZScore();
        saveTestZScore();
    }

    private static void saveTestZScore() throws IOException {
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "spam/test_data_nomal.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};

        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        DataSetUtil.setLabelTranslator(data, labelTranslator);
        TRECFormat.save(data, new File(TMP, "test.trec"));
    }

    private static void saveTrainZScore() throws IOException {
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "spam/train_data_nomal.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        DataSetUtil.setLabelTranslator(data, labelTranslator);
        TRECFormat.save(data, new File(TMP, "train.trec"));
    }

    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader(new File(DATASETS, "spam/feature_names.txt")))
        ){
            String line;
            while((line = br.readLine())!=null){
                String name = line.split(Pattern.quote(":"))[0];
                names.add(name);
            }
        }
        return names;
    }

    static void saveTrainData()throws Exception{
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        DataSetUtil.setLabelTranslator(data, labelTranslator);
        TRECFormat.save(data, new File(TMP, "train.trec"));
    }

    static void saveTestData()throws Exception{
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};

        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        DataSetUtil.setLabelTranslator(data, labelTranslator);
        TRECFormat.save(data, new File(TMP, "test.trec"));
    }
}

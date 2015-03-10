package edu.neu.ccs.pyramid.data_formatter.faculty;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created by Rainicy on 12/6/14.
*/
public class Formatter {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        saveTrainData();
        saveTestData();
    }

    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader(new File(DATASETS, "faculty/feature_names.txt")))
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
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "faculty/train_data.txt"),
                new File(DATASETS, "faculty/train_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-influence","influence"};
        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        data.setLabelTranslator(labelTranslator);
        TRECFormat.save(data, new File(TMP, "train.trec"));
    }

    static void saveTestData()throws Exception{
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet(2,new File(DATASETS, "faculty/test_data.txt"),
                new File(DATASETS, "faculty/test_label.txt"), ",", DataSetType.CLF_DENSE,false);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-influence","influence"};

        LabelTranslator labelTranslator = new LabelTranslator(extLabels);

        data.setLabelTranslator(labelTranslator);
        TRECFormat.save(data, new File(TMP, "test.trec"));
    }
}


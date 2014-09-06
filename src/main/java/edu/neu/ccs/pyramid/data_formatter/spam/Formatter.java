package edu.neu.ccs.pyramid.data_formatter.spam;

import edu.neu.ccs.pyramid.dataset.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created by chengli on 9/4/14.
 */
public class Formatter {
    public static void main(String[] args) throws Exception{
        saveTrainData();
        saveTestData();

    }

    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader("/Users/chengli/Datasets/spam/feature_names.txt"))
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
        ClfDataSet data = StandardFormat.loadClfDataSet("/Users/chengli/Datasets/spam/train_data.txt",
                "/Users/chengli/Datasets/spam/train_label.txt", ",", DataSetType.CLF_DENSE);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};

        DataSetUtil.setExtLabels(data,extLabels);
        TRECFormat.save(data,"/Users/chengli/tmp/train.trec");
    }

    static void saveTestData()throws Exception{
        List<String> featureNames = loadFeatures();
        ClfDataSet data = StandardFormat.loadClfDataSet("/Users/chengli/Datasets/spam/test_data.txt",
                "/Users/chengli/Datasets/spam/test_label.txt", ",", DataSetType.CLF_DENSE);

        DataSetUtil.setFeatureNames(data,featureNames);
        String[] extLabels = {"non-spam","spam"};

        DataSetUtil.setExtLabels(data,extLabels);
        TRECFormat.save(data,"/Users/chengli/tmp/test.trec");
    }
}

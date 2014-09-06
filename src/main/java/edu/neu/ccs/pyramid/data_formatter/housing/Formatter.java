package edu.neu.ccs.pyramid.data_formatter.housing;

import edu.neu.ccs.pyramid.dataset.*;

import java.io.BufferedReader;
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
        saveData();
    }


    static List<String> loadFeatures() throws IOException {
        List<String> names = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader("/Users/chengli/Datasets/housing/feature_names.txt"))
        ){
            String line;
            while((line = br.readLine())!=null){
                int pos = line.indexOf(".");
                String name = line.substring(pos + 1).trim();
                names.add(name);
            }
        }
        return names;
    }

    static void partition() throws Exception{
        DataSetUtil.extractColumns("/Users/chengli/Datasets/housing/housing.data",
                "/Users/chengli/Datasets/housing/features.txt",0,12,Pattern.compile("\\s+"));
        DataSetUtil.extractColumns("/Users/chengli/Datasets/housing/housing.data",
                "/Users/chengli/Datasets/housing/labels.txt",13,13,Pattern.compile("\\s+"));

    }

    static void saveData() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet("/Users/chengli/Datasets/housing/standard_format/features.txt",
                "/Users/chengli/Datasets/housing/standard_format/labels.txt", ",", DataSetType.REG_DENSE);
        List<String> names = loadFeatures();
        DataSetUtil.setFeatureNames(dataSet,names);
        TRECFormat.save(dataSet,"/Users/chengli/Datasets/housing/trec_format/all.trec");
    }

}

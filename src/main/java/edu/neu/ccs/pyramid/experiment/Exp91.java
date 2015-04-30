package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.Ngram;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * select a subset of features based on max n and max slop, fields
 * save data
 * Created by chengli on 4/29/15.
 */
public class Exp91 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        (new File(config.getString("output.folder"))).mkdirs();
        sampleTrain(config);
        sampleTest(config);

    }

    public static void mainFromConfig(Config config) throws Exception{
        System.out.println(config);

        (new File(config.getString("output.folder"))).mkdirs();
        sampleTrain(config);
        sampleTest(config);
    }



    static ClfDataSet sample(Config config, ClfDataSet dataSet){
        int maxN = config.getInt("maxN");
        int maxSlop = config.getInt("maxSlop");
        Set<String> fields = new HashSet<>(config.getStrings("fields"));

        List<Integer> list = dataSet.getFeatureList().getAll().stream()
                .filter(feature -> ((Ngram)feature).getN()<=maxN&&((Ngram)feature).getSlop()<=maxSlop
                &&fields.contains(((Ngram) feature).getField()))
                .map(feature -> feature.getIndex()).collect(Collectors.toList());

        ClfDataSet subSet = DataSetUtil.sampleFeatures(dataSet,list);
        return subSet;
    }

    static void sampleTrain(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet subSet = sample(config,dataSet);
        String output = config.getString("output.folder");
        TRECFormat.save(subSet,new File(output, config.getString("output.trainData")));
    }

    static void sampleTest(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.testData")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet subSet = sample(config,dataSet);
        String output = config.getString("output.folder");
        TRECFormat.save(subSet,new File(output, config.getString("output.testData")));
    }


}

package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGBInspector;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.IMLGradientBoosting;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * dump boosting selected feature matrix
 * Created by chengli on 10/28/15.
 */
public class Exp136 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        new File(config.getString("output.folder")).mkdirs();

        List<Integer> features = loadList(config);
        selectTrain(config,features);
        selectTest(config,features);



    }


    private static void selectTrain(Config config, List<Integer> features) throws Exception{
        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.train"), DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet selected = DataSetUtil.sampleFeatures(train,features);
        TRECFormat.save(selected,new File(config.getString("output.folder"),"train.trec"));
    }

    private static void selectTest(Config config, List<Integer> features) throws Exception{
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.test"), DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet selected = DataSetUtil.sampleFeatures(test,features);
        TRECFormat.save(selected,new File(config.getString("output.folder"),"test.trec"));
    }

    private static List<Integer> loadList(Config config) throws Exception{
        Set<Integer> set = new HashSet<>();
        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));
        for (int k=0;k<boosting.getNumClasses();k++){
            TopFeatures topFeatures = IMLGBInspector.topFeatures(boosting, k, 100);
            topFeatures.getTopFeatures().stream().map(Feature::getIndex).forEach(set::add);
        }
        List<Integer> list =  set.stream().sorted().collect(Collectors.toList());
        System.out.println("list size = "+list.size());
        return list;
    }
}

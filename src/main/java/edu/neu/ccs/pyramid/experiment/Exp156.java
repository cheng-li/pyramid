package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * write mulit-label data for virgil
 * Created by chengli on 2/3/16.
 */
public class Exp156 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;
        trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                DataSetType.ML_CLF_SPARSE, true);
        testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                DataSetType.ML_CLF_SPARSE, true);

        String output = config.getString("output.folder");
        File folder = new File(output);
        folder.mkdirs();
        File train = new File(folder,"train");
        train.mkdirs();
        File test = new File(folder,"test");
        test.mkdirs();

        StandardFormat.save(trainSet,new File(train,"features.txt"),new File(train,"labels.txt")," ");
        StandardFormat.save(testSet,new File(test,"features.txt"),new File(test,"labels.txt")," ");

    }
}

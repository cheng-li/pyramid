package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * concatenate datasets by features
 * Created by chengli on 5/5/15.
 */
public class Exp96 {



    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        (new File(config.getString("output.folder"))).mkdirs();
//        concatenateTrain(config);
        concatenateTest(config);

    }

    public static void concatenateTrain(Config config) throws Exception{
        String input1 = config.getString("input.folder1");
        System.out.println("loading train 1");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(input1, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println("done");
        String input2 = config.getString("input.folder2");
        System.out.println("loading train 2");
        ClfDataSet dataSet2 = TRECFormat.loadClfDataSet(new File(input2, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println("done");
        System.out.println("concatenating");
        ClfDataSet dataSet = DataSetUtil.concatenateByColumn(dataSet1, dataSet2);
        System.out.println("done");
        String output = config.getString("output.folder");
        TRECFormat.save(dataSet,new File(output, "train.trec"));
    }


    public static void concatenateTest(Config config) throws Exception{
        String input1 = config.getString("input.folder1");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(input1, "test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println("1 loaded");
        String input2 = config.getString("input.folder2");
        ClfDataSet dataSet2 = TRECFormat.loadClfDataSet(new File(input2, "test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println("2 loaded");
        ClfDataSet dataSet = DataSetUtil.concatenateByColumn(dataSet1, dataSet2);
        System.out.println("merged");
        String output = config.getString("output.folder");
        TRECFormat.save(dataSet,new File(output, "test.trec"));
    }
}

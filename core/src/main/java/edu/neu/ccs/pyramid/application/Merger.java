package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

/**
 * merge two datasets into one
 * Created by chengli on 5/18/16.
 */
public class Merger {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        System.out.println(config);
        String dataType = config.getString("dataSetType");
        switch (dataType) {
            case "clf":
                mergeClfData(config);
                break;
            case "reg":
                //todo
                break;
        }

    }


    private static void mergeClfData(Config config) throws Exception{
        String input1 = config.getString("input.data1");
        String input2 = config.getString("input.data2");
        String output= config.getString("output.data");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(input1,DataSetType.CLF_DENSE,true);
        ClfDataSet dataSet2 = TRECFormat.loadClfDataSet(input2,DataSetType.CLF_DENSE,true);
        ClfDataSet merged = DataSetUtil.concatenateByRow(dataSet1,dataSet2);
        TRECFormat.save(merged,output);

    }
}

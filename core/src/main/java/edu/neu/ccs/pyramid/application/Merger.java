package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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
            case "mlclf":
                mergeMLClfData(config);
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

    private static void mergeMLClfData(Config config) throws Exception{
        List<String> inputs = config.getStrings("input.dataSets");
        String output= config.getString("output.dataSet");
        List<MultiLabelClfDataSet> dataSets = new ArrayList<>();
        for (String input: inputs){
            MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(input,DataSetType.ML_CLF_SPARSE,true);
            dataSets.add(dataSet);
        }
        MultiLabelClfDataSet merged = DataSetUtil.concatenateByRow(dataSets);
        TRECFormat.save(merged,output);

    }
}

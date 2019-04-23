package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.util.ArrayList;
import java.util.List;

public class InstanceConcatenator {

    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        List<String> inputs = config.getStrings("input.dataSets");
        String output = config.getString("output.dataSet");

        List<MultiLabelClfDataSet> dataSets = new ArrayList<>();
        for (String input: inputs){
            MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(input, DataSetType.ML_CLF_SPARSE,true);
//            MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(input, DataSetType.ML_CLF_DENSE,true);
            dataSets.add(dataSet);
        }

        MultiLabelClfDataSet merged = DataSetUtil.concatenateByRow(dataSets);
        TRECFormat.save(merged,output);
    }
}

package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.util.List;

/**
 * Created by chengli on 6/11/15.
 */
public class FeatureBinarizer {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        System.out.println(config);

        List<String> inputs = config.getStrings("input.trecFolders");
        List<String> outputs = config.getStrings("output.trecFolders");
        if (inputs.size()!=outputs.size()){
            throw new IllegalArgumentException("inputs.size()!=outputs.size()");
        }

        for (int i=0;i<inputs.size();i++){
            String input = inputs.get(i);
            String output = outputs.get(i);
            binarize(config,input,output);
        }

    }

    private static void binarize(Config config, String inputData, String outputData) throws Exception{
        String dataType = config.getString("dataSetType");
        DataSet dataSet;
        switch (dataType){
            case "reg":
                dataSet = TRECFormat.loadRegDataSet(inputData, DataSetType.REG_SPARSE,true);
                break;
            case "clf":
                dataSet = TRECFormat.loadClfDataSet(inputData,DataSetType.CLF_SPARSE,true);
                break;
            case "multiLabel":
                dataSet = TRECFormat.loadMultiLabelClfDataSet(inputData,DataSetType.ML_CLF_SPARSE,true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }

        DataSetUtil.binarizeFeature(dataSet);
        TRECFormat.save(dataSet,outputData);
    }
}

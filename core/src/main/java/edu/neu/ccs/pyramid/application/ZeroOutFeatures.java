package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class ZeroOutFeatures {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("inputData"), DataSetType.ML_CLF_SPARSE,true);
        List<Integer> indices = parse(config.getString("featureIndices"));
        System.out.println("features to zero out = "+indices);
        for (int j:indices){
            List<Integer> nonZeros = new ArrayList<>();
            for (Vector.Element nonZero: dataSet.getColumn(j).nonZeroes()){
                nonZeros.add(nonZero.index());
            }

            for (int i: nonZeros){
                dataSet.setFeatureValue(i,j,0);
            }
        }

        TRECFormat.save(dataSet,config.getString("outputData"));
    }

    private static List<Integer> parse(String featureIndices){
        List<Integer> list = new ArrayList<>();
        String[] split = featureIndices.replace(" ","").split(",");
        for (String range: split){
            if (range.contains("-")){
                int start = Integer.parseInt(range.split(Pattern.quote("-"))[0]);
                int end = Integer.parseInt(range.split(Pattern.quote("-"))[1]);
                for (int i=start;i<=end;i++){
                    list.add(i);
                }
            } else {
                list.add(Integer.parseInt(range));
            }
        }
        return list;
    }
}

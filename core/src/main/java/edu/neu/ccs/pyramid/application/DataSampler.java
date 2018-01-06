package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DataSampler {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.dataSet"), DataSetType.ML_CLF_SPARSE,true);
        List<Integer> all = IntStream.range(0,dataSet.getNumDataPoints()).boxed().collect(Collectors.toList());
        int randomSeed = config.getInt("randomSeed");
        List<Integer> keep = Sampling.sampleByPercentage(all,config.getDouble("percentage"),randomSeed);
        MultiLabelClfDataSet subset = DataSetUtil.sampleData(dataSet,keep);
        TRECFormat.save(subset,config.getString("output.dataSet"));
    }
}

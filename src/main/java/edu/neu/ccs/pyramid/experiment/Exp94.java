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
import java.util.stream.IntStream;

/** keep features with mindf
 * Created by chengli on 5/4/15.
 */
public class Exp94 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        (new File(config.getString("output.folder"))).mkdirs();

        List<Integer> list = getList(config);

        sampleTrain(config,list);
        sampleTest(config,list);

    }



    static List<Integer> getList(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);
        int minDf = config.getInt("minDf");
        List<Integer> list = IntStream.range(0,dataSet.getNumFeatures())
                .filter(i -> dataSet.getColumn(i).getNumNonZeroElements()>=minDf)
                .mapToObj(i -> i)
                .collect(Collectors.toList());
        return list;
    }


    static ClfDataSet sample(List<Integer> list, ClfDataSet dataSet){

        ClfDataSet subSet = DataSetUtil.sampleFeatures(dataSet, list);
        return subSet;
    }

    static void sampleTrain(Config config,List<Integer> list) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet subSet = sample(list,dataSet);
        String output = config.getString("output.folder");
        TRECFormat.save(subSet,new File(output, config.getString("output.trainData")));
    }

    static void sampleTest(Config config,List<Integer> list) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.testData")),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet subSet = sample(list,dataSet);
        String output = config.getString("output.folder");
        TRECFormat.save(subSet,new File(output, config.getString("output.testData")));
    }

}

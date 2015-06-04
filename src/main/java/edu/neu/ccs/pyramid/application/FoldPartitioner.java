package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 6/3/15.
 */
public class FoldPartitioner {
    public static void main(String[] args) throws Exception{
        Config config = new Config(args[0]);
        System.out.println(config);
        String dataType = config.getString("dataSetType");
        switch (dataType) {
            case "clf":
                partitionClfData(config);
                break;
            case "reg":
                partitionRegData(config);
                break;
        }

    }

    private static void partitionClfData(Config config) throws Exception{
        //todo implement
    }

    private static void partitionRegData(Config config) throws Exception{

        String input = config.getString("input.data");
        String output= config.getString("output.folder");
        int numFolds = config.getInt("numFolds");
        RegDataSet all = TRECFormat.loadRegDataSet(input, DataSetType.REG_SPARSE,true);
        for (int i=1;i<=numFolds;i++){
            Set<Integer> trainFold = new HashSet<>();
            for (int j=1;j<=numFolds;j++){
                if (j!=i){
                    trainFold.add(j);
                }
            }
            Set<Integer> testFold = new HashSet<>();
            testFold.add(i);
            RegDataSet trainSet = DataSetUtil.sampleByFold(all,numFolds,trainFold);
            RegDataSet testSet = DataSetUtil.sampleByFold(all,numFolds,testFold);

            File foldFolder = new File(output,"fold_"+i);
            foldFolder.mkdirs();
            TRECFormat.save(trainSet,new File(foldFolder,"train.trec"));
            TRECFormat.save(testSet,new File(foldFolder,"test.trec"));
        }
    }



}

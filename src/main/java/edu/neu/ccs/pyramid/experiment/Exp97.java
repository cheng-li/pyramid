package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Pair;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

/**
 * merge train and test, repartition into folds
 * Created by chengli on numFold/numFold/1numFold.
 */
public class Exp97 {
    
    static int numFold = 5;
    //todo make number a para

    public static void mainFromConfig(Config config) throws Exception{

        ClfDataSet all = merge(config);
        for (int i=1;i<=numFold;i++){
            produceFold(config,all,numFold,i);

        }
    }


    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ClfDataSet all = merge(config);
        for (int i=1;i<=numFold;i++){
            produceFold(config,all,numFold,i);

        }

    }

    public static ClfDataSet merge(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet dataSet2 = TRECFormat.loadClfDataSet(new File(input, "test.trec"),
                DataSetType.CLF_SPARSE, true);
        return DataSetUtil.concatenateByRow(dataSet1,dataSet2);
    }

    public static void produceFold(Config config, ClfDataSet all, int numFolds, int fold){
        Pair<ClfDataSet, ClfDataSet> pair = DataSetUtil.splitToTrainValidation(all,1-1.0/numFolds);
        String output = config.getString("output.folder");
        File foldFolder = new File(output,"fold_"+fold);
        foldFolder.mkdirs();
        TRECFormat.save(pair.getFirst(),new File(foldFolder, "train.trec"));
        TRECFormat.save(pair.getSecond(),new File(foldFolder, "test.trec"));
    }



}

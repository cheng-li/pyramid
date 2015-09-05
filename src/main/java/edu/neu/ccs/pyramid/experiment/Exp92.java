package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Pair;

import java.io.File;

/**
 * partition train into train/valid
 * Created by chengli on 5/3/15.
 */
public class Exp92 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, config.getString("input.trainData")),
                DataSetType.CLF_SPARSE, true);

        Pair<ClfDataSet,ClfDataSet> split = DataSetUtil.splitToTrainValidation(dataSet,0.8);
        String output = config.getString("output.folder");
        TRECFormat.save(split.getFirst(),new File(output, config.getString("output.trainData")));
        TRECFormat.save(split.getSecond(),new File(output, config.getString("output.testData")));

    }


}

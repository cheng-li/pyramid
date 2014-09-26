package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * Created by chengli on 9/26/14.
 */
public class Exp7 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(config.getString("trecFolder"), DataSetType.CLF_SPARSE,true);
        DataSetUtil.dumpDataSettings(dataSet,new File(config.getString("trecFolder"),"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(dataSet,new File(config.getString("trecFolder"),"feature_settings.txt"));
    }


}

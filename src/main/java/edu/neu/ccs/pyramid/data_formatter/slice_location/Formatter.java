package edu.neu.ccs.pyramid.data_formatter.slice_location;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;

/**
 * Created by chengli on 5/25/15.
 */
public class Formatter {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        saveData();
    }

    static void saveData() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "/slice_location/standard/features.txt"),
                new File(DATASETS, "/slice_location/standard/labels.txt"), ",", DataSetType.REG_DENSE, false);
//        List<String> names = loadFeatures();
//        DataSetUtil.setFeatureNames(dataSet,names);
        TRECFormat.save(dataSet, new File(DATASETS, "slice_location/trec_format/all.trec"));
    }
}

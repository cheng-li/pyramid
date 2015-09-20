package edu.neu.ccs.pyramid.data_formatter.ccpp;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;
import java.nio.file.Paths;

/**
 * Created by chengli on 9/20/15.
 */
public class Formatter {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        File feature = Paths.get(DATASETS, "ccpp", "feature.csv").toFile();
        File label = Paths.get(DATASETS, "ccpp", "label.csv").toFile();
        RegDataSet dataSet = StandardFormat.loadRegDataSet(feature, label, ",", DataSetType.REG_SPARSE, false);
        TRECFormat.save(dataSet, Paths.get(TMP, "all.trec").toFile());
    }
}

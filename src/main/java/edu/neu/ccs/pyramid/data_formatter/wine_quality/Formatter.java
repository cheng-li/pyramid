package edu.neu.ccs.pyramid.data_formatter.wine_quality;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by chengli on 9/20/15.
 */
public class Formatter {

    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        File feature = Paths.get(DATASETS,"wine_quality","standard","all_feature.txt").toFile();
        File label = Paths.get(DATASETS, "wine_quality", "standard", "all_label.txt").toFile();
        RegDataSet dataSet = StandardFormat.loadRegDataSet(feature,label,";", DataSetType.REG_SPARSE,false);
        TRECFormat.save(dataSet,Paths.get(TMP,"all.trec").toFile());
    }
}

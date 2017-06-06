package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.MekaFormat;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by Rainicy on 8/3/15.
 */
public class Trec2Meka {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        Config config = new Config(args[0]);
        System.out.println(config);


        List<String> trecs = config.getStrings("trec");
        List<String> mekas = config.getStrings("meka");

        for (int i=0; i<trecs.size(); i++) {
            MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(trecs.get(i)),
                    DataSetType.ML_CLF_SPARSE, true);
            System.out.println(i + " -- Translating on trecs: " + trecs.get(i));
            MekaFormat.save(dataSet, mekas.get(i), config);
        }
    }
}

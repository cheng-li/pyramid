package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.io.File;
import java.util.List;

/**
 * Created by Rainicy on 8/22/15.
 */
public class MLTrec2LibSvm {
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        System.out.println(config);
        List<String> trecs = config.getStrings("trec");
        List<String> libSVMs = config.getStrings("libSVM");


        for (int i=0; i<trecs.size(); i++) {
            MultiLabelClfDataSet trecDataset = TRECFormat.loadMultiLabelClfDataSet(new File(trecs.get(i)),
                    DataSetType.ML_CLF_SPARSE, false);
            System.out.println(i + " -- Translating on trecs: " + trecs.get(i));
            LibSvmFormat.save(trecDataset, libSVMs.get(i));
        }
    }
}

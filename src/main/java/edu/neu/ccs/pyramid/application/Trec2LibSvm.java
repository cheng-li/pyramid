package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.LibSvmFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.File;
import java.util.List;

/**
 * Created by chengli on 11/11/14.
 */
public class Trec2LibSvm {
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        System.out.println(config);
        List<String> trecs = config.getStrings("trec");
        List<String> libSVMs = config.getStrings("libSVM");


        for (int i=0; i<trecs.size(); i++) {
            ClfDataSet trecDataset = TRECFormat.loadClfDataSet(new File(trecs.get(i)),
                    DataSetType.CLF_SPARSE, false);
            System.out.println(i + " -- Trainslating on trecs: " + trecs.get(i));
            LibSvmFormat.save(trecDataset, libSVMs.get(i));
        }
    }
}

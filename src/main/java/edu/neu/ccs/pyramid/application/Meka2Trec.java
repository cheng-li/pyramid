package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.MekaFormat;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;

import java.io.IOException;
import java.util.List;

/**
 * Created by Rainicy on 10/31/15.
 */
public class Meka2Trec {
    /**
     * this is only support multi-label classification dataset.
     * @param args
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        Config config = new Config(args[0]);
        System.out.println(config);


        List<String> trecs = config.getStrings("trec");
        List<String> mekas = config.getStrings("meka");
        int numLabels = config.getInt("numLabels");
        int numFeatures = config.getInt("numFeatures");
        String dataMode = config.getString("dataMode");
        for (int i=0; i<mekas.size(); i++) {
            System.out.println("processing on: " + trecs.get(i));
            MultiLabelClfDataSet dataSet = MekaFormat.loadMLClfDataset(mekas.get(i), numFeatures, numLabels, dataMode);
            TRECFormat.save(dataSet, trecs.get(i));
        }

    }
}

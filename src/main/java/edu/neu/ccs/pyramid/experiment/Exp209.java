package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelSuggester;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Rainicy on 9/26/15.
 */
public class Exp209 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String inputData = config.getString("input.data");
        int numClusters = config.getInt("num.clusters");

        MultiLabelClfDataSet multiLabelClfDataSet;
        if (config.getBoolean("isDense")) {
            multiLabelClfDataSet = TRECFormat.loadMultiLabelClfDataSet(
                    new File(inputData), DataSetType.ML_CLF_DENSE, true);
        } else {
            multiLabelClfDataSet = TRECFormat.loadMultiLabelClfDataSet(
                    new File(inputData), DataSetType.ML_CLF_SPARSE, true);
        }

        System.out.println("Starting to train clusters ... ");
        MultiLabelSuggester suggester = new MultiLabelSuggester(multiLabelClfDataSet, numClusters);

        System.out.println("bmm="+suggester.getBmm());
        if (config.getBoolean("isSample")) {
            int numSamples = config.getInt("num.samples");
//
//            System.out.println();
//            System.out.println();
//            System.out.println("Sampling for each cluster: ");
//            for(int k=0; k<suggester.getBmm().getNumClusters(); k++) {
//                System.out.println("Sample for cluster: " + k);
//                Set<Vector> sampleVectors = new HashSet<>();
//                while (sampleVectors.size() < numSamples) {
//                    Vector vector = suggester.getBmm().sample(k);
//                    if (!sampleVectors.contains(vector)) {
//                        sampleVectors.add(vector);
//                    }
//                }
//                for (Vector vector : sampleVectors) {
//                    System.out.println(vector.toString());
//                }
//            }

            System.out.println("Sampling for mixtures clusters:");
            Set<Vector> sampleVectors = new HashSet<>();
            while(sampleVectors.size() < numSamples) {
                Vector vector = suggester.getBmm().sample();
                if (!sampleVectors.contains(vector)) {
                    sampleVectors.add(vector);
                }
            }
            List<String> names = suggester.getBmm().getNames();
            for (Vector vector : sampleVectors) {
                double prob = 0.0;
                for (int k=0; k<suggester.getBmm().getNumClusters(); k++) {
                    prob += suggester.getBmm().getMixtureCoefficients()[k] * suggester.getBmm().probability(vector, k);
                }
                System.out.print("prob: " + prob + "\t");
                for (Vector.Element e : vector.nonZeroes()) {
                    System.out.print(names.get(e.index()) + "\t");
                }
                System.out.println();
            }
        }

    }
}

package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.FeatureColumn;
import edu.neu.ccs.pyramid.dataset.DenseFeatureColumn;
import edu.neu.ccs.pyramid.dataset.FeatureSetting;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by Rainicy on 10/8/14.
 */
public class ParseDataSet {

    /**
     * Given the ClfDataset, and filter the dataset by
     * given the label and the feature.
     *
     */
    static public double[][] getFeatureColumnByLabelFeature(
            ClfDataSet clfDataSet, int label, int feature) {


        int[] labels = clfDataSet.getLabels();

        // Go through the data set.
        for (int i=0; i<labels.length; i++) {
            if (labels[i] == label) {

            }
        }
        return null;
    }
}

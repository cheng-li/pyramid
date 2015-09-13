package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.clustering.bmm.BMM;
import edu.neu.ccs.pyramid.clustering.bmm.BMMSelector;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.SetUtil;
import org.apache.mahout.math.Vector;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by chengli on 9/12/15.
 */
public class MultiLabelSuggester {
    private BMM bmm;
    private MultiLabel[] multiLabels;
    private int numClusters;

    public MultiLabelSuggester(MultiLabelClfDataSet dataSet, int numClusters){
        this(dataSet.getNumClasses(),dataSet.getMultiLabels(),numClusters);
    }

    public MultiLabelSuggester(int numClasses, MultiLabel[] multiLabels, int numClusters) {
        this.multiLabels = multiLabels;
        this.numClusters = numClusters;
        DataSet dataSet = DataSetBuilder.getBuilder()
                .numDataPoints(multiLabels.length)
                .numFeatures(numClasses)
                .build();
        for (int i=0;i<multiLabels.length;i++){
            MultiLabel multiLabel = multiLabels[i];
            for (int label: multiLabel.getMatchedLabels()){
                dataSet.setFeatureValue(i,label,1);
            }
        }
        this.bmm = BMMSelector.select(dataSet,numClusters,10);
    }

    public Set<MultiLabel> suggestNewOnes(int numSamples){
        Set<MultiLabel> found = new HashSet<>();
        Set<MultiLabel> old = new HashSet<>();
        for (MultiLabel multiLabel:multiLabels){
            found.add(multiLabel);
            old.add(multiLabel);
        }
        for (int i=0;i<numSamples;i++){
            Vector vector = bmm.sample();
            MultiLabel multiLabel = new MultiLabel(vector);
            found.add(multiLabel);
        }
        Set<MultiLabel> newOnes = SetUtil.complement(found,old);
        return newOnes;
    }

    public BMM getBmm() {
        return bmm;
    }
}

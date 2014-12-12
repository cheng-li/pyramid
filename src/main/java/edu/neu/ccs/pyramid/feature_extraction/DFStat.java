package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Created by chengli on 9/13/14.
 */
public class DFStat implements Serializable {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private String phrase;
    private long df;
    private long[] dfForClasses;

    public DFStat(int numClasses, String phrase) {
        this.numClasses = numClasses;
        this.phrase = phrase;
        this.dfForClasses = new long[numClasses];
    }

    public DFStat(int numClasses, String term, SingleLabelIndex index, String[] ids) {
        this(numClasses,term);
        List<String> docs = null;
        try {
            docs = index.getDocs(term,ids);
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (String doc: docs){
            int label = index.getLabel(doc);
            this.dfForClasses[label] += 1;
        }
    }

    public long getDf() {
        return df;
    }

    public void setDf(long df) {
        this.df = df;
    }

    public void setDfForClass(long df, int classIndex){
        this.dfForClasses[classIndex] = df;
    }

    public long getDfForClass(int classIndex){
        return this.dfForClasses[classIndex];
    }

    public double getPurityForClass(int classIndex){
        return ((double)dfForClasses[classIndex])/(double)df;
    }

    public String getPhrase() {
        return phrase;
    }

    public int getBestMatchedClass(){
        int match = 0;
        long maxDf = 0;
        for (int i=0;i<numClasses;i++){
            if (dfForClasses[i]>maxDf){
                match = i;
                maxDf = dfForClasses[i];
            }
        }
        return match;
    }

    @Override
    public String toString() {
        return "DFStat{" +
                "numClasses=" + numClasses +
                ", phrase='" + phrase + '\'' +
                ", df=" + df +
                ", dfForClasses=" + Arrays.toString(dfForClasses) +
                '}';
    }
}

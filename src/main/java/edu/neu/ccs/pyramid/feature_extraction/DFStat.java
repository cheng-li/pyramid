package edu.neu.ccs.pyramid.feature_extraction;

import java.io.Serializable;
import java.util.Arrays;

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

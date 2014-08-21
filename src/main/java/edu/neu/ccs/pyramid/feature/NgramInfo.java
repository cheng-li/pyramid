package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 8/20/14.
 */
public class NgramInfo {
    private Ngram ngram;
    private int tf=0;
    private int df=0;
    private float tfidf=0;

    public NgramInfo(Ngram ngram) {
        this.ngram = ngram;
    }

    /**
     * sum up tf and tfidf
     * @param ngramInfo1
     * @param ngramInfo2
     */
    public static NgramInfo combine(NgramInfo ngramInfo1, NgramInfo ngramInfo2){
        if (!ngramInfo1.ngram.equals(ngramInfo2.ngram)){
            throw new IllegalArgumentException("!ngramInfo1.ngram.equals(ngramInfo2.ngram");
        }
        NgramInfo ngramInfo = new NgramInfo(ngramInfo1.ngram);
        ngramInfo.tf = ngramInfo1.tf + ngramInfo2.tf;
        ngramInfo.df = ngramInfo1.df;
        ngramInfo.tfidf = ngramInfo1.tfidf + ngramInfo2.tfidf;
        return ngramInfo;
    }

    public Ngram getNgram() {
        return ngram;
    }

    public int getTf() {
        return tf;
    }

    public NgramInfo setTf(int tf) {
        this.tf = tf;
        return this;
    }

    public int getDf() {
        return df;
    }

    public NgramInfo setDf(int df) {
        this.df = df;
        return this;
    }

    public float getTfidf() {
        return tfidf;
    }

    public NgramInfo setTfidf(float tfidf) {
        this.tfidf = tfidf;
        return this;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NgramInfo ngramInfo = (NgramInfo) o;

        return ngram.equals(ngramInfo.ngram);

    }

    @Override
    public int hashCode() {
        return ngram.hashCode();
    }

    @Override
    public String toString() {
        return "NgramInfo{" +
                "ngram=" + ngram +
                ", tf=" + tf +
                ", df=" + df +
                ", tfidf=" + tfidf +
                '}';
    }
}

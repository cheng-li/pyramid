package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 8/20/14.
 */
public class Ngram extends Feature{
    protected int numTerms;
    public Ngram(String featureName) {
        super(featureName);
        String[] terms = this.featureName.split("~");
        this.numTerms = terms.length;
    }

    public int getNumTerms() {
        return numTerms;
    }

    @Override
    public String toString() {
        return this.featureName;
    }



    @Override
    public int hashCode() {
        return this.featureName.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Ngram ngram = (Ngram) o;

        if (numTerms != ngram.numTerms) return false;
        if (! featureName.equals(ngram.featureName)) return false;

        return true;
    }
}

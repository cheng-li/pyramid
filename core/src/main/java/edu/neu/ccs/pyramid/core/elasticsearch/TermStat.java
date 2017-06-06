package edu.neu.ccs.pyramid.core.elasticsearch;

/**
 * Created by chengli on 9/6/14.
 */
public class TermStat {
    private String term;
    private int tf=0;
    private int df=0;
    private float tfidf=0;

    public TermStat(String term) {
        this.term = term;
    }

    public String getTerm() {
        return term;
    }

    public void setTerm(String term) {
        this.term = term;
    }

    public int getTf() {
        return tf;
    }

    public TermStat setTf(int tf) {
        this.tf = tf;
        return this;
    }

    public int getDf() {
        return df;
    }

    public TermStat setDf(int df) {
        this.df = df;
        return this;
    }

    public float getTfidf() {
        return tfidf;
    }

    public TermStat setTfidf(float tfidf) {
        this.tfidf = tfidf;
        return this;
    }

    public static TermStat combine(TermStat termStat1, TermStat termStat2){
        if (!termStat1.term.equals(termStat2.term)){
            throw new IllegalArgumentException("!termStat1.term.equals(termStat2.term)");
        }
        TermStat termStat = new TermStat(termStat1.term);
        termStat.tf = termStat1.tf + termStat2.tf;
        termStat.df = termStat1.df;
        termStat.tfidf = termStat1.tfidf + termStat2.tfidf;
        return termStat;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TermStat termStat = (TermStat) o;

        if (!term.equals(termStat.term)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return term.hashCode();
    }

    @Override
    public String toString() {
        return "TermStat{" +
                "term='" + term + '\'' +
                ", tf=" + tf +
                ", df=" + df +
                ", tfidf=" + tfidf +
                '}';
    }
}

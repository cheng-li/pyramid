package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 4/26/15.
 */
public class SpanNotNgram extends Feature {
    private Ngram include;
    private Ngram exclude;
    private int pre;
    private int post;

    public Ngram getInclude() {
        return include;
    }

    public void setInclude(Ngram include) {
        this.include = include;
    }

    public Ngram getExclude() {
        return exclude;
    }

    public void setExclude(Ngram exclude) {
        this.exclude = exclude;
    }

    public int getPre() {
        return pre;
    }

    public void setPre(int pre) {
        this.pre = pre;
    }

    public int getPost() {
        return post;
    }

    public void setPost(int post) {
        this.post = post;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        SpanNotNgram that = (SpanNotNgram) o;

        if (post != that.post) return false;
        if (pre != that.pre) return false;
        if (!exclude.equals(that.exclude)) return false;
        if (!include.equals(that.include)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + include.hashCode();
        result = 31 * result + exclude.hashCode();
        result = 31 * result + pre;
        result = 31 * result + post;
        return result;
    }
}

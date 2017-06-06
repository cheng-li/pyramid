package edu.neu.ccs.pyramid.core.feature;

import java.util.ArrayList;
import java.util.List;

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


    public static List<SpanNotNgram> breakBigram(Ngram ngram){
        if (ngram.getN()!=2){
            throw new IllegalArgumentException("n!=2");
        }

        Ngram ngram1 = new Ngram();
        ngram1.setNgram(ngram.getTerms()[0]);
        ngram1.setSlop(0);
        ngram1.setField(ngram.getField());
        ngram1.setInOrder(true);


        Ngram ngram2 = new Ngram();
        ngram2.setNgram(ngram.getTerms()[1]);
        ngram2.setSlop(0);
        ngram2.setField(ngram.getField());
        ngram2.setInOrder(true);

        List<SpanNotNgram> spanNotNgrams = new ArrayList<>();
        SpanNotNgram spanNotNgram1 = new SpanNotNgram();
        spanNotNgram1.setInclude(ngram1);
        spanNotNgram1.setExclude(ngram);

        SpanNotNgram spanNotNgram2 = new SpanNotNgram();
        spanNotNgram2.setInclude(ngram2);
        spanNotNgram2.setExclude(ngram);

        spanNotNgrams.add(spanNotNgram1);
        spanNotNgrams.add(spanNotNgram2);
        return spanNotNgrams;

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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("SpanNotNgram{");
        sb.append("include=").append(include);
        sb.append(", exclude=").append(exclude);
        sb.append(", pre=").append(pre);
        sb.append(", post=").append(post);
        if (indexAssigned){
            sb.append("index=").append(index);
        }
        sb.append('}');
        return sb.toString();
    }
}

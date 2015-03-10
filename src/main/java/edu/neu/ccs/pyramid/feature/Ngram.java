package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 3/7/15.
 */
public class Ngram extends Feature {
    private String ngram;
    private String field;
    private int slop;

    public String getNgram() {
        return ngram;
    }

    public void setNgram(String ngram) {
        this.ngram = ngram;
    }

    public String getField() {
        return field;
    }

    public void setField(String field) {
        this.field = field;
    }

    public int getSlop() {
        return slop;
    }

    public void setSlop(int slop) {
        this.slop = slop;
    }

    public int getN(){
        return ngram.split(" ").length;
    }
}

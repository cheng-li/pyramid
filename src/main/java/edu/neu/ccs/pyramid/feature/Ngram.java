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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Ngram{");
        sb.append(super.toString()).append(", ");
        sb.append("ngram='").append(ngram).append('\'');
        sb.append(", field='").append(field).append('\'');
        sb.append(", slop=").append(slop);
        sb.append('}');
        return sb.toString();
    }
}

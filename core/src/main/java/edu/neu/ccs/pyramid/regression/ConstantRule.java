package edu.neu.ccs.pyramid.regression;

/**
 * Created by chengli on 2/28/15.
 */
public class ConstantRule implements Rule{
    private double score;

    public ConstantRule(double score) {
        this.score = score;
    }

    public double getScore(){
        return score;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("ConstantRule{");
        sb.append("score=").append(score);
        sb.append('}');
        return sb.toString();
    }
}

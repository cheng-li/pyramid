package edu.neu.ccs.pyramid.core.feature;

import java.io.Serializable;

/**
 * Created by chengli on 1/27/15.
 */
public class FeatureUtility implements Serializable {
    private static final long serialVersionUID = 1L;
    private Feature feature;
    private double utility;
    private int rank;


    public FeatureUtility(Feature feature) {
        this.feature = feature;
    }

    public Feature getFeature() {
        return feature;
    }

    public double getUtility() {
        return utility;
    }

    public FeatureUtility setUtility(double utility) {
        this.utility = utility;
        return this;
    }

    public int getRank() {
        return rank;
    }

    public FeatureUtility setRank(int rank) {
        this.rank = rank;
        return this;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        FeatureUtility that = (FeatureUtility) o;

        if (!feature.equals(that.feature)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return feature.hashCode();
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("FeatureUtility{");
        sb.append("feature=").append(feature);
        sb.append(", utility=").append(utility);
        sb.append(", rank=").append(rank);
        sb.append('}');
        return sb.toString();
    }
}

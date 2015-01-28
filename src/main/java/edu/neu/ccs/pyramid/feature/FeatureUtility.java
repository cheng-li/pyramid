package edu.neu.ccs.pyramid.feature;

/**
 * Created by chengli on 1/27/15.
 */
public class FeatureUtility {
    private int index;
    private String name;
    private double utility;
    private int rank;


    public FeatureUtility(int index, String name) {
        this.index = index;
        this.name = name;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
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

        if (index != that.index) return false;
        if (!name.equals(that.name)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = index;
        result = 31 * result + name.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "FeatureUtility{" +
                "index=" + index +
                ", name='" + name + '\'' +
                ", utility=" + utility +
                ", rank=" + rank +
                '}';
    }
}

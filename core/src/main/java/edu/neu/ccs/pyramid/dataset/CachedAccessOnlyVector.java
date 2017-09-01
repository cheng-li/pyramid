package edu.neu.ccs.pyramid.dataset;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.OrderedIntDoubleMapping;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;

/**
 * an access wrapper for RandomAccessSparseVector with value cache
 * the only method implemented is get
 * not thread safe!
 * Created by chengli on 6/11/17.
 */
public class CachedAccessOnlyVector implements Vector {

    private RandomAccessSparseVector randomAccessSparseVector;
    private double[] cachedValues;
    private boolean[] cached;

    public CachedAccessOnlyVector(RandomAccessSparseVector randomAccessSparseVector) {
        this.randomAccessSparseVector = randomAccessSparseVector;
        this.cachedValues = new double[randomAccessSparseVector.size()];
        this.cached = new boolean[randomAccessSparseVector.size()];
    }

    @Override
    public double get(int i) {
        if (cached[i]){
            return cachedValues[i];
        } else {
            double value = randomAccessSparseVector.get(i);
            cached[i] = true;
            cachedValues[i] = value;
            return value;
        }
    }


    @Override
    public String asFormatString() {
        return null;
    }

    @Override
    public Vector assign(double v) {
        return null;
    }

    @Override
    public Vector assign(double[] doubles) {
        return null;
    }

    @Override
    public Vector assign(Vector vector) {
        return null;
    }

    @Override
    public Vector assign(DoubleFunction doubleFunction) {
        return null;
    }

    @Override
    public Vector assign(Vector vector, DoubleDoubleFunction doubleDoubleFunction) {
        return null;
    }

    @Override
    public Vector assign(DoubleDoubleFunction doubleDoubleFunction, double v) {
        return null;
    }

    @Override
    public int size() {
        return 0;
    }

    @Override
    public boolean isDense() {
        return false;
    }

    @Override
    public boolean isSequentialAccess() {
        return false;
    }

    @Override
    public Vector clone() {
        return null;
    }

    @Override
    public Iterable<Element> all() {
        return null;
    }

    @Override
    public Iterable<Element> nonZeroes() {
        return null;
    }

    @Override
    public Element getElement(int i) {
        return null;
    }

    @Override
    public void mergeUpdates(OrderedIntDoubleMapping orderedIntDoubleMapping) {

    }

    @Override
    public Vector divide(double v) {
        return null;
    }

    @Override
    public double dot(Vector vector) {
        return 0;
    }



    @Override
    public double getQuick(int i) {
        return 0;
    }

    @Override
    public Vector like() {
        return null;
    }

    @Override
    public Vector like(int i) {
        return null;
    }

    @Override
    public Vector minus(Vector vector) {
        return null;
    }

    @Override
    public Vector normalize() {
        return null;
    }

    @Override
    public Vector normalize(double v) {
        return null;
    }

    @Override
    public Vector logNormalize() {
        return null;
    }

    @Override
    public Vector logNormalize(double v) {
        return null;
    }

    @Override
    public double norm(double v) {
        return 0;
    }

    @Override
    public double minValue() {
        return 0;
    }

    @Override
    public int minValueIndex() {
        return 0;
    }

    @Override
    public double maxValue() {
        return 0;
    }

    @Override
    public int maxValueIndex() {
        return 0;
    }

    @Override
    public Vector plus(double v) {
        return null;
    }

    @Override
    public Vector plus(Vector vector) {
        return null;
    }

    @Override
    public void set(int i, double v) {

    }

    @Override
    public void setQuick(int i, double v) {

    }

    @Override
    public void incrementQuick(int i, double v) {

    }

    @Override
    public int getNumNondefaultElements() {
        return 0;
    }

    @Override
    public int getNumNonZeroElements() {
        return 0;
    }

    @Override
    public Vector times(double v) {
        return null;
    }

    @Override
    public Vector times(Vector vector) {
        return null;
    }

    @Override
    public Vector viewPart(int i, int i1) {
        return null;
    }

    @Override
    public double zSum() {
        return 0;
    }

    @Override
    public Matrix cross(Vector vector) {
        return null;
    }

    @Override
    public double aggregate(DoubleDoubleFunction doubleDoubleFunction, DoubleFunction doubleFunction) {
        return 0;
    }

    @Override
    public double aggregate(Vector vector, DoubleDoubleFunction doubleDoubleFunction, DoubleDoubleFunction doubleDoubleFunction1) {
        return 0;
    }

    @Override
    public double getLengthSquared() {
        return 0;
    }

    @Override
    public double getDistanceSquared(Vector vector) {
        return 0;
    }

    @Override
    public double getLookupCost() {
        return 0;
    }

    @Override
    public double getIteratorAdvanceCost() {
        return 0;
    }

    @Override
    public boolean isAddConstantTime() {
        return false;
    }
}

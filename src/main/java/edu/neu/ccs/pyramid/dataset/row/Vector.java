package edu.neu.ccs.pyramid.dataset.row;

import edu.neu.ccs.pyramid.dataset.row.function.FloatFloatFunction;
import edu.neu.ccs.pyramid.dataset.row.function.FloatFunction;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.IndexException;

/**
 * Created by Rainicy on 10/31/17
 *
 * Reimplement the Vector from org.apache.mahout.math by
 *
 * 1) change double to float
 * 2)
 */
public interface Vector extends Cloneable{

    /** @return a formatted String suitable for output */
    String asFormatString();

    /**
     * Assign the value to all elements of the receiver
     *
     * @param value a double value
     * @return the modified receiver
     */
    Vector assign(float value);

    /**
     * Assign the values to the receiver
     *
     * @param values a double[] of values
     * @return the modified receiver
     * @throws CardinalityException if the cardinalities differ
     */
    Vector assign(float[] values);

    /**
     * Assign the other vector values to the receiver
     *
     * @param other a Vector
     * @return the modified receiver
     * @throws CardinalityException if the cardinalities differ
     */
    Vector assign(Vector other);

    /**
     * Apply the function to each element of the receiver
     *
     * @param function a FloatFunction to apply
     * @return the modified receiver
     */
    Vector assign(FloatFunction function);

    /**
     * Apply the function to each element of the receiver and the corresponding element of the other argument
     *
     * @param other    a Vector containing the second arguments to the function
     * @param function a FloatFloatFunction to apply
     * @return the modified receiver
     * @throws CardinalityException if the cardinalities differ
     */
    Vector assign(Vector other, FloatFloatFunction function);

    /**
     * Apply the function to each element of the receiver, using the y value as the second argument of the
     * DoubleDoubleFunction
     *
     * @param f a FloatFloatFunction to be applied
     * @param y a double value to be argument to the function
     * @return the modified receiver
     */
    Vector assign(FloatFloatFunction f, double y);

    /**
     * Return the cardinality of the recipient (the maximum number of values)
     *
     * @return an int
     */
    int size();

    /**
     * true if this implementation should be considered dense -- that it explicitly
     *  represents every value
     *
     * @return true or false
     */
    boolean isDense();

    /**
     * true if this implementation should be considered to be iterable in index order in an efficient way.
     * In particular this implies that {@link #all()} and {@link #nonZeroes()} ()} return elements
     * in ascending order by index.
     *
     * @return true iff this implementation should be considered to be iterable in index order in an efficient way.
     */
    boolean isSequentialAccess();

    Vector clone();

    Iterable<Element> all();

    Iterable<Element> nonZeroes();

    /**
     * Return an object of Vector.Element representing an element of this Vector. Useful when designing new iterator
     * types.
     *
     * @param index Index of the Vector.Element required
     * @return The Vector.Element Object
     */
    Element getElement(int index);


    /**
     * Merge a set of (index, value) pairs into the vector.
     * @param updates an ordered mapping of indices to values to be merged in.
     */
    void mergeUpdates(OrderedIntDoubleMapping updates);



    /**
     * A holder for information about a specific item in the Vector. <p>
     * When using with an Iterator, the implementation
     * may choose to reuse this element, so you may need to make a copy if you want to keep it
     */
    interface Element {

        /** @return the value of this vector element. */
        float get();

        /** @return the index of this vector element. */
        int index();

        /** @param value Set the current element to value. */
        void set(double value);
    }

    /**
     * Return a new vector containing the values of the recipient divided by the argument
     *
     * @param x a float value
     * @return a new Vector
     */
    Vector divide(float x);

    /**
     * Return the dot product of the recipient and the argument
     *
     * @param x a Vector
     * @return a new Vector
     * @throws CardinalityException if the cardinalities differ
     */
    float dot(Vector x);

    /**
     * Return the value at the given index
     *
     * @param index an int index
     * @return the double at the index
     * @throws IndexException if the index is out of bounds
     */
    float get(int index);

    /**
     * Return the value at the given index, without checking bounds
     *
     * @param index an int index
     * @return the double at the index
     */
    float getQuick(int index);

    /**
     * Return a new vector containing the element by element difference of the recipient and the argument
     *
     * @param x a Vector
     * @return a new Vector
     * @throws CardinalityException if the cardinalities differ
     */
    Vector minus(Vector x);

    /**
     * Return a new vector containing the normalized (L_2 norm) values of the recipient
     *
     * @return a new Vector
     */
    Vector normalize();


    /**
     * Return the k-norm of the vector. <p/> See http://en.wikipedia.org/wiki/Lp_space <p>
     * Technically, when {@code 0 > power < 1}, we don't have a norm, just a metric, but we'll overload this here. Also supports power == 0 (number of
     * non-zero elements) and power = {@link Double#POSITIVE_INFINITY} (max element). Again, see the Wikipedia page for
     * more info.
     *
     * @param power The power to use.
     */
    double norm(float power);


    /** @return The minimum value in the Vector */
    double minValue();

    /** @return The index of the minimum value */
    int minValueIndex();

    /** @return The maximum value in the Vector */
    double maxValue();

    /** @return The index of the maximum value */
    int maxValueIndex();

    /**
     * Return a new vector containing the sum of each value of the recipient and the argument
     *
     * @param x a float
     * @return a new Vector
     */
    Vector plus(float x);

    /**
     * Return a new vector containing the element by element sum of the recipient and the argument
     *
     * @param x a Vector
     * @return a new Vector
     * @throws CardinalityException if the cardinalities differ
     */
    Vector plus(Vector x);

    /**
     * Set the value at the given index
     *
     * @param index an int index into the receiver
     * @param value a float value to set
     * @throws IndexException if the index is out of bounds
     */
    void set(int index, float value);


    /**
     * Set the value at the given index, without checking bounds
     *
     * @param index an int index into the receiver
     * @param value a float value to set
     */
    void setQuick(int index, float value);


    /**
     * Increment the value at the given index by the given value.
     *
     * @param index an int index into the receiver
     * @param increment sets the value at the given index to value + increment;
     */
    void incrementQuick(int index, double increment);

    /**
     * Return the number of non zero elements in the vector.
     *
     * @return an int
     */
    int getNumNonZeroElements();

    /**
     * Return a new vector containing the product of each value of the recipient and the argument
     *
     * @param x a float argument
     * @return a new Vector
     */
    Vector times(float x);


    /**
     * Return a new vector containing the element-wise product of the recipient and the argument
     *
     * @param x a Vector argument
     * @return a new Vector
     * @throws CardinalityException if the cardinalities differ
     */
    Vector times(Vector x);

    /**
     * Return a new vector containing the subset of the recipient
     *
     * @param offset an int offset into the receiver
     * @param length the cardinality of the desired result
     * @return a new Vector
     * @throws CardinalityException if the length is greater than the cardinality of the receiver
     * @throws IndexException       if the offset is negative or the offset+length is outside of the receiver
     */
    Vector viewPart(int offset, int length);


    /**
     * Return the sum of all the elements of the receiver
     *
     * @return a double
     */
    double zSum();


}

package edu.neu.ccs.pyramid.dataset.row;


import java.io.Serializable;

/**
 * Created by Rainicy on 10/30/17.
 * Reimplement the OrderedIntDoubleMapping from org.apache.mahout.math by
 *
 * 1) change double to float
 * 2) noDefault is removed.
 * 3) remove "merge" function.
 */
public class OrderedIntDoubleMapping implements Serializable, Cloneable{

    static final float DEFAULT_VALUE = 0;


    private int[] indices;
    private float[] values;
    private int numMappings;

    public OrderedIntDoubleMapping() {
        // no-arg constructor for deserializer
        this(11);
    }

    public OrderedIntDoubleMapping(int capacity) {
        indices = new int[capacity];
        values = new float[capacity];
        numMappings = 0;
    }

    public OrderedIntDoubleMapping(int[] indices, float[] values, int numMappings) {
        this.indices = indices;
        this.values = values;
        this.numMappings = numMappings;
    }

    public int indexAt(int offset) {
        return indices[offset];
    }

    public void setIndexAt(int offset, int index) {
        indices[offset] = index;
    }

    public int[] getIndices() {
        return indices;
    }

    void setIndices(int[] indices) {
        this.indices = indices;
    }

    public void setValueAt(int offset, float value) {
        values[offset] = value;
    }

    public float[] getValues() {
        return values;
    }

    void setValues(float[] values) {
        this.values = values;
    }

    public int getNumMappings() {
        return numMappings;
    }

    void setNumMappings(int numMappings) {
        this.numMappings = numMappings;
    }

    private void growTo(int newCapacity) {
        if (newCapacity > indices.length) {
            int[] newIndices = new int[newCapacity];
            System.arraycopy(indices, 0, newIndices, 0, numMappings);
            indices = newIndices;
            float[] newValues = new float[newCapacity];
            System.arraycopy(values, 0, newValues, 0, numMappings);
            values = newValues;
        }
    }

    /**
     * find the offset from indices by the given index
     * @param index
     * @return
     */
    private int find(int index) {
        int low = 0;
        int high = numMappings - 1;
        while (low <= high) {
            int mid = low + (high - low >>> 1);
            int midVal = indices[mid];
            if (midVal < index) {
                low = mid + 1;
            } else if (midVal > index) {
                high = mid - 1;
            } else {
                return mid;
            }
        }
        return -(low + 1);
    }

    /**
     * get the value by the given index
     * @param index
     * @return
     */
    public float get(int index) {
        int offset = find(index);
        return offset >= 0 ? values[offset] : DEFAULT_VALUE;
    }

    /**
     * set the new value by the given index and its value
     * @param index
     * @param value
     */
    public void set(int index, float value) {
        if (numMappings == 0 || index > indices[numMappings - 1]) {
            if (value != DEFAULT_VALUE) {
                if (numMappings >= indices.length) {
                    growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
                }
                indices[numMappings] = index;
                values[numMappings] = value;
                ++numMappings;
            }
        } else {
            int offset = find(index);
            if (offset >= 0) {
                insertOrUpdateValueIfPresent(offset, value);
            } else {
                insertValueIfNotDefault(index, offset, value);
            }
        }
    }

    private void insertOrUpdateValueIfPresent(int offset, float newValue) {
        // TODO: it is not necessary to remove the DEFAULT_VALUE.
        if (newValue == DEFAULT_VALUE) {
            for (int i = offset + 1, j = offset; i < numMappings; i++, j++) {
                indices[j] = indices[i];
                values[j] = values[i];
            }
            numMappings--;
        } else {
            values[offset] = newValue;
        }
    }

    private void insertValueIfNotDefault(int index, int offset, float value) {
        if (value != DEFAULT_VALUE) {
            if (numMappings >= indices.length) {
                growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
            }
            int at = -offset - 1;
            if (numMappings > at) {
                for (int i = numMappings - 1, j = numMappings; i >= at; i--, j--) {
                    indices[j] = indices[i];
                    values[j] = values[i];
                }
            }
            indices[at] = index;
            values[at] = value;
            numMappings++;
        }
    }

    @Override
    public int hashCode() {
        int result = 0;
        for (int i = 0; i < numMappings; i++) {
            result = 31 * result + indices[i];
            result = 31 * result + (int) Double.doubleToRawLongBits(values[i]);
        }
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof OrderedIntDoubleMapping) {
            OrderedIntDoubleMapping other = (OrderedIntDoubleMapping) o;
            if (numMappings == other.numMappings) {
                for (int i = 0; i < numMappings; i++) {
                    if (indices[i] != other.indices[i] || values[i] != other.values[i]) {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder(10 * numMappings);
        for (int i = 0; i < numMappings; i++) {
            result.append('(');
            result.append(indices[i]);
            result.append(',');
            result.append(values[i]);
            result.append(')');
        }
        return result.toString();
    }

    @Override
    public OrderedIntDoubleMapping clone() {
        return new OrderedIntDoubleMapping(indices.clone(), values.clone(), numMappings);
    }

    public void increment(int index, float increment) {
        int offset = find(index);
        if (offset >= 0) {
            float newValue = values[offset] + increment;
            insertOrUpdateValueIfPresent(offset, newValue);
        } else {
            insertValueIfNotDefault(index, offset, increment);
        }
    }
}

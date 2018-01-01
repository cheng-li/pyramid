package edu.neu.ccs.pyramid.util;

import java.util.List;

public interface ParallelStringMapper<T> {
    String mapToString(List<T> list, int i);
}

package edu.neu.ccs.pyramid.util;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * keep the max N items
 * Created by chengli on 2/7/17.
 */
public class BoundedBlockPriorityQueue<E> {
    private PriorityQueue<E> queue;
    private int maxCapacity;
    private Comparator<E> comparator;

    public BoundedBlockPriorityQueue(int maxCapacity, Comparator<E> comparator) {
        this.maxCapacity = maxCapacity;
        this.comparator = comparator;
        this.queue = new PriorityQueue<E>(comparator);
    }

    synchronized public void add(E e){
        if (queue.size()==maxCapacity){
            E head = queue.peek();
            if (comparator.compare(head, e)<0){
                queue.poll();
                queue.add(e);
            }
        } else {
            queue.add(e);
        }
    }

    public E poll(){
        return queue.poll();
    }

    public int size(){
        return queue.size();
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("BoundedBlockPriorityQueue{");
        sb.append("queue=").append(queue);
        sb.append('}');
        return sb.toString();
    }
}

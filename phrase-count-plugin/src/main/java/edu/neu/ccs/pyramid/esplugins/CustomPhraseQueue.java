package edu.neu.ccs.pyramid.esplugins;

/**
 * Created by maoqiuzi on 5/24/17.
 */
import org.apache.lucene.util.PriorityQueue;

final class CustomPhraseQueue extends PriorityQueue<CustomPhrasePositions> {
    CustomPhraseQueue(int size) {
        super(size);
    }

    @Override
    protected final boolean lessThan(CustomPhrasePositions pp1, CustomPhrasePositions pp2) {
        if (pp1.position == pp2.position)
            // same doc and pp.position, so decide by actual term positions. 
            // rely on: pp.position == tp.position - offset. 
            if (pp1.offset == pp2.offset) {
                return pp1.ord < pp2.ord;
            } else {
                return pp1.offset < pp2.offset;
            }
        else {
            return pp1.position < pp2.position;
        }
    }
}

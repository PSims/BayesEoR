/*
 * (c) COPYRIGHT 1999 World Wide Web Consortium
 * (Massachusetts Institute of Technology, Institut National de Recherche
 *  en Informatique et en Automatique, Keio University).
 * All Rights Reserved. http://www.w3.org/Consortium/Legal/
 *
 * $Id: QuickSort.java,v 1.2 2008/02/26 19:28:25 msdemlei Exp $
 */

// See Comparator for why this is here

/**
 * The quicksort algorithm.
 *
 * @version $Revision: 1.2 $
 * @author  Philippe Le H'egaret
 */
public class QuickSort implements Sortable {

    private static QuickSort sort;

    private QuickSort() {
	// nothing to do
    }

    public static QuickSort getInstance() {
	if (sort == null) {
	    sort = new QuickSort();
	}
	return sort;
    }

    private int partition(Object[] objs,
		  int part_low_ind, int part_high_ind, 
		  Comparator comp) {
	int lastsmall;
	int comp1;
	Object median_val;
	Object transit;
	
	// swap median value an first value of array
	comp1 = ( part_low_ind + part_high_ind ) / 2;         
	
	transit = objs[part_low_ind];
	objs[part_low_ind] = objs[comp1];
	objs[comp1] = transit;
	
	median_val = objs[part_low_ind];
	
	lastsmall = part_low_ind;
	for (int i = part_low_ind + 1; i<=part_high_ind; i++) {
	    if (comp.compare(objs[i], median_val)) {
		lastsmall++;
		// swap lastsmall and i
		transit=objs[lastsmall];
		objs[lastsmall]=objs[i];
		objs[i]=transit;
	    }	    
	}
	// swap part_low_ind and lastsmall
	transit=objs[part_low_ind];
	objs[part_low_ind]=objs[lastsmall];
	objs[lastsmall]=transit;
	
	return lastsmall;
    }
    
    /**
     * The quicksort function.
     *
     * @param objs the array with all objects
     * @param start the start offset in the array
     * @param end the end offset in the array
     * @param comp The comparaison function between objects
     */    
    public void sort(Object[] objs,
		     int start, int end, 
		     Comparator comp) {
	if (start < end) {
	    int median = partition(objs, start, end, comp);
	    sort(objs, start, median, comp);
	    sort(objs, median+1, end, comp);
	}
    }
    
    
}

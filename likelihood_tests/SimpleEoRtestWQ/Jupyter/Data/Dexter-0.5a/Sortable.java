/*
 * (c) COPYRIGHT 1999 World Wide Web Consortium
 * (Massachusetts Institute of Technology, Institut National de Recherche
 *  en Informatique et en Automatique, Keio University).
 * All Rights Reserved. http://www.w3.org/Consortium/Legal/
 *
 * $Id: Sortable.java,v 1.2 2008/02/26 19:28:25 msdemlei Exp $
 */


// see Comparator.java for why this is here

/**
 * This interface is only to sort an array with an abstract algorithm.
 *
 * @version $Revision: 1.2 $
 * @author  Philippe Le H'egaret
 */
public interface Sortable {

    /**
     * The sort function.
     *
     * @param objs the array with all objects
     * @param start the start offset in the array
     * @param end the end offset in the array
     * @param comp The comparaison function between objects
     */    
    public void sort(Object[] objs, int start, int end, Comparator comp);
}

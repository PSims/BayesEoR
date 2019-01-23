/*
 * (c) COPYRIGHT 1999 World Wide Web Consortium
 * (Massachusetts Institute of Technology, Institut National de Recherche
 *  en Informatique et en Automatique, Keio University).
 * All Rights Reserved. http://www.w3.org/Consortium/Legal/
 *
 * $Id: Comparator.java,v 1.2 2008/02/26 19:28:25 msdemlei Exp $
 */

// This is in Dexter just because MS still has ancient VMs
// in IE

/**
 * The comparaison function for the Sortable class
 *
 * @version $Revision: 1.2 $
 * @author  Philippe Le H'egaret
 * @see Sortable
 */
public interface Comparator {
    public boolean compare(Object obj1, Object obj2);
}

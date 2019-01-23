/****************************************************************************
 * NCSA HDF                                                                 *
 * National Comptational Science Alliance                                   *
 * University of Illinois at Urbana-Champaign                               *
 * 605 E. Springfield, Champaign IL 61820                                   *
 *                                                                          *
 * For conditions of distribution and use, see the accompanying             *
 * COPYING.ncsa file.                                                       *
 *                                                                          *
 ****************************************************************************/
// tabsize 8


import  java.util.EventListener;

public interface SliderListener extends EventListener {

	public void sliderStateChanged(SliderEvent evt);
}

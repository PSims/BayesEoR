// A class that tries to automate axis marking
//
// Copyright (c) 2000 Markus Demleitner <mdemleitner@head-cfa.harvard.edu>
//  This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
// tabsize=2

import java.awt.*;
import java.util.*;

class AutoAxisFinder extends Recogniser
{	
	double tickRatio=1.5;

	public AutoAxisFinder(ImageWithPoints parent,RecogniserSettings settings)
	{
		super(parent, settings);
	}


	// this is not used and just here because Recogniser wants me to define it
	public synchronized boolean putCoordinate(Point p)
	{	
		this.haveCoos = true;
		this.notify();
		return false;
	}

	// all the analysis I need is already done by the base class
	public boolean analyseIt()
	{	
		return true;
	}


	// takes a Vector of (coo1,coo2,weight) triples and selects two
	// with the largest weights and greatest distance in coo1
	// !! this may destroy points !!
	protected int[] chooseLimits(Vector<int []> points) throws Exception
	{	
		int ct = 0;
		int i, leftpos = 0,rightpos = 0;
		int left = px_w, right = 0;
		int maxWeight = 0, maxWeightPos = 0;

		if (points.isEmpty()) {
			throw new Exception("No ticks found");
		}

		// Compute maximum weight (to find longest ticks)
		for (i=0;i<points.size();i++) {	
			int pt[]=(int[])points.elementAt(i);
			if (pt[2]>maxWeight) {	
				maxWeight = pt[2];
				maxWeightPos = i;
			}
		}

		int thresh = maxWeight-maxWeight/3;
		// count how many points with suffient weight I have
		for (i=0;i<points.size();i++) {	
			int pt[]=(int[])points.elementAt(i);
			if (pt[2]>=thresh) {	
				ct++;
				if (pt[0]<left) {	
					left = pt[0];
					leftpos = i;
				}
				if (pt[0]>right) {	
					right = pt[0];
					rightpos = i;
				}
			}
		}

		if (ct<2) { 
			points.removeElementAt(maxWeightPos);
			return chooseLimits(points);
		}
		return new int[]{leftpos,rightpos};
	}


	protected void lowerAxis() throws Exception
	{	
		Vector<int[]> ticks = new Vector<int[]>();
		int tickStart = -1;
		int tickWeight = 0;
		int meanDenom = 0, meanNumer = 0;
		int axisPos = 0;

		if (axisBottom==px_h) {
			return;
		}
		if (debug) {
			System.out.println("Lower Axis");
		}

		int xstart=axisLeft+(axisRight-axisLeft)/30;
		int ind=xstart+axisBottom*px_w;
		// if I'm not on a black spot, I need to find one
		while (pixels[ind]>blackThresh) {
			ind += px_w;
		}

		// walk along the lower axis, starting left and memorising the tick marks
		Point cp=new Point(ind%px_w,ind/px_w);
		int lastweight = findCB(cp.x+cp.y*px_w).x;
		int axisWeight = lastweight;
		int xlimit=axisRight-(axisRight-axisLeft)/30;
		while (nextY(cp,cp.y,1)) {	
			cp.x++;
			int thisweight = findCB(cp.x+cp.y*px_w).x;
			if (cp.x-xstart<30) {
				if (axisWeight>tickRatio*thisweight) {
					axisWeight = thisweight;
				}
			}
			if (tickStart==-1) {	
				if (thisweight>=tickRatio*axisWeight) {	
					tickStart = cp.x;
					tickWeight = thisweight;
					meanNumer = thisweight*cp.x;
					meanDenom = thisweight;
				} else {
					if (thisweight<3*axisWeight/2) {
						axisPos = cp.y;
					}
				}
			} else {
				if (thisweight<tickRatio*axisWeight) {	
					ticks.addElement(new int[]{(int)Math.round(
						meanNumer/(double)meanDenom), axisPos, tickWeight});
					tickStart = -1;
				} else {	
					if (tickWeight<thisweight)
						tickWeight = thisweight;
					meanNumer += thisweight*cp.x;
					meanDenom += thisweight;
				}
			}
			lastweight = thisweight;
			if (cp.x>xlimit)
				break;
		}

		int inds[] = chooseLimits(ticks);
		int p1[] = ticks.elementAt(inds[0]);
		int p2[] = ticks.elementAt(inds[1]);
		parent.setGauge(new Point(p1[0], p1[1]), new Point(p2[0], p2[1]));
	}


	protected void leftAxis() throws Exception
	{	
		Vector<int[]> ticks=new Vector<int[]>();
		int tickStart=-1;
		int tickWeight=0;
		int meanNumer=0,meanDenom=0;
		int axisPos=0;

		if (axisLeft==0)
			return;
		if (debug)
			System.out.println("Left Axis");

		int ystart=axisBottom-(axisBottom-axisTop)/30;
		int ind=axisLeft+ystart*px_w;
		// if I'm not on a black spot, I need to find one
		while (pixels[ind]>blackThresh) {
			ind--;
		}

		// walk up along the left axis and memorising the tick marks
		Point cp=new Point(ind%px_w,ind/px_w);
		int res[] = findxCB(cp.x+cp.y*px_w);
		int thisweight = res[0];
		int targx = res[3];
		int lastweight = thisweight;
		int axisWeight = lastweight;
		int ylimit=axisTop+(axisBottom-axisTop)/30;
		while (nextX(cp,targx,-1)) {	
			cp.y--;
			res = findxCB(cp.x+cp.y*px_w);
			thisweight = res[0];
			targx = res[3];
			if (ystart-cp.y<30)
				if (axisWeight>tickRatio*thisweight)
					axisWeight = thisweight;
			if (tickStart==-1) {	
				if (thisweight>=tickRatio*axisWeight) {	
					tickStart = cp.y;
					tickWeight = thisweight;
					meanNumer = thisweight*cp.y;
					meanDenom = thisweight;
				} else {
					if (thisweight<=3*axisWeight/2) {
				 		axisPos = cp.x;
					}
				}
			}
			else {
				if (thisweight<tickRatio*axisWeight) {	
					ticks.addElement(new int[]{(int)Math.round(
						meanNumer/(double)meanDenom), axisPos, tickWeight});
					tickStart = -1;
				} else {	
					if (tickWeight<thisweight) {
						tickWeight = thisweight;
					}
					meanNumer += thisweight*cp.y;
					meanDenom += thisweight;
				}
			}
			lastweight = thisweight;
			if (cp.y<ylimit) {
				break;
			}
		}

		int inds[] = chooseLimits(ticks);
		int p1[] = ticks.elementAt(inds[0]);
		int p2[] = ticks.elementAt(inds[1]);
		parent.setGauge(new Point(p1[1], p1[0]),new Point(p2[1], p2[0]));
	}


	public void recogniseIt() throws Exception
	{	
		try {	
			this.lowerAxis();
		} catch (Exception e) { 
		}
		try { 
			this.leftAxis(); 
		}	catch (Exception e) { 
		}
	}
}


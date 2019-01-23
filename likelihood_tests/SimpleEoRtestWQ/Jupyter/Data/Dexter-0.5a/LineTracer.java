// A recogniser module for Dexter: Trace a line and set a number of
// points on it.  A naive approach
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


class LineTracer extends Recogniser
{	Point startPoint;
	int yThresh=3;	//drop points when more than yThresh below or above max/min
	int pointSpacing;


	public LineTracer(ImageWithPoints parent, RecogniserSettings settings)
	{
		super(parent,settings);
		pointSpacing = settings.getIntProp("LineTracerSpacing");
		if (pointSpacing<0 || (pointSpacing>1 && pointSpacing<5))
		{	System.out.println("Invalid point Spacing -- resetting to 0");
			pointSpacing = 0;
		}
	}


	public boolean followLine(int dx, boolean drawFirst) throws Exception
	{	
		Point curpoint=new Point(startPoint);
		Point lastmax=null, lastmin=null;
		int targy=curpoint.y;
		int lastXDrawn;
		int lastdY=0, dY, lastY=curpoint.y;

		lastXDrawn = curpoint.x;
		if (!debug && drawFirst) {
			parent.addPoint(curpoint);
		}
		while (nextY(curpoint, targy, dx)) {	
			if (debug) {	
				System.out.println("----"+curpoint.x+" "+curpoint.y+" "+lastdY+" "+
					(lastY-curpoint.y)+" la"+lastmax+" li"+lastmin);
			}
			if (this.stopMe) {
				throw new Exception("Ignore");
			}
			
			// drop a point if the function does something exciting or if I feel
			// it's too long since the last one
			dY = lastY-curpoint.y;
			if (!debug && ((Math.abs(lastXDrawn-curpoint.x)>pointSpacing) || 
				(lastdY!=0 && Math.abs(lastdY-dY)>3))) {	
				parent.addPoint(curpoint);
				lastXDrawn = curpoint.x;
			}
			else {
				if (debug)
					System.out.println("s"+curpoint);
			}

			if (lastmax!=null && lastmax.y-curpoint.y>yThresh) {	
				parent.addPoint(lastmax);
				lastmax = null;
				if (debug)
					System.out.println("s"+lastmax);
			}
			if (lastmin!=null && curpoint.y-lastmin.y>yThresh) {	
				parent.addPoint(lastmin);
				lastmin = null;
				if (debug) {
					System.out.println("s"+lastmax);
				}
			}

			if (lastdY*dY<=0)
			{	if (lastdY>0)	// a minimum (->maximum on paper)
				{	if (lastmin==null)
						lastmin = new Point(curpoint);
					else
						if (lastmin.y>curpoint.y)
							lastmin = new Point(curpoint);
				}
				if (lastdY<0)	// a maximum
				{	if (lastmax==null)
						lastmax = new Point(curpoint);
					else
						if (lastmax.y<curpoint.y)
							lastmax = new Point(curpoint);
				}
			}
			lastdY = dY;
			lastY = curpoint.y;
			targy = lastY-dY*dx;
			curpoint.x += dx;
			if (curpoint.x<=axisLeft||curpoint.x>=axisRight||curpoint.y<axisTop||
				curpoint.y>axisBottom)
				break;
		}
		if (!debug)
			parent.addPoint(new Point(curpoint.x-dx,lastY));
		return true;
	}


	public synchronized boolean putCoordinate(Point p)
	{	
		startPoint = p;
		this.notify();
		return false;
	}


	// home in on a good starting point
	protected boolean analyseIt()
	{	int ind=startPoint.x+startPoint.y*px_w;

		yThresh = px_h/50+1;

		if (pixels[ind]>blackThresh)
		{	int stopat=Math.min(ind+20,px_max);
			for (;ind<stopat;ind++)
				if (pixels[ind]<=blackThresh)
					break;
			
			if (pixels[ind]>blackThresh)
			{	ind=startPoint.x+startPoint.y*px_w;
				stopat = Math.max(0,ind-20);
				for (;ind>stopat;ind--)
					if (pixels[ind]<=blackThresh)
						break;
			}

			if (pixels[ind]>blackThresh)
			{	ind=startPoint.x+startPoint.y*px_w;
				stopat = Math.max(0,ind-20*px_w);
				for (;ind>stopat;ind-=px_w)
					if (pixels[ind]<=blackThresh)
						break;
			}
					
			if (pixels[ind]>blackThresh)
			{	ind=startPoint.x+startPoint.y*px_w;
				stopat = Math.min(px_max,ind+20*px_w);
				for (;ind<stopat;ind+=px_w)
					if (pixels[ind]<=blackThresh)
						break;
			}

			if (pixels[ind]>blackThresh)
				return false;

			startPoint.x = ind%px_w;
			startPoint.y = ind/px_w;
		}
	
		// XXX Need a better idea here
		if (pointSpacing==0)
			pointSpacing = px_w/20;

		return true;
	}


	protected void recogniseIt() throws Exception
	{	followLine(-1,true);
		followLine(+1,false);
	}

}

// A recogniser module for Dexter: Find points resembling a user-selected
// template.  A naive approach
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

class Feature
{	byte[] tPixels;
	int tWidth,tHeight;
	int hheights[],vwidths[];
	int maxWidth,maxHeight;
	Point markPos;
	boolean debug=false;
	int blackThresh;


	public Feature(Rectangle bbox,PointFinder parent)
	{	
		this(bbox,parent,0);
	}


	public Feature(Rectangle bbox,PointFinder parent,int pad)
	{
		bbox.x -= pad;
		bbox.y -= pad;
		bbox.width += 2*pad;
		bbox.height += 2*pad;
		tWidth = bbox.width;
		tHeight = bbox.height;
		tPixels = new byte[tWidth*tHeight];
		blackThresh = parent.blackThresh;
		collect(bbox,parent.pixels,parent.px_w,parent.px_h);
		computeProps();

	}


	// collect all the negative pixel values in bbox into feature
	public void collect(Rectangle bbox,byte pixels[],int px_w,int px_h)
	{ int ind;

		for (int y=0;y<bbox.height;y++)
		{	ind = (bbox.y+y)*px_w+bbox.x;
			for (int x=0;x<bbox.width;x++,ind++)
				if (pixels[ind]<0)
				{	tPixels[x+y*bbox.width] = (byte)(-pixels[ind]-1);
					pixels[ind] = 127;
				}
				else
					tPixels[x+y*bbox.width] = 127;
		}
	}

	
	private void computeWidths()
	{	int tmp,ind,stopat;

		vwidths = new int[tHeight];
		maxWidth = 0;
		for (int i=0;i<tHeight;i++)
		{	ind = i*tWidth;
			stopat = (i+1)*tWidth;
			while (ind<stopat&&tPixels[ind]>blackThresh)
				ind++;
			int left = ind%tWidth;
			ind = (i+1)*tWidth-1;
			stopat = i*tWidth;
			while (ind>stopat&&tPixels[ind]>blackThresh)
				ind--;
			int right = ind%tWidth;
			if (left<right)
				tmp = right-left+1;
			else
				tmp = 0;
			if (tmp>maxWidth)
				maxWidth = tmp;
			vwidths[i] = tmp;
		}
	}
	

	private void computeHeights()
	{	int tmp,ind,stopat;

		hheights = new int[tWidth];
		maxHeight = 0;
		for (int i=0;i<tWidth;i++)
		{	ind = i;
			stopat = tHeight*tWidth;
			while (ind<stopat&&tPixels[ind]>blackThresh)
				ind += tWidth;
			int upper = ind/tWidth;
			ind = tWidth*(tHeight-1)+i;
			stopat = 0;
			while (ind>=stopat&&tPixels[ind]>blackThresh)
				ind -= tWidth;
			int lower = ind/tWidth;
			if (upper<lower)
				tmp = lower-upper+1;
			else
				tmp = 0;
			if (tmp>maxHeight)
				maxHeight = tmp;
			hheights[i] = tmp;
		}
	}


	// compute the width of an error bar from a width array, return -1 if
	// there doesn't seem to be one
	// The idea is that if there are many small widths and a few large ones,
	// the small ones will represent the error bar.
	int getErrbarWidth(int widthary[])
	{	int mean=0;
		int maxWidth=0;

		for (int i=0;i<widthary.length;i++)
		{	mean += widthary[i];
			if (widthary[i]>maxWidth)
				maxWidth = widthary[i];
		}
		mean = Math.round(mean/(float)widthary.length);
		mean++;

		int aboveMean=0,belowMean=0;
		for (int i=0;i<widthary.length;i++)
			if (widthary[i]<mean)
				belowMean++;
			else
				aboveMean++;

		if (belowMean<=2*aboveMean)
			return -1;

		int errMeanNumer = 0;
		int errMeanDenom = 0;
		for (int i=0;i<widthary.length;i++)
			if (widthary[i]<mean)
			{	errMeanNumer += widthary[i];
				errMeanDenom++;
			}

		if (errMeanDenom==0)
			return -1;
		else
		{	int errBarWidth = Math.round(errMeanNumer/(float)errMeanDenom);
			if (errBarWidth>maxWidth/2)
				return -1;
			else
				return errBarWidth;
		}
	}


	// isolate a point from an error bar in a width array.
	// Strategy: Guess if there are error bars at all.  If I think there are
	// any, look for the fattest guy around, start there and gobble up 
	// everthing that's loosely connected to it
	protected int[] findBlob(int widthary[])
	{	int errbarWidth=getErrbarWidth(widthary);

		if (errbarWidth<1)
			return new int[]{0,widthary.length};

		int maxPos=0,maxVal=-1;
		for (int i=0;i<widthary.length;i++)
			if (widthary[i]>maxVal)
			{	maxPos = i;
				maxVal = widthary[i];
			}

		int featureLimit = Math.max(widthary.length/15,2);
		int errbarLimit = errbarWidth+(maxVal-errbarWidth)/4;
		int left=Math.max(0,maxPos-1);
		for (int curPos=maxPos-1;curPos>=0;curPos--)
		{	if (left-curPos>featureLimit)
				break;
			if (widthary[curPos]>errbarLimit)
				left = curPos;
		}

		int right=Math.min(widthary.length,maxPos+1);
		for (int curPos=maxPos+1;curPos<widthary.length;curPos++)
		{	if (curPos-right>featureLimit)
				break;
			if (widthary[curPos]>errbarLimit)
				right = curPos;
		}

		return new int[]{left,right};
	}


	// try to guess where the marking shold be w.r.t. to
	// (0,0) of the bounding box.  For the time being, I take
	// the mean of the lines and columns that are close to maximal
	// width or height 
	private void computeMarkPos()
	{ int meanNum=0;
		int meanDenom=0;
		int limit;

		int dd[] = findBlob(vwidths);
		limit = maxWidth-maxWidth/2;
		for (int i=dd[0];i<dd[1];i++)
			if (vwidths[i]>=limit)
			{	meanNum += i*vwidths[i];
				meanDenom += vwidths[i];
			}
		int yofs = Math.round(meanNum/(float)meanDenom);

		meanNum=0;
		meanDenom=0;
		limit = maxHeight-maxHeight/2;
		dd = findBlob(hheights);
		for (int i=dd[0];i<dd[1];i++)
			if (hheights[i]>=limit)
			{	meanNum += i*hheights[i];
				meanDenom += hheights[i];
			}

		int xofs = Math.round(meanNum/(float)meanDenom);

		markPos = new Point(xofs,yofs);
	}


	// compute some properties of the feature, most notably,
	// the location of the point
	protected void computeProps()
	{
		computeWidths();
		computeHeights();
		computeMarkPos();
	}

}


class Template extends Feature
{	Rectangle matchBox;

	public Template(Rectangle bbox,PointFinder parent)
	{
		super(bbox,parent);
	}


	// distance between me and another feature, with offsetting
	public double dist(Feature other,int dx,int dy) 
		throws ArrayIndexOutOfBoundsException
	{	int othery=other.markPos.y-(markPos.y-matchBox.y)+dy;
		int otherx=other.markPos.x-(markPos.x-matchBox.x)+dx;
		int myind=matchBox.y*tWidth+matchBox.x;
		int otherind=othery*other.tWidth+otherx;
		int sum=0;

		if (otherx<0 || othery<0)
			return 1d/0d;

		for (int y=0;y<matchBox.height;y++)
		{	for (int x=0;x<matchBox.width;x++)
				if (((tPixels[myind+x]>blackThresh)^
					(other.tPixels[otherind+x]>blackThresh)))
					sum++;
			myind += tWidth;
			otherind += other.tWidth;
		}
		
		return sum/(double)(matchBox.height*matchBox.width);
	}


	// distance between me and another feature, with autohoming
	public double[] dist(Feature other,int dx,int dy,int recursionLevel) 
		throws Exception
	{	
		if (recursionLevel>8)
			throw new Exception("Recursion too deep");
		double here = dist(other,dx,dy);
		if (here>1)
			return new double[]{here,dx,dy};
		double right = dist(other,dx+1,dy);
		double left = dist(other,dx-1,dy);
		double up = dist(other,dx,dy-1);
		double down = dist(other,dx,dy+1);
		double mm=Math.min(right,Math.min(left,Math.min(up,down)));
		if (mm<here)
		{	if (mm==right)
				return dist(other,dx+1,dy,recursionLevel++);
			if (mm==left)
				return dist(other,dx-1,dy,recursionLevel++);
			if (mm==up)
				return dist(other,dx,dy-1,recursionLevel++);
			if (mm==down)
				return dist(other,dx,dy+1,recursionLevel++);
		}
		return new double[]{here,dx,dy};
	}


	public double[] dist(Feature other)
	{
		try
		{	return dist(other,0,0,0);
		} catch (Exception e)
		{	return new double[]{1f/0f,0,0}; }
	}


	protected void computeMatchBox()
	{	int[] blobd;
		int upper=0,lower=tHeight,left=0,right=tWidth;

		blobd = findBlob(vwidths);
		if (blobd[0]>0 && blobd[0]<markPos.y)
			upper = blobd[0];
		if (blobd[1]>0 && blobd[1]>markPos.y)
			lower = blobd[1];

		blobd = findBlob(hheights);
		if (blobd[0]>0 && blobd[0]<markPos.x)
			left = blobd[0];
		if (blobd[1]>0 && blobd[1]>markPos.x)
			right = blobd[1];

		matchBox = new Rectangle(left,upper,right-left,lower-upper);
		if (debug)
			System.out.println("mb "+matchBox);
	}


	protected void computeProps()
	{
		super.computeProps();
		computeMatchBox();
	}
}


class PointFinder extends Recogniser
{	Point startPoint;
	double distThresh;
	Template markerTemplate;


	public PointFinder(ImageWithPoints parent,RecogniserSettings settings)
	{
		super(parent,settings);
		distThresh = settings.getDoubleProp("PointFinderThresh");
	}


	public synchronized boolean putCoordinate(Point p)
	{	
		startPoint = p;
		this.notify();
		return false;
	}


	// This does a line-oriented flood fill starting at index ind,
	// returning a bounding box of the filled area;  visited fields get
	// negative
	// semi-nasty recursive approach
	private void floodFill(int ind,Rectangle bbox)
	{
		int y=ind/px_w;
		if (y<bbox.y)
			bbox.y = y;
		if (y>bbox.y+bbox.height)
			bbox.height = y-bbox.y;

		//walk to the left end of the structure
		int stopat=ind-ind%px_w;
		while (ind>=stopat&&pixels[ind]>=0&&pixels[ind]<blackThresh)
			ind--;
		if (pixels[ind]>=blackThresh)
			ind++;

		if (bbox.x>ind%px_w)
			bbox.x = ind%px_w;

		//now walk right, memorising where I have to go up and down
		stopat=ind+(ind-ind%px_w);
		int lineDown = px_w;
		int lineUp = -px_w;
		Vector<Integer> upInds=new Vector<Integer>(5);
		Vector<Integer> downInds=new Vector<Integer>(5);
		if (ind+lineDown>=px_max)
			lineDown = 0;
		if (ind+lineUp<0)
			lineUp = 0;
		boolean checkDown = true;
		boolean checkUp = true;
		while (ind<stopat&&pixels[ind]>=0&&pixels[ind]<blackThresh)
		{	pixels[ind] = (byte)(-pixels[ind]-1); // pixels<=127 guaranteed
			if (pixels[ind+lineDown]>=0&&pixels[ind+lineDown]<blackThresh)
				if (checkDown)
				{	downInds.addElement(new Integer(ind+lineDown));
					checkDown = false;
				}
			else
				checkDown = true;
			if (pixels[ind+lineUp]>=0&&pixels[ind+lineUp]<blackThresh)
				if (checkUp)
				{	upInds.addElement(new Integer(ind+lineUp));
					checkUp = false;
				}
			else
				checkUp = true;
			ind++;
		}

		if (ind%px_w>bbox.x+bbox.width)
			bbox.width = ind%px_w-bbox.x;

		Enumeration<Integer> cts=upInds.elements();
		while (cts.hasMoreElements())
			floodFill(cts.nextElement().intValue(),bbox);
		cts=downInds.elements();
		while (cts.hasMoreElements())
			floodFill(cts.nextElement().intValue(),bbox);
	}


	// Clear every negative pixel within bbox
	protected void clearNegPix(Rectangle bbox)
	{	
		for (int y=bbox.y;y<bbox.y+bbox.height;y++)
		{	int ind=y*px_w+bbox.x;
			int stopat=ind+bbox.width;
			for (;ind<stopat;ind++)
				if (pixels[ind]<0)
					pixels[ind] = 127;
		}
	}
		

	// look for a template starting on the black spot p
	protected void findTemplate(Point p)
	{	Rectangle bbox=new Rectangle(p.x,p.y,1,1);
		int ind=p.x+p.y*px_w;

		floodFill(ind,bbox);
		if (debug)
		{	System.out.println("tpl-bbox"+bbox);
			parent.getGraphics().drawRect(bbox.x,bbox.y,bbox.width,bbox.height);
		}
		markerTemplate = new Template(bbox,this);
		parent.addPoint(new Point(bbox.x+markerTemplate.markPos.x,
			bbox.y+markerTemplate.markPos.y));
	}


	// this function starts on a black pixel and tries to match the
	// markerTemplate somewhere on black pixels connected with it
	protected void checkMatch(int ind)
	{	double dist[];
		Feature feat=null;

		Rectangle bbox=new Rectangle(ind%px_w,ind/px_w,1,1);
		floodFill(ind,bbox);
		if (bbox.width>px_w/2 || bbox.height>px_h/2)
		{	if (debug)
				System.out.println(bbox+" too large -- skipping");
			clearNegPix(bbox);
			return;
		}
		if (debug)
			System.out.println("Checking "+bbox);
		try
		{	feat=new Feature(bbox,this,2);
			dist = markerTemplate.dist(feat);
		} catch (ArrayIndexOutOfBoundsException e)
		{	return; }
		if (debug)
		{	System.out.println("d="+dist[0]+", x="+(bbox.x+feat.markPos.x+dist[1])+
				", y="+(bbox.y+feat.markPos.y+dist[2]));
			parent.getGraphics().drawRect(bbox.x,bbox.y,bbox.width,bbox.height);
		}
		if (dist[0]<distThresh)
			parent.addPoint(new Point(bbox.x+feat.markPos.x+(int)Math.round(dist[1]),
				bbox.y+feat.markPos.y+(int)Math.round(dist[2])));
	}


	// look for a black spot in the vicinity of the point described
	// by the byte index ind and return a corrected index, or -1 if
	// we found none
	protected int findBlack(int ind)
	{	int stopat;
		
		if (pixels[ind]>blackThresh)
		{	stopat=Math.min(ind+20,px_max);
			for (;ind<stopat;ind++)
				if (pixels[ind]<=blackThresh)
					break;
		}
			
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
			return -1;

		return ind;
	}


	// look for the template in the vicinity of
	// the point the user clicked on
	protected boolean analyseIt()
	{	int ind=findBlack(startPoint.x+startPoint.y*px_w);

		if (ind==-1)
				return false;

		startPoint.x = ind%px_w;
		startPoint.y = ind/px_w;
		if (debug)
			System.out.println("Start Point "+startPoint);

		findTemplate(new Point(startPoint));
		return true;
	}


	protected void recogniseIt() throws Exception
	{	int ind;

		for (int y=axisTop;y<axisBottom;y++)
		{	ind = y*px_w+axisLeft;
			for (int x=axisLeft;x<axisRight;x++,ind++)
				if (pixels[ind]>=0&&pixels[ind]<blackThresh)
					checkMatch(ind);
		}
	}


}

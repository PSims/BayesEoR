// An abstract class with utility functions for Recognisers, plus a
// bit of interface.
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
import java.awt.image.*;
import java.util.*;

// an auxilliary class for handling grayscale images.  Since java
// doesn't have unsigned bytes, the grayscales returned are between
// 0 (black) and 127 (white).  For our purposes, that's still more 
// than enough.  The MSb is used by recognizers to mark stuff they've
// already seen.
class ByteGrabber implements ImageConsumer 
{	
	ImageProducer prod;
	byte pixels[];
	int px_w;
	int readPixels = 0;
	boolean errorOccurred=false, pixelsComplete=false;
	public void setDimensions(int w,int h) 
	{
	}

	public void setProperties(Hashtable<?,?> props) {}
	public void setHints(int hintflags) {}
	public void setColorModel(ColorModel model) {}

	public synchronized byte[] retrievePixels(Image im, int w, int h)
		throws Exception
	{ 
		this.prod = im.getSource();
		this.px_w = w;
		this.pixels = new byte[w*h];
		int expectPixels = w*h;
		this.prod.startProduction(this);
		try {	
			while (!this.pixelsComplete) {
				this.wait();
			}
		}	catch (InterruptedException e) {
		}
		if (this.errorOccurred)
			throw new Exception("Could not retrieve pixels");
		else
			return this.pixels;
	}


	public synchronized void imageComplete(int s) 
	{ 
		switch (s) { 
			case STATICIMAGEDONE:
			case SINGLEFRAMEDONE:
				this.errorOccurred = false;
				break;
			default:
			case IMAGEERROR:
			case IMAGEABORTED:
				this.errorOccurred = true;
				break;
		}
		this.pixelsComplete = true;
		this.prod.removeConsumer(this);
		this.notifyAll();
	}
	
	public synchronized void setPixels(int x, int y, int w, int h,
		ColorModel model, byte pix[], int off, int scansize)
	{	
		for (int dy=0; dy<h; dy++) { 
			int si=off+dy*scansize;
			int di=(y+dy)*this.px_w+x;
			for (int dx=0; dx<w; dx++,si++,di++) {
				this.pixels[di] = (byte)(model.getRed(pix[si])/2);
				this.readPixels++;
			}
		}
	}

	// I should go through the ColorModel, I guess, but for the
	// forseeable future just masking should be ok.
	public synchronized void setPixels(int x, int y, int w, int h,
		ColorModel model, int pix[], int off, int scansize)
	{	
		for (int dy=0; dy<h; dy++) { 
			int si=off+dy*scansize;
			int di=(y+dy)*px_w+x;
			for (int dx=0; dx<w; dx++,si++,di++) {
				this.pixels[di] = (byte)((pix[si]&0xff)/2);
			}
		}
	}
}


public abstract class Recogniser extends Thread
{ 
	ImageWithPoints parent;
	Image im;
	int px_h, px_w, px_max;
	byte pixels[];
	Image deb_buf;
	Graphics deb_gc;
	boolean debug=false;
	int blackThresh=110;	//everything below is considered black
	int weightThresh=50;	//abandon line when weight drops below this
	int axisLeft,axisRight,axisTop,axisBottom;
	boolean stopMe=false;
	boolean haveCoos=false;
	RecogniserSettings settings;

	// this function is used to communicate coordinates set by the user to
	// the recogniser -- it should return true when it wants more coordinates
	public abstract boolean putCoordinate(Point p);

	// this function is called to do the actual recognising work
	protected abstract void recogniseIt() throws Exception;

	// this is a hook in which recognisers can do some analysis of
	// the image up front -- it should return false or better raise an
	// exception with an explanation if the recognition 
	// cannot be performed
	protected abstract boolean analyseIt() throws Exception;


	public Recogniser(ImageWithPoints parent,RecogniserSettings settings)
	{	
		this.parent = parent;
		this.settings = settings;
		blackThresh= settings.getIntProp("blackThresh");
		weightThresh = settings.getIntProp("weightThresh");
	}


	// Computes the center of the black distribution on the vertical
	// defined by the pixel index ind (i.e. somthing like x+y*px_w.)
	// Returns a quadruple with the 
	// cumulative weight (all black in column), the numerator
	// for a weighted mean, and the maximum weight (the latter is
	// used to dodge antialiasing artifacts), and the maximal y value
	// considered (to avoid checking pixels twice)
	// We really shouldn't be using Rectangles for that -- what was
	// I thinking?
	protected final Rectangle findCB(int ind)
	{	
		int pos=0, cumweight=0;
		int y=ind/px_w;
		int weight;
		int maxweight=0;

		// walk down until white or end of picture
		if (pixels[ind]<=blackThresh) { 
			for (;ind<px_max; ind+=px_w, y++) {
				if (pixels[ind]>blackThresh) { 
					ind -= px_w;
					y--;
					break;
				}
			}
		}
		// back up if there was black all the way down
		if (ind>=px_max) {
			ind -= px_w;
		}
		// walk up again until we hit white, collecting distribution info
		for (;ind>=0&&(pixels[ind]<=blackThresh); ind-=px_w, y--) {	
			weight = 127-pixels[ind];
			if (weight>maxweight)
				maxweight = weight;
			pos += y*weight;
			cumweight += weight;
		}
		if (maxweight>120)
			maxweight = 127;  // That's a bad hack.  Feel free to remove it.
				// Except that this seems to improve tracing of thin lines.
		return new Rectangle(cumweight, pos, maxweight, y);
	}

	// as findCB, only in x direction
	protected final int[] findxCB(int ind)
	{	int pos=0, cumweight=0;
		int x=ind%px_w, leftx;
		int weight;
		int maxweight=0;

		int xlimit=ind-ind%px_w; // index of start of current line
		// walk left until white or start of line
		if (pixels[ind]<=blackThresh) {
			for (;ind>xlimit; ind--, x--) {
				if (pixels[ind]>blackThresh) { 
					ind++;
					x++;
					break;
				}
			}
		}
		leftx = x;
		xlimit = xlimit+px_w;  // index of start of line
		// walk right until white or end of line
		for (;ind<xlimit&&(pixels[ind]<=blackThresh); ind++, x++) {	
			weight = 127-pixels[ind];
			if (weight>maxweight)
				maxweight = weight;
			pos += x*weight;
			cumweight += weight;
		}
		if (maxweight>120)
			maxweight = 127;  // That's a bad hack.  Feel free to remove it.
				// Except that this seems to improve tracing of thin lines.
		return new int[]{cumweight, pos, maxweight, leftx, x};
	}



	// start at starting point and try to identify where
	// the line is going by checking the adjacent pixels.
	protected boolean nextY(Point curpoint,int targy,int dx)
	{	int y=curpoint.y;
		int ind;
		Rectangle res;
		int maxweight=0,maxpos=0,maxblack=0;
		int seenThrough=px_h;
		int whiteCount=0;
		int whiteLim=3;
		int blackHeight;
		int inity=y;

		if (curpoint.x==px_w-1 || curpoint.x==0)
			return false;

		if (debug)
			System.out.println("y target"+targy);
		// walk to the bottom of the black zone in current column,
		// accepting a little white in between.
		for (ind=y*px_w+curpoint.x; ind<px_max; ind+=px_w, y++) {	
			if (pixels[ind]>blackThresh)
				whiteCount++;
			else
				whiteCount = 0;
			if (whiteCount>=whiteLim) {	
				ind -= px_w;
				y--;
				break;
			}
		}
		// back up if we've left the image
		while (ind>px_max) {
			ind -= px_w;
		}
		// walk up again, looking at the column dx distant 
		// (i.e., the left or right column) some black stuff
		blackHeight = 0;
		whiteCount = -1;
		for (;ind>0; ind-=px_w, y--) {	
			// only run findCB if it hasn't already looked at these pixels
			if (y>=seenThrough)  
				continue;
			if (pixels[ind]>blackThresh)
				whiteCount++;
			else
				whiteCount = 0;
			if (whiteCount>=whiteLim)
				break;   // stop if there's too much white
			blackHeight++;

			if (pixels[ind+dx]<blackThresh) {	
				// Let's see where the center of the current blackness is
				res = findCB(ind+dx);
				seenThrough = res.height; // we don't want to check twice
				if (res.width>maxblack || (res.width==maxblack && 
						Math.abs(res.y/res.x-targy)<Math.abs(maxpos-targy))) {	
						maxweight = res.x;
						maxpos = res.y/res.x;
						maxblack = res.width;
					}
			}
		}
		
		if (y>inity)
			System.out.println("Didn't reach start point at x="+curpoint.x);
		if (blackHeight>px_h/2 || maxweight/100>px_h/2) {	
			if (debug)
				System.out.println("Axis Alert");
			return false;
		}
		if (maxweight>weightThresh) {	
			curpoint.y = maxpos;
			return true;
		}
		return false;
	}

	// as nextY, only in x direction
	protected boolean nextX(Point curpoint,int targx,int dy)
	{	int x=curpoint.x;
		int ind;
		int res[];
		int maxweight=0,maxpos=0,maxblack=0;
		int seenThrough=0;
		int whiteCount=0;
		int whiteLim=3;
		int blackWidth;
		int initx=x;

		dy *= px_w;
		if (curpoint.y==px_h-1 || curpoint.y==0)
			return false;

		if (debug)
			System.out.println("xt"+targx);
		//walk to the left of the black zone
		ind=curpoint.y*px_w+x;
		int xlimit=ind%px_w;
		for (;ind>xlimit;ind--,x--)
		{	if (pixels[ind]>blackThresh)
				whiteCount++;
			else
				whiteCount = 0;
			if (whiteCount>=whiteLim)
			{	ind++;
				x++;
				break;
			}
		}

		// walk right again, looking at the column dy remote to find 
		// some black stuff
		xlimit = ind-ind%px_w+px_w;
		blackWidth = 0;
		whiteCount = -1;
		for (;ind<xlimit;ind++,x++)
		{	if (pixels[ind]>blackThresh)
				whiteCount++;
			else
				whiteCount = 0;
			if (whiteCount>=whiteLim)
				break;
			blackWidth++;
			if (x<=seenThrough)
				continue;

			if (pixels[ind+dy]<blackThresh)
			{	res = findxCB(ind+dy);
				seenThrough = res[4];
				if (res[2]>maxblack || (res[2]==maxblack && 
						Math.abs(res[1]/res[0]-targx)<Math.abs(maxpos-targx)))
					{	maxweight = res[0];
						maxpos = res[1]/res[0];
						maxblack = res[2];
					}
			}
		}
		
		if (x<initx)
			System.out.println("Didn't reach start point at y="+curpoint.y);
		if (blackWidth>px_w/2 || maxweight/100>px_w/2)
		{	if (debug)
				System.out.println("Axis Alert");
			return false;
		}
		if (maxweight>weightThresh)
		{	curpoint.x = maxpos;
			return true;
		}
		return false;
	}


	private int xwidth(int ind)
	{	int w=0;
		int limLeft=ind-ind%px_w;
		int limRight=limLeft+px_w;

		while (pixels[ind]<blackThresh && ind>limLeft)
			ind--;
		ind++;
		while (pixels[ind++]<blackThresh && ind<limRight)
			w++;
		return w;
	}

	// preloads the image, then
	// tries to locate the axes and does some analysis of the image
	// (much TBD), finally calls analyseIt, a hook in which individual
	// recognisers can do their own preliminaria
	protected boolean analyse() throws Exception
	{
		px_w = im.getWidth(parent);
		px_h = im.getHeight(parent);
		if (px_w==-1 || px_h==-1)
			throw new Exception(
				"Please wait until image \nis completely retrieved");
		px_max = px_w*px_h;
		pixels = new ByteGrabber().retrievePixels(im, px_w, px_h);

		axisRight = px_w;
		axisBottom = px_h;
		axisLeft = 0;
		axisTop = 0;

		int ind;
		// axis on the left
		for (int ystart=px_h/6;ystart<px_h;ystart+=px_h/6) {	
			int stopat = ystart*px_w+px_w/4;
			for (ind=ystart*px_w;ind<stopat;ind++) {	
				if (pixels[ind]<=blackThresh)
					break;
			}
			if (pixels[ind]<=blackThresh) {	
				Rectangle r=findCB(ind);
				if (r.width==0 || r.x/r.width<px_h/4)
					r = findCB(++ind);
				if (r.width!=0 && r.x/r.width>px_h/4) {	
					int width = 0;
					for (;pixels[ind]<=blackThresh; ind++) {
						width++;
					}
					axisLeft = ind%px_w+width;
					break;
				}
			}
		}

		// bottom axis 
		for (int xstart=px_w/6; xstart<px_w; xstart+=px_w/6) {	
			int stopat = (px_h-px_h/4)*px_w;
			for (ind=xstart+(px_h-1)*px_w; ind>stopat; ind-=px_w) {	
				if (pixels[ind]<=blackThresh)
					break;
			}
			if (pixels[ind]<=blackThresh) {	
				int w=xwidth(ind);
				if (w<px_w/2) {	
					ind -= px_w;
					w = xwidth(ind);
				}
				if (w>px_w/2) {	
					int height = 0;
					for (;pixels[ind]<=blackThresh; ind-=px_w)
						height++;
					axisBottom = ind/px_w-height;
					break;
				}
			}
		}

		// axis on right edge
		int stopat = (px_h/2)*px_w-px_w/4;
		for (ind=px_h/2*px_w-1; ind>stopat; ind--) {	
			if (pixels[ind]<=blackThresh)
				break;
		}
		if (pixels[ind]<=blackThresh) {	
			Rectangle r=findCB(ind);
			if (r.width==0 || r.x/r.width<px_h/4)
				r = findCB(++ind);
			if (r.width!=0 && r.x/r.width>px_h/4) {	
				int width = 0;
				for (;pixels[ind]<=blackThresh;ind--)
					width++;
				axisRight = ind%px_w-width;
			}
		}

		// the axis at the top
		stopat = (px_h/4)*px_w;
		for (ind=px_w/2;ind<stopat;ind+=px_w) {	
			if (pixels[ind]<=blackThresh)
				break;
		}
		if (pixels[ind]<=blackThresh) {	
			int w=xwidth(ind);
			if (w<px_w/2) {	
				ind += px_w;
				w = xwidth(ind);
			}
			if (w>px_w/2) {	
				int height = 0;
				for (;pixels[ind]<=blackThresh;ind+=px_w)
					height++;
				axisTop = ind/px_w+height;
			}
		}

		if (debug)
			System.out.println("detected axis"+axisLeft+" "+axisRight+" "+
				axisTop+" "+axisBottom);

		return analyseIt();
	}

	
	public synchronized void stopRecogniser()
	{	
		stopMe = true;
		this.notify();
	}


	protected synchronized boolean waitForCoos()
	{ 
		if (haveCoos)
			return true;
		try {	
			wait(); 
		}	catch (InterruptedException e) {	
			return false; 
		}
		return true;
	}


	public void run()
	{ 
		try { 
		try {
		try {
			if (!waitForCoos() || stopMe)
				throw new Exception("Ignore");

			im = parent.getRecImage();
		
			if (analyse()) {	
				if (stopMe)
					throw new Exception("Ignore");
				recogniseIt();
			}
			else {
				throw new Exception("Could not find start object");
			}
		
			if (debug)
				System.out.println("Recogniser ends");
		} catch (OutOfMemoryError e) { 
			new AlertBox(parent.parent,"Recogniser failure",
			"Out of memory.\nPlease try recognising at a lower resolution");}
		} catch (Exception e) { 
			if (!e.getMessage().equals("Ignore"))
				if (e.getClass().toString().equals("class java.lang.Exception"))
					new AlertBox(parent.parent,"Recogniser failure", e.getMessage());
				else
					new AlertBox(parent.parent,"Recogniser failure", e.toString()+
						"\nYou shouldn't get such an insensible message.\n"+
						"Please report how you managed to provoke it.");
			e.printStackTrace();
		}
		} finally {	
			parent.recogniserStopped(); 
		}
	}

}

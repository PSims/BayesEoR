// A class that represents gauges on an image.
//
// Copyright (c) 2000 Markus Demleitner
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
import java.awt.event.*;


class Gauge
{	
	Point start, end;
	public boolean horiz;
	Color hcolor,vcolor;
	Rectangle startListen=null,endListen=null;
	int listenSize=6;
	int iAmDragging=0; //1 dragging start, 2 dragging end
	ImageWithPoints parent;
	boolean horizLock=false; //when people start to drag, I don't change my gender

	Gauge(ImageWithPoints parent,int sx, int sy, int ex, int ey)
	{ 
		this.parent = parent;
		setStart(sx, sy);
		setEnd(ex, ey);
		hcolor = new Color(255, 0, 0);
		vcolor = new Color(0, 0, 255);
		sethoriz();
	}


	void sethoriz()
	{
		if (horizLock || start==null || end==null)
			return;
		if (java.lang.Math.abs(start.x-end.x)>java.lang.Math.abs(start.y-end.y))
			horiz = true;
		else
			horiz = false;
	}


	public void setStart(int ex,int ey)
	{
		start = new Point(ex,ey);
		startListen = new Rectangle(ex-listenSize/2, ey-listenSize/2,
			listenSize, listenSize);
		sethoriz();
	}


	public void setEnd(int ex, int ey)
	{	
		end = new Point(ex,ey);
		endListen = new Rectangle(ex-listenSize/2, ey-listenSize/2, 
			listenSize, listenSize);
		sethoriz();
	}


	public void normalise()
	{	
		if (this.horiz) {	
			if (this.start.x>this.end.x) {	
				Point tmp = this.start;
				this.setStart(this.end.x, this.end.y);
				this.setEnd(tmp.x, tmp.y);
			}
		}
		else {	
			if (this.start.y<this.end.y) {	
				Point tmp = this.start;
				this.setStart(this.end.x, this.end.y);
				this.setEnd(tmp.x, tmp.y);
			}
		}
	}


	public synchronized void paint(Graphics g)
	{	
		int dx, dy;
		double scale;

		if (horiz) {
			g.setColor(this.hcolor);
		} else {
			g.setColor(this.vcolor);
		}
		g.drawLine(this.start.x, this.start.y, this.end.x, this.end.y);
		dx = this.start.x-this.end.x;
		dy = this.start.y-this.end.y;
		scale = 10/Math.sqrt(dx*dx+dy*dy);
		g.drawLine((int)Math.round(this.start.x-dy/2*scale),
			(int)Math.round(this.start.y+dx/2*scale),
			(int)Math.round(this.start.x+dy/2*scale),
			(int)Math.round(this.start.y-dx/2*scale));
		g.drawLine((int)java.lang.Math.round(this.end.x-dy/2*scale),
			(int)Math.round(this.end.y+dx/2*scale),
			(int)Math.round(this.end.x+dy/2*scale),
			(int)Math.round(this.end.y-dx/2*scale));
	}


	public Rectangle bbox()
	{	
		int x0, y0, w, h;

		x0 = Math.min(this.start.x, this.end.x);
		y0 = Math.min(this.start.y, this.end.y);
		w = Math.abs(this.start.x-this.end.x);
		h = Math.abs(this.start.y-this.end.y);
		return new Rectangle(x0-10, y0-10, w+20, h+20);
	}


	public double length()
	{	int dx, dy;
	
		dx = this.start.x-this.end.x;
		dy = this.start.y-this.end.y;
		return Math.sqrt(dx*dx+dy*dy);
	}


	public void newScale(int oldscale, int newscale)
	{
		this.start.x = Math.round(this.start.x*oldscale/(float)newscale);
		this.start.y = this.start.y*oldscale/newscale;
		this.end.x = this.end.x*oldscale/newscale;
		this.end.y = this.end.y*oldscale/newscale;
	}


	public void mousePressed(MouseEvent e)
	{
		if (this.startListen!=null) {
			if (this.startListen.contains(e.getPoint())) {
				this.iAmDragging = 1;
				this.horizLock = true;
				e.consume();
			}
		}
		if (this.endListen!=null) {
			if (this.endListen.contains(e.getPoint())) {	
				this.iAmDragging = 2;
				horizLock = true;
				e.consume();
			}
		}
	}


	public void mouseReleased(MouseEvent e)
	{	
		if (this.iAmDragging!=0) {	
			this.mouseDragged(e);
			e.consume();
			this.iAmDragging = 0;
			this.parent.setGauge(this);
		}
	}

	public void mouseDragged(MouseEvent e)
	{	
		if (this.iAmDragging==0) {
			return;
		}
		Rectangle obb = this.bbox();
		if (this.iAmDragging==1) {
			this.setStart(e.getX(), e.getY());
		}
		if (this.iAmDragging==2) {
			this.setEnd(e.getX(), e.getY());
		}
		this.parent.makeImage(this.bbox().union(obb));
	}
}

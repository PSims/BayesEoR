// This class models a data point for ImageWithPoints, i.e., it has
// error bars,  knows how to draw itself and can delegate transformation 
// to physical coordinates.  One day I'll MVC this, really :-)
//
// Copyright (c) 2004 Markus Demleitner
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

public class Datapoint implements Comparator
{	
	protected Point center;
	protected Point leftBar=null, rightBar=null, upBar=null;
	protected Point downBar=null, tmpBar=null;
	protected Color mycolor;
	protected Rectangle listenRect;
	protected Rectangle bbox;
	private ImageWithPoints parent;
	protected Datapoint me=this;  // For mouse handlers' use
	protected boolean iAmDragging=false;
	protected boolean eraseMe=false;

	final Color tmpBarColor=new Color(255,0,255);
	final int crossSize=6;
	final int listenSize=4;

	public Datapoint(int x, int y, Color mycolor, ImageWithPoints parent)
	{	
		this.center = new Point(x, y);
		this.mycolor = mycolor;
		this.parent = parent;
		this.listenRect = new Rectangle(x-this.listenSize, y-this.listenSize,
			this.listenSize*2+1, this.listenSize*2+1);
		this.computeBbox();
	}


	// Event handlers for removing and setting error bars


	public void mouseClicked(MouseEvent e) 
	{	
		if (!this.listenRect.contains(e.getPoint()) || e.isConsumed()) {
			return;
		}
		if (e.isShiftDown() ||
				((e.getModifiers()&InputEvent.BUTTON2_MASK)
					==InputEvent.BUTTON2_MASK)) {	
			this.parent.removePoint(this.me);
			e.consume();
		}
	}

	public void mouseDragged(MouseEvent e) 
	{	
		if (!this.iAmDragging) {
			return;
		}
		Graphics g=parent.getGraphics();
		try { 
			e.consume();
			if (g!=null) {	
				g.setXORMode(Color.white);
				g.setColor(this.tmpBarColor);
				if (this.eraseMe)
					g.drawLine(this.center.x, this.center.y, 
						this.tmpBar.x, this.tmpBar.y);
				this.eraseMe = false;
			}
			this.tmpBar = this.parent.makeParallelToAxes(this, e.getPoint());
			if (g!=null) {	
				g.drawLine(this.center.x, this.center.y, 
					this.tmpBar.x, this.tmpBar.y);
				this.eraseMe = true;
			}
			this.computeBbox();
		} finally {	
			g.dispose(); 
		}
	}

	public void mousePressed(MouseEvent e)
	{	
		if (this.parent.hgauge==null || this.parent.vgauge==null || 
			!this.listenRect.contains(e.getPoint()) || e.isConsumed()) {
			return;
		}
		this.iAmDragging = true;
		this.eraseMe = false;
		e.consume();
	}

	public void mouseReleased(MouseEvent e)
	{	
		if (this.iAmDragging && this.tmpBar!=null) {	
			this.iAmDragging = false;
			int dx = this.center.x-this.tmpBar.x;
			int dy = this.center.y-this.tmpBar.y;
			if (Math.abs(dx)>Math.abs(dy)) {
				if (dx<0)
					this.rightBar = this.tmpBar;
				else
					this.leftBar = this.tmpBar;
			} else {
				if (dy<0)
					this.downBar = this.tmpBar;
				else
					this.upBar = this.tmpBar;
			}
			this.tmpBar = null;
			this.computeBbox();
			this.eraseMe = false;
			this.parent.makeImage(this.bbox);
			e.consume();
		}
	}

	// Utility methods

	private void computeBbox()
	{	
		Rectangle bbox=new Rectangle(this.center.x-this.crossSize, 
			this.center.y-this.crossSize, this.crossSize*2+1, this.crossSize*2+1);
		if (this.leftBar!=null)
			this.bbox.add(this.leftBar);
		if (this.rightBar!=null)
			this.bbox.add(this.rightBar);
		if (this.upBar!=null)
			this.bbox.add(this.upBar);
		if (this.downBar!=null)
			this.bbox.add(this.downBar);
		if (this.tmpBar!=null)
			bbox.add(tmpBar);
		this.bbox = new Rectangle(bbox.x-1, bbox.y-1,
			bbox.width+2, bbox.height+2);
	}


	public void newScale(int oldscale, int newscale)
	{	
		this.center.x = Math.round(this.center.x*oldscale/(float)newscale);
		this.center.y = Math.round(this.center.y*oldscale/(float)newscale);
		this.listenRect = new Rectangle(
			this.center.x-this.listenSize, this.center.y-this.listenSize,
			this.listenSize*2+1, this.listenSize*2+1);
		this.computeBbox();
	}


	public synchronized void paint(Graphics g)
	{	
		g.setPaintMode();
		g.setColor(mycolor);
		g.drawLine(this.center.x-this.crossSize, this.center.y-this.crossSize, 
			this.center.x+this.crossSize, this.center.y+this.crossSize);
		g.drawLine(this.center.x+this.crossSize, this.center.y-this.crossSize,
			this.center.x-this.crossSize, this.center.y+this.crossSize);
		if (this.upBar!=null)
			g.drawLine(this.center.x, this.center.y, 
				this.upBar.x, this.upBar.y);
		if (this.downBar!=null)
			g.drawLine(this.center.x, this.center.y, 
				this.downBar.x, this.downBar.y);
		if (this.leftBar!=null)
			g.drawLine(this.center.x, this.center.y, 
				this.leftBar.x, this.leftBar.y);
		if (this.rightBar!=null)
			g.drawLine(this.center.x, this.center.y, 
				this.rightBar.x, this.rightBar.y);
		g.setXORMode(Color.white);
		g.setColor(this.tmpBarColor);
		if (this.tmpBar!=null)
			g.drawLine(this.center.x, this.center.y, 
				this.tmpBar.x, this.tmpBar.y);
		g.setPaintMode();
	}

	// public interface (ImageWithPoints manipulates me directly)

	// returns the point's center in physical coordiates
	public DoublePoint getPhysicalCenter()
	{
		return this.parent.transformPhysicalToLogical(this.center);
	}

	public int getScrX()
	{
		return this.center.x;
	}

	public int getScrY()
	{
		return this.center.y;
	}

	public Point getCenter()
	{
		return this.center;
	}

	// these guys compare according to the transformed abscissa value
	// (which is what we need when preparing data for output).  This
	// unfortunately also means that we can't compare them until some
	// embedding object knows how to transform.  Plus, we don't compare
	// very efficiently.  Let's hope java's sort doesn't need too many
	// comparisions.
//	public int compare(Object o1, Object o2)
//	{
//		Point2D.Double myCenter=((Datapoint)o1).getPhysicalCenter();
//		Point2D.Double otherCenter=((Datapoint)o2).getPhysicalCenter();
//		
//		return new Double(myCenter.getX()).compareTo(
//			new Double(otherCenter.getX()));
//	}
//
// And here's a quick hack while we're waiting for j2 adoption


	public boolean compare(Object o1, Object o2)
	{
		double myX=((Datapoint)o1).getPhysicalCenter().getX();
		double otherX=((Datapoint)o2).getPhysicalCenter().getX();
		
		return myX<otherX;
	}

	private double getPhysError(Point bar, DoublePoint physCenter)
	{
		if (bar==null) {
			return 0.0;
		} else {
			DoublePoint physBar = this.parent.transformPhysicalToLogical(bar);
			// the following works because one summand will be zero within
			// machine accuracy
			return physBar.getX()-physCenter.getX()+
				physBar.getY()-physCenter.getY();
		}
	}
				
	public String getRepr(boolean showXErrors, boolean showYErrors,
		int sigFigX, int sigFigY)
	{
		StringBuffer resultLine = new StringBuffer();
		DoublePoint physCenter = this.getPhysicalCenter();
		double pcx = physCenter.getX(), pcy = physCenter.getY();

		resultLine.append(Fmt.fmt(pcx, 4, sigFigX)+"\t"+
				Fmt.fmt(pcy, 4, sigFigY));
		if (showXErrors) {
			resultLine.append("\t"+Fmt.fmt(this.getPhysError(this.leftBar,
				physCenter), 4, sigFigX));
			resultLine.append("\t"+Fmt.fmt(this.getPhysError(this.rightBar,
				physCenter), 4, sigFigX));
		}
		if (showYErrors) {
			resultLine.append("\t+"+Fmt.fmt(this.getPhysError(this.downBar,
				physCenter), 4, sigFigX));
			resultLine.append("\t+"+Fmt.fmt(this.getPhysError(this.upBar,
				physCenter), 4, sigFigX));
		}
		resultLine.append("\n");
		return resultLine.toString();
	}

	public boolean hasHorizErrBars()
	{
		return this.leftBar!=null || this.rightBar!=null;
	}

	public boolean hasVertErrBars()
	{
		return this.upBar!=null || this.downBar!=null;
	}
}


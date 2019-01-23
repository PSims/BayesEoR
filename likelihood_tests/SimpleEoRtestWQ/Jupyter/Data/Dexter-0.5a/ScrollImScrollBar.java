// Another ScrollPane-like thing.  ScrollPane gave me all kinds of weird
//  error message when I tried what I'm doing here.
//
// Copyright (c) 2000,2003 Markus Demleitner <msdemlei@cl.uni-heidelberg.de>
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


// This class provides an auto-scrolling image on which a rubber band
// can be drawn.  On releasing the mouse, the parent will be notified
// of the selection.
class ScrollImWithRubber extends Canvas
	implements MouseListener,MouseMotionListener
{	
	Image embeddedImage;
	ScrollImScrollBar parent;
	boolean imageAvailable=false;
	boolean eraseit = false;
	Point bandStart, bandEnd;
	boolean onIndexedDisplay;
	Point leftUpper, lastLeftUpper;
	int im_lastupdate=0;
	static final long serialVersionUID=20060308L;
	
	public ScrollImWithRubber(ScrollImScrollBar parent,
		Image embeddedImage, boolean onIndexedDisplay)
	{	
		this.parent = parent;
		this.embeddedImage = embeddedImage;
		this.onIndexedDisplay = onIndexedDisplay;
		if (this.embeddedImage.getHeight(this)!=-1) {
			this.imageAvailable = true;
			repaint();
		}

		bandStart = new Point(0,0);
		bandEnd = new Point(0,0);
		leftUpper = new Point(0,0);
		lastLeftUpper = new Point(0,0);

		addMouseListener(this);
		addMouseMotionListener(this);
		if (this.imageAvailable) {
			this.doLayout();
		}
	}

	public void mousePressed(MouseEvent e)
	{	
		if (eraseit) {	
			rubberDraw();
			eraseit = false;
		}
		bandStart = toCanvCoo(new Point(e.getX(), e.getY()));
		e.consume();
	}

	public void mouseReleased(MouseEvent e)
	{
		bandEnd = toCanvCoo(new Point(e.getX(), e.getY()));
		selectionFinished();
		e.consume();
	}

	public void mouseDragged(MouseEvent e)
	{
		if (eraseit) {	
			rubberDraw();
		}
		bandEnd = toCanvCoo(new Point(e.getX(), e.getY()));
		rubberDraw();
		scrollCheck();
		e.consume();
	}

	public void mouseClicked(MouseEvent e) {}
	public void mouseEntered(MouseEvent e) {}
	public void mouseExited(MouseEvent e) {}
	public void mouseMoved(MouseEvent e) {}

	protected Point toCanvCoo(Point p)
	{	
		p.translate(-leftUpper.x, -leftUpper.y);
		return p;
	}

	protected synchronized void drawBand(Graphics g)
	{	
		int x,y,w,h;

		g.setXORMode(Color.white);
		g.setColor(Color.black);
		x = Math.min(bandStart.x, bandEnd.x);
		y = Math.min(bandStart.y, bandEnd.y);
		w = Math.abs(bandStart.x-bandEnd.x);
		h = Math.abs(bandStart.y-bandEnd.y);
		g.drawRect(x,y,w,h);
		lastLeftUpper = new Point(leftUpper.x, leftUpper.y);
	}
		
	protected synchronized void rubberDraw()
	{ 
		Graphics g=getGraphics();
		
		if (g!=null)
		{	
			try {	
				g.translate(leftUpper.x,leftUpper.y);
				drawBand(g);
				eraseit = true;
			} finally { 
				g.dispose();
			}
		}
	}

	public void selectionFinished()
	{	int x0, x1, y0, y1;

		x0 = bandStart.x;
		x1 = bandEnd.x;
		y0 = bandStart.y;
		y1 = bandEnd.y;
		if (true) {	
			int width=Math.abs(x0-x1), height=Math.abs(y0-y1);
			Rectangle selectedRegion;
			if (x0>x1) {
				x0 = x1;
			}
			if (y0>y1) {
				y0 = y1;
			}
			selectedRegion = new Rectangle(x0, y0, width, height);
			parent.notifySelection(selectedRegion);
		}
	}

	public void setx0(int x0)
	{	
		this.leftUpper.x=this.cropPoint(new Point(-x0, 0)).x;
		repaint();
	}

	public int getx0()
	{
		return this.leftUpper.x;
	}

	public void sety0(int y0)
	{
		this.leftUpper.y=this.cropPoint(new Point(0, -y0)).y;
		repaint();
	}

	public int gety0()
	{
		return this.leftUpper.y;
	}

	protected Rectangle getVisibleArea()
	{	
		return new Rectangle(this.leftUpper, this.getSize());
	}

	protected void scrollCheck()
	{	
		Rectangle vis = getVisibleArea();
		Point cm = bandEnd;
		boolean changed = false;

		if (cm.x<-leftUpper.x+20) {	
			leftUpper.translate(+20,0);
			changed = true;
		}
		if (cm.x>-vis.x+vis.width-20) {	
			leftUpper.translate(-20,0);
			changed = true;
		}
		if (cm.y<-leftUpper.y+20) {	
			leftUpper.translate(0,+20);
			changed = true;
		}
		if (cm.y>-vis.y+vis.height-20) {	
			leftUpper.translate(0,-20);
			changed = true;
		}

		if (changed) {	
			leftUpper = cropPoint(leftUpper);
			if (!leftUpper.equals(lastLeftUpper)) {	
				repaint();
				parent.updateScrollbars();
			}
		}
	}

	// crops p to something that's a possible leftUpper
	protected Point cropPoint(Point p)
	{	
		int maxx = this.getPreferredSize().width-this.getSize().width;
		int maxy = this.getPreferredSize().height-this.getSize().height;

		if (maxx>0 && p.x<-maxx)
			p.x = -maxx;
		if (maxy>0 && p.y<-maxy)
			p.y = -maxy;
		if (maxx<=0)
			p.x = 0;
		if (maxy<=0)
			p.y = 0;
		if (p.x>0)
			p.x = 0;
		if (p.y>0)
			p.y = 0;
		return p;
	}

	public synchronized void paint(Graphics g)
	{
		if (!this.imageAvailable) {
			g.drawString("Retrieving image. Please stand by...", 10, 10);
		}
		if (eraseit) {	
			Graphics f = g.create();
			f.translate(lastLeftUpper.x, lastLeftUpper.y);
			drawBand(f);
			f.dispose();
		}
		g.translate(leftUpper.x, leftUpper.y);
		g.drawImage(this.embeddedImage, 0, 0, this);
		if (eraseit) {	
			this.drawBand(g);
		}
	}

	public Dimension getPreferredSize()
	{
		if (this.imageAvailable) {
			return new Dimension(this.embeddedImage.getWidth(this), 
				this.embeddedImage.getHeight(this));
		} else {
			return new Dimension(400, 400);
		}
	}

	public synchronized boolean imageUpdate(Image im, int infoflags,
		int x, int y, int width, int height)
	{	
		if ((infoflags&java.awt.image.ImageObserver.SOMEBITS)!=0) {
			if ((y+height-im_lastupdate<100)) {
				return true;
			}
			Graphics g=getGraphics();

			if (g!=null) {	
				try {	
					g.drawString("Retrieving image. Please stand by...", 10, 10);
					g.clipRect(x, im_lastupdate,width, y+height-im_lastupdate);
					g.drawImage(im, 0, 0, this);
					this.im_lastupdate = y+height;
				}
				finally {	
					g.dispose();
				}
			}
		}
		if ((infoflags&java.awt.image.ImageObserver.ALLBITS)!=0) {
			this.imageAvailable = true;
			this.doLayout();
			this.repaint();
			this.parent.notifyChangedSize();
			return false;
		}
		return true;
	}
}


// This is a hacked replacement for the Scrollbar part of ScrollPane
// What a pain.
class ScrollImScrollBar extends Panel
	implements AdjustmentListener
{	
	ScrollImWithRubber child;
	Scrollbar vAdjustable, hAdjustable;
	MainServices parent;
	boolean skipSelection;
	static final long serialVersionUID=20060308L;
	
	public ScrollImScrollBar(MainServices parent, Image theim,
		boolean onIndexedDisplay, boolean skipSelection)
	{	
		super();
		this.parent = parent;
		this.skipSelection = skipSelection;
		this.makeContents(theim, onIndexedDisplay);
		this.parent.notifyChangedSize();
		if (this.skipSelection) {
			this.parent.notifySelection(null);
		}
	}

	protected void makeContents(Image theim,
		boolean onIndexedDisplay)
	{
		this.vAdjustable = new Scrollbar(Scrollbar.VERTICAL);
		this.hAdjustable = new Scrollbar(Scrollbar.HORIZONTAL);
		this.child = new ScrollImWithRubber(this, theim, onIndexedDisplay);
		this.setLayout(new BorderLayout());
		this.add("Center", child);
		this.add("South", hAdjustable);
		this.add("East", vAdjustable);
		this.hAdjustable.addAdjustmentListener(this);
		this.vAdjustable.addAdjustmentListener(this);
		this.doLayout();
		validate();
	}

	public void notifySelection(Rectangle bbox)
	{
		parent.notifySelection(bbox);
	}

	void resizeHoriz()
	{	
		if (this.child==null || this.hAdjustable==null) {
			return;
		}
		int canvasWidth = this.child.getSize().width;
		int imWidth = this.child.getPreferredSize().width;

		this.child.setx0(0);
		this.hAdjustable.setValues(0, canvasWidth, 0, imWidth);
	}

	void resizeVert()
	{	
		if (this.child==null || this.vAdjustable==null) {
			return;
		}
		int canvasHeight = this.child.getSize().height;
		int imHeight = this.child.getPreferredSize().height;
		
		this.child.sety0(0);
		this.vAdjustable.setValues(
			this.child.gety0(), canvasHeight, 0, imHeight);
	}

	public void doLayout()
	{
		this.resizeHoriz();
		this.resizeVert();
		super.doLayout();
	}

	public void updateScrollbars()
	{	
		hAdjustable.setValue(-child.getx0());
		vAdjustable.setValue(-child.gety0());
	}

	public void notifyChangedSize()
	{
		doLayout();
		this.parent.notifyChangedSize();
	}

	public void adjustmentValueChanged(AdjustmentEvent e)
	{	
		if (e.getAdjustable()==vAdjustable)
			child.sety0(e.getValue());
		else if (e.getAdjustable()==hAdjustable)
			child.setx0(e.getValue());
	}
}

// vi:ts=2:
